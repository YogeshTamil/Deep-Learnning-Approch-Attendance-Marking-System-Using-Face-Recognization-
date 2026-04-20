import os
import io
import threading
import sqlite3
import json
import pandas as pd
import time
import datetime
from datetime import UTC, date, timedelta
import pytz  # Added for timezone handling

LAST_ATTENDANCE = {}
COOLDOWN_SECONDS = 5  # Set to 60 seconds (1 minute) for production stability
# Set to 'Asia/Kolkata' for IST (UTC+5:30)
LOCAL_TIMEZONE = 'Asia/Kolkata'

from flask import Flask, render_template, request, jsonify, send_file, abort
# Ensure you have your model imports
from model import train_model_background, extract_embedding_for_image, MODEL_PATH

APP_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(APP_DIR, "attendance.db")
DATASET_DIR = os.path.join(APP_DIR, "dataset")
os.makedirs(DATASET_DIR, exist_ok=True)

TRAIN_STATUS_FILE = os.path.join(APP_DIR, "train_status.json")

app = Flask(__name__, static_folder="static", template_folder="templates")


# ---------- DB helpers ----------
def init_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    c = conn.cursor()
    # Updated table creation to include all necessary student fields
    c.execute("""CREATE TABLE IF NOT EXISTS students (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    roll TEXT,
                    class TEXT,
                    section TEXT,
                    reg_no TEXT,
                    created_at TEXT
                )""")
    c.execute("""CREATE TABLE IF NOT EXISTS attendance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    student_id INTEGER,
                    name TEXT,
                    timestamp TEXT
                )""")
    conn.commit()
    conn.close()


init_db()


# ---------- Train status helpers ----------
def write_train_status(status_dict):
    try:
        with open(TRAIN_STATUS_FILE, "w") as f:
            json.dump(status_dict, f)
    except Exception as e:
        app.logger.error(f"Error writing train status: {e}")


def read_train_status():
    if not os.path.exists(TRAIN_STATUS_FILE):
        return {"running": False, "progress": 0, "message": "Not trained"}
    try:
        with open(TRAIN_STATUS_FILE, "r") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return {"running": False, "progress": 0, "message": "Corrupt status file"}


# ensure initial train status file exists
write_train_status({"running": False, "progress": 0, "message": "No training yet."})


# ---------- Routes ----------
@app.route("/")
def index():
    return render_template("index.html")


# Dashboard simple API for attendance stats (last 30 days)
@app.route("/attendance_stats")
def attendance_stats():
    import pandas as pd
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    df = pd.read_sql_query("SELECT timestamp FROM attendance", conn)
    conn.close()
    if df.empty:
        days = [(datetime.date.today() - timedelta(days=i)).strftime("%d-%b") for i in range(29, -1, -1)]
        return jsonify({"dates": days, "counts": [0] * 30})

    # Timezone processing for plotting consistency
    df['dt_object'] = pd.to_datetime(df['timestamp'], format='ISO8601', errors='coerce', utc=True)
    df['dt_object'] = df['dt_object'].dt.tz_convert(LOCAL_TIMEZONE)
    df['date'] = df['dt_object'].dt.date

    last_30 = [(datetime.date.today() - timedelta(days=i)) for i in range(29, -1, -1)]
    counts = [int(df[df['date'] == d].shape[0]) for d in last_30]
    dates = [d.strftime("%d-%b") for d in last_30]
    return jsonify({"dates": dates, "counts": counts})


# -------- Add student (form) --------
@app.route("/add_student", methods=["GET", "POST"])
def add_student():
    if request.method == "GET":
        return render_template("add_student.html")

    # POST: save student metadata and return student_id
    data = request.form
    name = data.get("name", "").strip()
    roll = data.get("roll", "").strip()
    cls = data.get("class", "").strip()
    sec = data.get("sec", "").strip()
    reg_no = data.get("reg_no", "").strip()

    if not name or not roll or not reg_no:
        return jsonify({"error": "Name, Roll, and Registration No. are required"}), 400

    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    c = conn.cursor()
    now = datetime.datetime.now(UTC).isoformat()
    c.execute("INSERT INTO students (name, roll, class, section, reg_no, created_at) VALUES (?, ?, ?, ?, ?, ?)",
              (name, roll, cls, sec, reg_no, now))
    sid = c.lastrowid
    conn.commit()
    conn.close()

    # create dataset folder for this student
    os.makedirs(os.path.join(DATASET_DIR, str(sid)), exist_ok=True)
    return jsonify({"student_id": sid})


# -------- Upload face images (after capture) --------
@app.route("/upload_face", methods=["POST"])
def upload_face():
    student_id = request.form.get("student_id")
    if not student_id:
        return jsonify({"error": "student_id required"}), 400
    files = request.files.getlist("images[]")
    saved = 0
    folder = os.path.join(DATASET_DIR, student_id)
    if not os.path.isdir(folder):
        os.makedirs(folder, exist_ok=True)
    for f in files:
        try:
            fname = f"{datetime.datetime.now(UTC).timestamp():.6f}_{saved}.jpg"
            path = os.path.join(folder, fname)
            f.save(path)
            saved += 1
        except Exception as e:
            app.logger.error("save error: %s", e)
    return jsonify({"saved": saved})


# -------- Train model (start background thread) --------
@app.route("/train_model", methods=["GET"])
def train_model_route():
    # if already running, respond accordingly
    status = read_train_status()
    if status.get("running"):
        return jsonify({"status": "already_running"}), 202
    # reset status
    write_train_status({"running": True, "progress": 0, "message": "Starting training"})
    # start background thread
    t = threading.Thread(target=train_model_background, args=(DATASET_DIR, lambda p, m: write_train_status(
        {"running": True, "progress": p, "message": m})))
    t.daemon = True
    t.start()
    return jsonify({"status": "started"}), 202


# -------- Train progress (polling) --------
@app.route("/train_status", methods=["GET"])
def train_status():
    return jsonify(read_train_status())


# -------- Mark attendance page --------
@app.route("/mark_attendance", methods=["GET"])
def mark_attendance_page():
    return render_template("mark_attendance.html")


# -------- Recognize face endpoint (POST image) --------
@app.route("/recognize_face", methods=["POST"])
def recognize_face():
    # 1. INITIAL CHECKS AND ML PREDICTION LOGIC
    if "image" not in request.files:
        return jsonify({"recognized": False, "error": "no image"}), 400
    img_file = request.files["image"]

    try:
        emb = extract_embedding_for_image(img_file.stream)
        if emb is None:
            return jsonify({"recognized": False, "error": "no face detected"}), 200

        # Attempt prediction
        from model import load_model_if_exists, predict_with_model
        clf = load_model_if_exists()
        if clf is None:
            return jsonify({"recognized": False, "error": "model not trained"}), 200

        pred_label, conf = predict_with_model(clf, emb)

        # Threshold confidence
        if conf < 0.5:
            return jsonify({"recognized": False, "confidence": float(conf)}), 200

        student_id = int(pred_label)

    except Exception as e:
        app.logger.exception("recognize error during ML prediction")
        return jsonify({"recognized": False, "error": str(e)}), 500

    # ----------------------------------------------------------------------
    # 2. DEBOUNCE / COOLDOWN CHECK (Uses global variable, prevents duplicate writes)
    # ----------------------------------------------------------------------
    global LAST_ATTENDANCE
    current_time = time.time()

    if student_id in LAST_ATTENDANCE and (current_time - LAST_ATTENDANCE[student_id] < COOLDOWN_SECONDS):
        # Prevent rapid duplicate logging
        return jsonify({"recognized": True, "student_id": student_id, "name": "Cooldowned", "status": "cooldown"}), 200

    # ----------------------------------------------------------------------
    # 3. DATABASE ACCESS (Quickly open, read, write, commit, and close)
    # ----------------------------------------------------------------------
    conn = None
    try:
        # Open connection for name lookup and insertion
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        c = conn.cursor()

        # A. Find student name
        c.execute("SELECT name FROM students WHERE id=?", (student_id,))
        row = c.fetchone()
        name = row[0] if row else "Unknown"

        # B. Save attendance record
        ts = datetime.datetime.now(UTC).isoformat()
        c.execute("INSERT INTO attendance (student_id, name, timestamp) VALUES (?, ?, ?)", (student_id, name, ts))

        # C. COMMIT AND CLOSE IMMEDIATELY
        conn.commit()

        # D. Update cooldown tracker after successful write
        LAST_ATTENDANCE[student_id] = current_time

        return jsonify({"recognized": True, "student_id": student_id, "name": name, "confidence": float(conf)}), 200

    except sqlite3.OperationalError:
        app.logger.error("recognize error: database is locked")
        return jsonify({"recognized": True, "student_id": student_id, "name": name,
                        "status": "lock_failure"}), 500

    except Exception as e:
        app.logger.exception("recognize error during DB operation")
        return jsonify({"recognized": False, "error": str(e)}), 500

    finally:
        if conn:
            conn.close()


@app.route("/attendance_record", methods=["GET"])
def attendance_record():
    period = request.args.get("period", "all")  # all, daily, weekly, monthly
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)

    # --- STEP 1: Update SQL Query to JOIN and select required fields ---
    q = """
        SELECT 
            S.roll,          
            S.reg_no,        
            S.class,         
            S.section,       
            A.name,          
            A.timestamp      
        FROM 
            attendance A
        JOIN 
            students S ON A.student_id = S.id 
        """
    params = ()

    # ... (Keep the rest of the 'elif' conditions for daily/weekly/monthly) ...
    if period == "daily":
        today = datetime.date.today().isoformat()
        q += " WHERE date(A.timestamp) = ?"
        params = (today,)
    elif period == "weekly":
        start = (datetime.date.today() - timedelta(days=7)).isoformat()
        q += " WHERE date(A.timestamp) >= ?"
        params = (start,)
    elif period == "monthly":
        start = (datetime.date.today() - timedelta(days=30)).isoformat()
        q += " WHERE date(A.timestamp) >= ?"
        params = (start,)

    q += " ORDER BY A.timestamp DESC LIMIT 5000"

    # Execute query and fetch raw data into a DataFrame
    df = pd.read_sql_query(q, conn, params=params)
    conn.close()

    # --- STEP 2: Pandas Processing and Formatting ---
    if not df.empty:
        # 2a. Convert timestamp to timezone-aware object (UTC)
        df['dt_object'] = pd.to_datetime(df['timestamp'], format='ISO8601', errors='coerce', utc=True)

        # 2b. Convert to Local Timezone (IST)
        df['dt_object'] = df['dt_object'].dt.tz_convert(LOCAL_TIMEZONE)

        # 2c. Format the datetime object (DD/Mon/YYYY | HH:MM:SS)
        df['formatted_time'] = df['dt_object'].dt.strftime('%d/%b/%Y | %H:%M:%S')

        # 2d. Combine class and section into a single column
        df['class_sec'] = df['class'] + ' ' + df['section']

        # 2e. Re-select and reorder the columns to match the new HTML structure (r[0] to r[4])
        df = df[['roll', 'reg_no', 'class_sec', 'name', 'formatted_time']]

        # 2f. Convert the DataFrame back to the list of lists/tuples format for HTML
        rows = df.values.tolist()
    else:
        rows = []

    # -------------------------------------------------------------

    # Return the data
    return render_template("attendance_record.html", records=rows, period=period)


# -------- CSV download (Uses same logic as records) --------
@app.route("/download_csv", methods=["GET"])
def download_csv():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)

    q = """
        SELECT 
            S.roll, S.reg_no, S.class, S.section, A.name, A.timestamp
        FROM 
            attendance A
        JOIN 
            students S ON A.student_id = S.id
        ORDER BY A.timestamp DESC
        """
    df = pd.read_sql_query(q, conn)
    conn.close()

    if not df.empty:
        # 1. Timezone conversion and formatting
        df['dt_object'] = pd.to_datetime(df['timestamp'], format='ISO8601', errors='coerce', utc=True)
        df['dt_object'] = df['dt_object'].dt.tz_convert(LOCAL_TIMEZONE)
        df['Formatted Time'] = df['dt_object'].dt.strftime('%d/%b/%Y | %H:%M:%S')

        # 2. Combine Class and Section
        df['Class/Section'] = df['class'] + ' ' + df['section']

        # 3. Select final columns and rename headers for CSV output
        output_df = df.rename(columns={
            'roll': 'Roll No',
            'reg_no': 'Reg No',
            'name': 'Name',
            'Formatted Time': 'Local Timestamp (IST)',
        })

        # Select and order the desired columns
        output_df = output_df[['Roll No', 'Reg No', 'Class/Section', 'Name', 'Local Timestamp (IST)']]

        # 4. Generate CSV in memory
        mem = io.StringIO()
        output_df.to_csv(mem, index=False)

        mem_bytes = io.BytesIO()
        mem_bytes.write(mem.getvalue().encode("utf-8"))
        mem_bytes.seek(0)

        return send_file(mem_bytes, as_attachment=True, download_name="attendance.csv", mimetype="text/csv")

    return jsonify({"error": "No attendance records found"}), 404


# -------- CSV download - END --------

# -------- Clear Attendance History --------
@app.route("/clear_attendance_history", methods=["GET"])
def clear_attendance_history():
    """Clears all records from the attendance table only."""
    conn = None
    try:
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        c = conn.cursor()
        # SQL command to delete all rows from the attendance table
        c.execute("DELETE FROM attendance")
        conn.commit()
        return jsonify({"status": "success", "message": "Attendance history cleared successfully."}), 200
    except Exception as e:
        app.logger.error(f"Error clearing history: {e}")
        return jsonify({"status": "error", "message": f"Failed to clear history: {e}"}), 500
    finally:
        if conn:
            conn.close()


# -------- Students API for listing/editing --------
@app.route("/students", methods=["GET"])
def students_list():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    c = conn.cursor()
    c.execute("SELECT id, name, roll, class, section, reg_no, created_at FROM students ORDER BY id DESC")
    rows = c.fetchall()
    conn.close()
    data = [{"id": r[0], "name": r[1], "roll": r[2], "class": r[3], "section": r[4], "reg_no": r[5], "created_at": r[6]}
            for r in rows]
    return jsonify({"students": data})


@app.route("/students/<int:sid>", methods=["DELETE"])
def delete_student(sid):
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    c = conn.cursor()
    c.execute("DELETE FROM students WHERE id=?", (sid,))
    c.execute("DELETE FROM attendance WHERE student_id=?", (sid,))
    conn.commit()
    conn.close()
    # also delete dataset folder
    folder = os.path.join(DATASET_DIR, str(sid))
    if os.path.isdir(folder):
        import shutil
        shutil.rmtree(folder, ignore_errors=True)
    return jsonify({"deleted": True})


# ---------------- run ------------------------
if __name__ == "__main__":
    app.run(debug=True)
