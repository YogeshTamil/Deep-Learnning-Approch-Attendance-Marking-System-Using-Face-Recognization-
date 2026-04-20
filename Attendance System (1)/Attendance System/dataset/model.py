import os
import cv2
import numpy as np
import pickle
import faiss
from PIL import Image
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1

# --- GLOBAL CONFIGURATION (UPDATED for FaceNet) ---
MODEL_PATH = "dataset/face_encodings.faiss"
ID_MAP_PATH = "dataset/student_ids.pkl"
EMBEDDING_DIM = 512  # FaceNet's default embedding size

# Haar Cascade is still used for detection during training/alignment check
HAAR_CASCADE_PATH = os.path.join(os.path.dirname(__file__), "dataset", "haarcascade_frontalface_default.xml")

# Initialize DCNN Models (Loaded once when the server starts)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Loading FaceNet models on device: {device}")

# 1. MTCNN for fast detection and face alignment
# image_size=160 is the standard input size for the ResNet model
mtcnn = MTCNN(image_size=160, margin=0, device=device, selection_method='center_weighted_size')

# 2. InceptionResnetV1 (FaceNet architecture) for embedding generation
facenet_model = InceptionResnetV1(pretrained='vggface2').eval().to(device)


# ---- Utility: DCNN Embedding Generation ----
def get_facenet_embedding(face_img):
    """
    Takes a single, aligned face tensor and generates the 512-dim embedding.
    """
    with torch.no_grad():
        # The unsqueeze(0) adds the batch dimension required by PyTorch (1, 3, 160, 160)
        embedding = facenet_model(face_img.to(device).unsqueeze(0))
        # Returns a 512-dimensional NumPy array
    return embedding.detach().cpu().numpy().flatten()


# ---- Face Detection and Embedding Extraction (UPDATED for DCNN) ----
def extract_embedding_for_image(stream_or_bytes):
    # Read image from stream into PIL/Pillow format for MTCNN
    data = stream_or_bytes.read()

    # MTCNN requires RGB image data
    img_array = np.frombuffer(data, np.uint8)
    img_bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img_bgr is None:
        return None

    # Convert BGR (OpenCV) to RGB (PIL/PyTorch)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)

    # MTCNN detects and aligns the face, outputting a tensor [3, 160, 160]
    # This step is much more robust than the simple Haar/32x32 crop
    aligned_face = mtcnn(img_pil, save_path=None)

    if aligned_face is None:
        return None

    # Generate the highly accurate 512-dimensional embedding
    return get_facenet_embedding(aligned_face)


# ---- Load model helpers (NO CHANGE to Faiss loading logic) ----
def load_model_if_exists():
    """Loads the Faiss index and the student ID map."""
    if not os.path.exists(MODEL_PATH) or not os.path.exists(ID_MAP_PATH):
        return None

    try:
        index = faiss.read_index(MODEL_PATH)
        with open(ID_MAP_PATH, 'rb') as f:
            student_ids_list = pickle.load(f)
        return (index, student_ids_list)
    except Exception as e:
        print(f"Error loading Faiss model: {e}")
        return None


def predict_with_model(clf_tuple, emb):
    """Uses the Faiss index to find the nearest neighbor quickly."""
    index, student_ids = clf_tuple

    # Faiss search expects a 2D numpy array of shape (1, D)
    query_vector = np.array(emb).astype('float32').reshape(1, -1)

    k = 1
    D, I = index.search(query_vector, k)

    nearest_index = I[0][0]
    distance = D[0][0]

    if nearest_index == -1:
        return ("Unknown", 0.0)

    predicted_student_id = student_ids[nearest_index]

    # --- Distance Thresholding (UPDATED for DCNN) ---
    # DCNN embeddings are tightly clustered; use a low threshold.
    DISTANCE_THRESHOLD = 0.9  # Adjusted for FaceNet L2 distance

    if distance < DISTANCE_THRESHOLD:
        # Confidence logic remains the same
        confidence = 1.0 - (distance / DISTANCE_THRESHOLD)
        if confidence < 0: confidence = 0.0
        return (str(predicted_student_id), confidence)
    else:
        return ("Unknown", 0.1)


# ---- Training function (UPDATED for DCNN) ----
def train_model_background(dataset_dir, progress_callback=None):
    """
    Trains the Faiss index using the highly accurate DCNN embeddings.
    """
    X = []  # List to hold all face embeddings (vectors)
    y_student_id = []  # List to hold the student IDs (labels)

    student_dirs = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d)) and d.isdigit()]
    total_students = max(1, len(student_dirs))
    processed = 0

    # 1. Gather all embeddings and labels
    for sid_str in student_dirs:
        folder = os.path.join(dataset_dir, sid_str)
        files = [f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

        for fn in files:
            path = os.path.join(folder, fn)

            try:
                # Read image using PIL/Pillow
                img_pil = Image.open(path).convert('RGB')
            except Exception:
                continue

            # MTCNN detects and aligns the face
            aligned_face = mtcnn(img_pil, save_path=None)

            if aligned_face is None:
                continue

            # Generate the 512-dimensional embedding
            emb = get_facenet_embedding(aligned_face)

            if emb is None:
                continue

            X.append(emb)
            y_student_id.append(int(sid_str))

        processed += 1
        if progress_callback:
            pct = int((processed / total_students) * 80)
            progress_callback(pct, f"Processed {processed}/{total_students} students using FaceNet")

    if len(X) == 0:
        if progress_callback:
            progress_callback(0, "No training data found")
        return

    # Rest of Faiss build/save logic (D=512 is handled automatically)
    X = np.stack(X).astype('float32')
    y_student_id = np.array(y_student_id)
    d = X.shape[1]

    if progress_callback:
        progress_callback(85, f"Building Faiss Index with D={d}...")

    index = faiss.IndexFlatL2(d)
    index.add(X)

    if progress_callback:
        progress_callback(95, "Saving Index and ID map...")

    faiss.write_index(index, MODEL_PATH)

    with open(ID_MAP_PATH, 'wb') as f:
        pickle.dump(y_student_id.tolist(), f)

    if progress_callback:
        progress_callback(100, "Training complete. Faiss index created with DCNN embeddings.")