const startMarkBtn = document.getElementById("startMarkBtn");
const stopMarkBtn = document.getElementById("stopMarkBtn");
const markVideo = document.getElementById("markVideo");
const markStatus = document.getElementById("markStatus");
const recognizedList = document.getElementById("recognizedList");

let markStream = null;
let markInterval = null;
let recognizedIds = new Set();
let isCooldowned = false; // NEW: Flag to pause frame sending after success

// The client doesn't need to know the server's long cooldown, but
// we set a short, aggressive debounce timer to prevent request overlap.
let isRecognizing = false;

startMarkBtn.addEventListener("click", async () => {
  if (isRecognizing) return; // Prevent double-clicking issues

  // Reset state on start
  recognizedIds.clear();
  isCooldowned = false;

  startMarkBtn.disabled = true;
  stopMarkBtn.disabled = false;

  try {
    markStream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
    markVideo.srcObject = markStream;
    await markVideo.play();
    markStatus.innerText = "Scanning...";

    // Start interval to capture and recognize every 1.2 seconds
    markInterval = setInterval(captureAndRecognize, 1200);
  } catch (err) {
    alert("Camera error: " + err.message);
    startMarkBtn.disabled = false;
    stopMarkBtn.disabled = true;
  }
});

stopMarkBtn.addEventListener("click", () => {
  if (markInterval) clearInterval(markInterval);
  if (markStream) markStream.getTracks().forEach(t => t.stop());
  startMarkBtn.disabled = false;
  stopMarkBtn.disabled = true;
  markStatus.innerText = "Stopped";
  isRecognizing = false;
  isCooldowned = false; // Reset flags
});

async function captureAndRecognize() {
  if (isCooldowned || isRecognizing) {
    return; // Do nothing if we are waiting for server response or cooling down
  }

  isRecognizing = true; // Set flag: Request sent

  const canvas = document.createElement("canvas");
  canvas.width = markVideo.videoWidth || 640;
  canvas.height = markVideo.videoHeight || 480;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(markVideo, 0, 0, canvas.width, canvas.height);

  // Convert to Blob and prepare FormData
  const blob = await new Promise(r => canvas.toBlob(r, "image/jpeg", 0.85));
  const fd = new FormData();
  fd.append("image", blob, "snap.jpg");

  try {
    const res = await fetch("/recognize_face", { method: "POST", body: fd });
    const j = await res.json();

    // Check for explicit server-side cooldown status or a fresh successful recognition
    if (j.recognized && (j.status === 'cooldown' || !recognizedIds.has(j.student_id))) {

      // If the status is NOT 'cooldown', it means a fresh entry was just logged on the server.
      if (j.status !== 'cooldown') {

        // --- CRITICAL CHANGE: Stop sending frames after successful mark ---
        if (markInterval) {
            clearInterval(markInterval); // Stop the periodic capture
        }
        isCooldowned = true; // Set client-side flag to true

        markStatus.innerText = `Attendance LOGGED for: ${j.name}`;

        // Display record in the list
        recognizedIds.add(j.student_id);
        const li = document.createElement("li");
        li.className = "list-group-item list-group-item-success";
        li.innerText = `${j.name} — LOGGED at ${new Date().toLocaleTimeString()}`;
        recognizedList.prepend(li);

        // The camera remains open, but scanning is paused until 'Stop' is pressed.
      } else {
        // Status is 'cooldown', meaning a recent successful entry was processed.
        markStatus.innerText = `Recognized, on cooldown: ${j.name}`;
      }
    } else {
      // Not recognized, or confidence too low
      if (j.error) markStatus.innerText = `Not recognized: ${j.error}`;
      else markStatus.innerText = `Scanning... (Searching)`;
    }
  } catch (err) {
    console.error(err);
    markStatus.innerText = "Error communicating with server.";
  } finally {
    isRecognizing = false; // Always reset the busy flag after the response
  }
}