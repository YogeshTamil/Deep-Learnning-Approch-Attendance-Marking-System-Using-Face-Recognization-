let recognizedIds = new Set();
// NEW: Counter to manage which frames are sent to the server
let frameCounter = 0;
// We read the camera every 200ms, and skip 5 frames, sending 1 frame out of 6.
// This means a server call happens every 1.2 seconds (6 * 200ms).
const SKIP_COUNT = 15;

const startMarkBtn = document.getElementById('startMarkBtn');
const stopMarkBtn = document.getElementById('stopMarkBtn');
const markVideo = document.getElementById('markVideo');
const markStatus = document.getElementById('markStatus');
const recognizedList = document.getElementById('recognizedList');

let markInterval;
let markStream;

startMarkBtn.addEventListener("click", async () => {
  startMarkBtn.disabled = true;
  stopMarkBtn.disabled = false;
  try {
    markStream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
    markVideo.srcObject = markStream;
    await markVideo.play();
    markStatus.innerText = "Scanning...";

    // MODIFIED: Faster interval (200ms) for smoother video display
    markInterval = setInterval(captureAndRecognize, 200);

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
});

async function captureAndRecognize() {

    // -----------------------------------------------------------------
    // FRAME SKIPPING LOGIC
    frameCounter++;
    if (frameCounter % SKIP_COUNT !== 0) {
        // Skip this frame, don't send it to the server
        return;
    }
    // -----------------------------------------------------------------

    const canvas = document.createElement("canvas");
    canvas.width = markVideo.videoWidth || 640;
    canvas.height = markVideo.videoHeight || 480;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(markVideo, 0, 0, canvas.width, canvas.height);

    // Note: The toBlob method is asynchronous, so the code awaits the result
    const blob = await new Promise(r => canvas.toBlob(r, "image/jpeg", 0.85));
    const fd = new FormData();
    fd.append("image", blob, "snap.jpg");

    try {
        const res = await fetch("/recognize_face", { method: "POST", body: fd });
        const j = await res.json();

        // -----------------------------------------------------------------
        // CRITICAL FIX: RESET COUNTER AFTER RECEIVING SERVER RESPONSE
        // This ensures the client waits for the slow 1.2s server response
        // before starting the next cycle, synchronizing the client and server.
        frameCounter = 0;
        // -----------------------------------------------------------------

        if (j.recognized) {
            // Note: If you reduced COOLDOWN_SECONDS, this will be much faster
            markStatus.innerText = `Recognized: ${j.name} (conf ${Math.round(j.confidence*100)}%)`;
            if (!recognizedIds.has(j.student_id)) {
                recognizedIds.add(j.student_id);
                const li = document.createElement("li");
                li.className = "list-group-item";
                li.innerText = `${j.name} — ${new Date().toLocaleTimeString()}`;
                recognizedList.prepend(li);
            }
        } else {
            if (j.error) markStatus.innerText = `Not recognized: ${j.error}`;
            else markStatus.innerText = `Not recognized`;
        }
    } catch (err) {
        console.error(err);
    }
}
