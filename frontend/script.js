// frontend\script.js
const backendUrl = "http://localhost:8000";
const uploadBtn = document.getElementById("uploadBtn");
const askBtn = document.getElementById("askBtn");
const pdfInput = document.getElementById("pdfInput");
const uploadStatus = document.getElementById("uploadStatus");
const queryInput = document.getElementById("queryInput");
const answerBox = document.getElementById("answerBox");

uploadBtn.addEventListener("click", uploadPDF);
askBtn.addEventListener("click", askQuestion);

async function uploadPDF() {
  if (!pdfInput.files || !pdfInput.files.length) {
    uploadStatus.innerText = "Please select a PDF file.";
    return;
  }
  const file = pdfInput.files[0];
  const fd = new FormData();
  fd.append("file", file);

  uploadStatus.innerText = "Uploading and processing PDF... (this may take a few seconds)";
  uploadBtn.disabled = true;

  try {
    const res = await fetch(`${backendUrl}/upload_pdf/`, {
      method: "POST",
      body: fd
    });
    const data = await res.json();
    if (data.status && data.status === "ok") {
      uploadStatus.innerText = `Processed: ${data.chunks} chunks added. You can now ask questions.`;
    } else {
      uploadStatus.innerText = data.message || "Upload failed or returned no text.";
    }
  } catch (err) {
    console.error(err);
    uploadStatus.innerText = "Error uploading file. See console.";
  } finally {
    uploadBtn.disabled = false;
  }
}

async function askQuestion() {
  const q = queryInput.value.trim();
  if (!q) {
    answerBox.innerText = "Please enter a question.";
    return;
  }
  answerBox.innerText = "Searching and preparing excerpts...";
  askBtn.disabled = true;

  try {
    const fd = new FormData();
    fd.append("query", q);

    const res = await fetch(`${backendUrl}/ask_question/`, {
      method: "POST",
      body: fd
    });
    const data = await res.json();
    if (data.answer) {
      answerBox.innerText = data.answer;
    } else {
      answerBox.innerText = JSON.stringify(data, null, 2);
    }
  } catch (err) {
    console.error(err);
    answerBox.innerText = "Error getting answer. See console.";
  } finally {
    askBtn.disabled = false;
  }
}
