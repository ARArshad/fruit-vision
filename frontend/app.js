const form = document.getElementById("upload-form");
const fileInput = document.getElementById("file-input");
const preview = document.getElementById("preview");
const previewImg = document.getElementById("preview-img");
const resultBox = document.getElementById("result");

const API_URL = "http://localhost:8000";

function showPreview(file) {
  const reader = new FileReader();
  reader.onload = (event) => {
    previewImg.src = event.target.result;
    preview.classList.remove("hidden");
  };
  reader.readAsDataURL(file);
}

function renderResult(data) {
  const { prediction, confidence } = data;
  const percentage = (confidence * 100).toFixed(2);

  resultBox.innerHTML = `
    <div>Prediction: <strong>${prediction}</strong></div>
    <div>Confidence: <strong>${percentage}%</strong></div>
  `;
  resultBox.classList.remove("hidden", "error");
}

function renderError(message) {
  resultBox.textContent = message;
  resultBox.classList.remove("hidden");
  resultBox.classList.add("error");
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const file = fileInput.files?.[0];
  if (!file) {
    renderError("Please select an image first.");
    return;
  }

  showPreview(file);

  const formData = new FormData();
  formData.append("file", file);

  resultBox.textContent = "Processing...";
  resultBox.classList.remove("hidden", "error");

  try {
    const response = await fetch(`${API_URL}/predict`, {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || "Prediction failed");
    }

    const data = await response.json();
    renderResult(data);
  } catch (error) {
    renderError(error.message);
  }
});
