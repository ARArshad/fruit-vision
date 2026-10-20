(function () {
  "use strict";

  var UPLOAD_MAX_EDGE = 256;

  var fileInput = document.getElementById("file-input");
  var chooseBtn = document.getElementById("choose-btn");
  var predictBtn = document.getElementById("predict-btn");
  var previewImg = document.getElementById("preview-img");
  var resultEl = document.getElementById("result");
  var modelInfoEl = document.getElementById("model-info");

  var selectedFile = null;
  var previewUrl = null;

  function escapeHtml(value) {
    return String(value).replace(/[&<>"']/g, function (ch) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[ch];
    });
  }

  function showStatus(message, isError) {
    resultEl.className = "result-box";
    resultEl.innerHTML =
      '<p class="status' + (isError ? " error" : "") + '">' +
      (isError ? "" : '<span class="spinner"></span>') +
      escapeHtml(message) +
      "</p>";
  }

  function selectFile(file) {
    if (!file) return;

    if (!/^image\//.test(file.type) && !/\.(png|jpe?g|bmp)$/i.test(file.name)) {
      showStatus("That does not look like an image. Please choose a JPG, PNG or BMP file.", true);
      return;
    }

    selectedFile = file;
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    previewUrl = URL.createObjectURL(file);

    previewImg.src = previewUrl;
    previewImg.classList.remove("hidden");
    predictBtn.disabled = false;
    resultEl.classList.add("hidden");
  }

  function shrinkToDataUrl(file) {
    return new Promise(function (resolve, reject) {
      var url = URL.createObjectURL(file);
      var image = new Image();

      image.onload = function () {
        try {
          var width = image.naturalWidth || image.width;
          var height = image.naturalHeight || image.height;
          if (!width || !height) throw new Error("empty image");

          var scale = Math.min(1, UPLOAD_MAX_EDGE / Math.max(width, height));
          var canvas = document.createElement("canvas");
          canvas.width = Math.max(1, Math.round(width * scale));
          canvas.height = Math.max(1, Math.round(height * scale));

          var ctx = canvas.getContext("2d");
          ctx.fillStyle = "#ffffff";
          ctx.fillRect(0, 0, canvas.width, canvas.height);
          ctx.drawImage(image, 0, 0, canvas.width, canvas.height);

          resolve({ dataUrl: canvas.toDataURL("image/png"), width: width, height: height });
        } catch (err) {
          reject(err);
        } finally {
          URL.revokeObjectURL(url);
        }
      };

      image.onerror = function () {
        URL.revokeObjectURL(url);
        reject(new Error("The browser could not read that image."));
      };

      image.src = url;
    });
  }

  function renderResult(data) {
    var confidence = (data.confidence * 100).toFixed(2);

    resultEl.className = "result-box";
    resultEl.innerHTML =
      '<p class="prediction">Prediction: ' + escapeHtml(data.prediction) + "</p>" +
      '<p class="confidence">Confidence: ' + confidence + "%</p>";
  }

  function predict() {
    if (!selectedFile) return;
    predictBtn.disabled = true;
    showStatus("Analysing image…", false);

    shrinkToDataUrl(selectedFile)
      .then(function (prepared) {
        return fetch("/predict", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ image: prepared.dataUrl })
        }).then(function (response) {
          return response.json().then(function (body) {
            if (!response.ok) throw new Error(body.detail || "Prediction failed.");
            return body;
          });
        });
      })
      .then(function (body) {
        renderResult(body);
      })
      .catch(function (error) {
        showStatus(error.message || "Something went wrong.", true);
      })
      .then(function () {
        predictBtn.disabled = false;
      });
  }

  chooseBtn.addEventListener("click", function () { fileInput.click(); });
  fileInput.addEventListener("change", function () { selectFile(fileInput.files[0]); });
  predictBtn.addEventListener("click", predict);

  fetch("/api/info")
    .then(function (response) { return response.json(); })
    .then(function (info) {
      if (!info.model_loaded) {
        modelInfoEl.textContent = "No trained model loaded — run: python train.py";
        return;
      }
      var parts = ["Recognises " + info.classes.join(", ")];
      parts.push(info.parameters.toLocaleString() + " parameters");
      if (info.metrics && typeof info.metrics.test_accuracy === "number") {
        parts.push("held-out accuracy " + Math.round(info.metrics.test_accuracy * 100) + "%");
      }
      modelInfoEl.textContent = parts.join(" · ");
    })
    .catch(function () {
      modelInfoEl.textContent = "Could not reach the server.";
    });
})();
