(function () {
  "use strict";

  // Same shrink the original page did before uploading; the model then
  // letterboxes this down to 32x32 and 48x48 exactly as the Python code does.
  var UPLOAD_MAX_EDGE = 256;

  var fileInput = document.getElementById("file-input");
  var chooseBtn = document.getElementById("choose-btn");
  var predictBtn = document.getElementById("predict-btn");
  var previewImg = document.getElementById("preview-img");
  var resultEl = document.getElementById("result");
  var modelInfoEl = document.getElementById("model-info");
  var samplesEl = document.getElementById("samples");

  var model = null;
  var modelError = null;
  var source = null; // a File or a sample URL
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

  function setActiveSample(src) {
    Array.prototype.forEach.call(samplesEl.querySelectorAll(".sample"), function (btn) {
      btn.classList.toggle("is-active", btn.getAttribute("data-src") === src);
    });
  }

  function choose(next, displayUrl) {
    source = next;
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    previewUrl = typeof next === "string" ? null : displayUrl;
    previewImg.src = displayUrl;
    previewImg.classList.remove("hidden");
    predictBtn.disabled = false;
    resultEl.classList.add("hidden");
  }

  function selectFile(file) {
    if (!file) return;
    if (!/^image\//.test(file.type) && !/\.(png|jpe?g|bmp|webp|gif)$/i.test(file.name)) {
      showStatus("That does not look like an image. Please choose a JPG, PNG or WebP file.", true);
      return;
    }
    setActiveSample(null);
    choose(file, URL.createObjectURL(file));
  }

  function loadImage(url) {
    return new Promise(function (resolve, reject) {
      var image = new Image();
      image.onload = function () { resolve(image); };
      image.onerror = function () { reject(new Error("The browser could not read that image.")); };
      image.src = url;
    });
  }

  // Draw onto a white canvas no bigger than 256px and read the RGB pixels.
  function toPixels(image) {
    var width = image.naturalWidth || image.width;
    var height = image.naturalHeight || image.height;
    if (!width || !height) throw new Error("That image appears to be empty.");
    var scale = Math.min(1, UPLOAD_MAX_EDGE / Math.max(width, height));
    var canvas = document.createElement("canvas");
    canvas.width = Math.max(1, Math.round(width * scale));
    canvas.height = Math.max(1, Math.round(height * scale));
    var ctx = canvas.getContext("2d", { willReadFrequently: true });
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
    return FruitVisionModel.fromImageData(ctx.getImageData(0, 0, canvas.width, canvas.height));
  }

  function renderResult(result) {
    var ranked = Object.keys(result.scores)
      .map(function (name) { return [name, result.scores[name]]; })
      .sort(function (a, b) { return b[1] - a[1]; })
      .slice(0, 3);

    var rows = ranked.map(function (entry, i) {
      var pct = (entry[1] * 100).toFixed(1);
      return '<li class="bar' + (i === 0 ? " bar--top" : "") + '">' +
        '<span class="bar__name">' + escapeHtml(entry[0]) + "</span>" +
        '<span class="bar__track"><span class="bar__fill" style="width:' + pct + '%"></span></span>' +
        '<span class="bar__pct">' + pct + "%</span></li>";
    }).join("");

    resultEl.className = "result-box";
    resultEl.innerHTML =
      '<p class="prediction">Prediction: ' + escapeHtml(result.prediction) + "</p>" +
      '<p class="confidence">Confidence: ' + (result.confidence * 100).toFixed(2) + "%</p>" +
      '<ol class="bars" aria-label="Top three guesses">' + rows + "</ol>";
  }

  function predict() {
    if (!source) return;
    if (modelError) { showStatus(modelError, true); return; }
    if (!model) { showStatus("The model is still loading, try again in a moment.", true); return; }

    predictBtn.disabled = true;
    showStatus("Analysing image…", false);

    var url = typeof source === "string" ? source : previewImg.src;
    loadImage(url)
      .then(function (image) {
        // Give the spinner a frame to paint before the (fast) maths runs.
        return new Promise(function (resolve) { requestAnimationFrame(function () { resolve(image); }); });
      })
      .then(function (image) {
        renderResult(FruitVisionModel.predictImage(model, toPixels(image)));
      })
      .catch(function (error) {
        showStatus(error.message || "Something went wrong.", true);
      })
      .then(function () {
        predictBtn.disabled = false;
      });
  }

  chooseBtn.addEventListener("click", function () { fileInput.click(); });
  fileInput.addEventListener("change", function () { selectFile(fileInput.files[0]); fileInput.value = ""; });
  predictBtn.addEventListener("click", predict);
  samplesEl.addEventListener("click", function (e) {
    var btn = e.target.closest(".sample");
    if (!btn) return;
    var src = btn.getAttribute("data-src");
    setActiveSample(src);
    choose(src, src);
    predict();
  });

  // Drag and drop a photo anywhere on the card.
  var card = document.querySelector(".card");
  card.addEventListener("dragover", function (e) { e.preventDefault(); card.classList.add("is-dragging"); });
  card.addEventListener("dragleave", function () { card.classList.remove("is-dragging"); });
  card.addEventListener("drop", function (e) {
    e.preventDefault();
    card.classList.remove("is-dragging");
    if (e.dataTransfer.files && e.dataTransfer.files[0]) selectFile(e.dataTransfer.files[0]);
  });

  Promise.all([
    fetch("model.json").then(function (r) { if (!r.ok) throw new Error(); return r.json(); }),
    fetch("info.json").then(function (r) { return r.ok ? r.json() : {}; }).catch(function () { return {}; })
  ])
    .then(function (loaded) {
      model = loaded[0];
      var info = loaded[1] || {};
      var parts = ["Recognises " + model.class_names.length + " fruits"];
      if (info.parameters) parts.push(Number(info.parameters).toLocaleString() + " parameters");
      if (typeof info.test_accuracy === "number") {
        parts.push((info.test_accuracy * 100).toFixed(1) + "% accuracy on " + (info.test_images || "unseen") + " unseen test photos");
      }
      modelInfoEl.textContent = parts.join(" · ");
    })
    .catch(function () {
      modelError = "The model could not be loaded. Please refresh the page.";
      modelInfoEl.textContent = modelError;
    });
})();
