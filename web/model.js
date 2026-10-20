/*
 * FruitVision in the browser.
 *
 * A line-for-line JavaScript port of the Python inference path, so the static
 * site gives the same answers as `python run.py` without any server:
 *
 *   src/fruitvision/utils/image.py   resize, letterbox, to_model_input
 *   src/fruitvision/features.py      describe (colour / shape / texture vector)
 *   src/fruitvision/models/cnn.py    Conv2D, ReLU, MaxPool2D, Dense, softmax
 *   src/fruitvision/models/descriptor.py  DescriptorNet.predict_proba
 *   src/fruitvision/inference.py     _blend, predict_image
 *
 * It reads the same artifacts/model.json the Python code trains and loads.
 * Keep the two in step: tests/test_web_parity.py runs both on the same pixels
 * and fails if their probabilities drift apart.
 */
(function (root) {
  "use strict";

  var HUE_BINS = 16;
  var GRID = 4;
  var DESCRIPTOR_LENGTH = HUE_BINS + 5 + 5 + 6 + 6 + GRID * GRID * 5 + 2 + 4;
  var TWO_PI = 2 * Math.PI;

  // Python's round() rounds halves to the nearest even number.
  function pyRound(x) {
    var f = Math.floor(x);
    var diff = x - f;
    if (diff > 0.5) return f + 1;
    if (diff < 0.5) return f;
    return f % 2 === 0 ? f : f + 1;
  }

  // Python's % on floats always returns a value with the sign of the divisor.
  function pyMod(a, b) {
    var r = a % b;
    return r < 0 ? r + b : r;
  }

  /* ---------------------------------------------------------------- image */

  // An image is { width, height, pixels } with pixels as packed RGB bytes.

  function resize(image, outW, outH) {
    var w = image.width, h = image.height, src = image.pixels;
    if (outW === w && outH === h) {
      return { width: w, height: h, pixels: new Uint8Array(src) };
    }
    var out = new Uint8Array(outW * outH * 3);
    var spans = [];
    for (var col = 0; col < outW; col++) {
      var sx0 = Math.floor((col * w) / outW);
      var sx1 = Math.max(sx0 + 1, Math.floor(((col + 1) * w) / outW));
      spans.push([sx0, Math.min(sx1, w)]);
    }
    var pos = 0;
    for (var row = 0; row < outH; row++) {
      var y0 = Math.floor((row * h) / outH);
      var y1 = Math.min(Math.max(y0 + 1, Math.floor(((row + 1) * h) / outH)), h);
      for (var s = 0; s < outW; s++) {
        var x0 = spans[s][0], x1 = spans[s][1];
        var r = 0, g = 0, b = 0, count = 0;
        for (var y = y0; y < y1; y++) {
          var base = (y * w + x0) * 3;
          for (var x = x0; x < x1; x++) {
            r += src[base]; g += src[base + 1]; b += src[base + 2];
            base += 3; count++;
          }
        }
        out[pos] = Math.floor(r / count);
        out[pos + 1] = Math.floor(g / count);
        out[pos + 2] = Math.floor(b / count);
        pos += 3;
      }
    }
    return { width: outW, height: outH, pixels: out };
  }

  function letterbox(image, size) {
    var scale = size / Math.max(image.width, image.height);
    var newW = Math.max(1, Math.min(size, pyRound(image.width * scale)));
    var newH = Math.max(1, Math.min(size, pyRound(image.height * scale)));
    var scaled = resize(image, newW, newH);

    var canvas = new Uint8Array(size * size * 3).fill(255);
    var offX = Math.floor((size - newW) / 2);
    var offY = Math.floor((size - newH) / 2);
    var rowBytes = newW * 3;
    for (var row = 0; row < newH; row++) {
      var dst = ((offY + row) * size + offX) * 3;
      canvas.set(scaled.pixels.subarray(row * rowBytes, row * rowBytes + rowBytes), dst);
    }
    return { width: size, height: size, pixels: canvas };
  }

  function toModelInput(image, size) {
    var px = letterbox(image, size).pixels;
    var area = size * size;
    var planes = new Float64Array(3 * area);
    for (var i = 0; i < area; i++) {
      var j = i * 3;
      planes[i] = px[j] / 255;
      planes[area + i] = px[j + 1] / 255;
      planes[2 * area + i] = px[j + 2] / 255;
    }
    return planes;
  }

  /* ----------------------------------------------------------- descriptor */

  function rgbToHsv(r, g, b) {
    var mx = r > g ? r : g;
    if (b > mx) mx = b;
    var mn = r < g ? r : g;
    if (b < mn) mn = b;
    var delta = mx - mn;
    var saturation = mx <= 0 ? 0 : delta / mx;
    var hue;
    if (delta <= 1e-9) hue = 0;
    else if (mx === r) hue = pyMod((g - b) / delta, 6);
    else if (mx === g) hue = (b - r) / delta + 2;
    else hue = (r - g) / delta + 4;
    return [hue / 6, saturation, mx];
  }

  function describe(pixels, size) {
    var count = size * size;
    var hue = new Float64Array(count), sat = new Float64Array(count), val = new Float64Array(count);
    var i, j, x, y;
    for (i = 0, j = 0; i < count; i++, j += 3) {
      var hsv = rgbToHsv(pixels[j] / 255, pixels[j + 1] / 255, pixels[j + 2] / 255);
      hue[i] = hsv[0]; sat[i] = hsv[1]; val[i] = hsv[2];
    }

    // Foreground mask: drop pixels that look like the border (the backdrop).
    var band = Math.max(1, Math.floor(size / 12));
    var border = [];
    for (y = 0; y < size; y++) {
      for (x = 0; x < size; x++) {
        if (y < band || y >= size - band || x < band || x >= size - band) border.push(y * size + x);
      }
    }
    var sats = border.map(function (k) { return sat[k]; }).sort(function (a, b) { return a - b; });
    var vals = border.map(function (k) { return val[k]; }).sort(function (a, b) { return a - b; });
    var backSat = sats[Math.floor(sats.length / 2)];
    var backVal = vals[Math.floor(vals.length / 2)];
    var cx = 0, cy = 0;
    for (var bi = 0; bi < border.length; bi++) cx += Math.cos(TWO_PI * hue[border[bi]]) * sat[border[bi]];
    for (bi = 0; bi < border.length; bi++) cy += Math.sin(TWO_PI * hue[border[bi]]) * sat[border[bi]];
    var backHue = pyMod(Math.atan2(cy, cx) / TWO_PI, 1);

    var mask = new Uint8Array(count).fill(1);
    var foreground = count;
    for (i = 0; i < count; i++) {
      if (Math.abs(val[i] - backVal) >= 0.16 || Math.abs(sat[i] - backSat) >= 0.16) continue;
      if (backSat >= 0.12) {
        var gap = Math.abs(hue[i] - backHue);
        if (Math.min(gap, 1 - gap) >= 0.08) continue;
      }
      mask[i] = 0;
      foreground -= 1;
    }
    if (foreground < 0.06 * count || foreground > 0.97 * count) {
      mask.fill(0);
      foreground = 0;
      var radius = size * 0.42, centre = (size - 1) / 2;
      for (y = 0; y < size; y++) {
        for (x = 0; x < size; x++) {
          if (Math.pow(x - centre, 2) + Math.pow(y - centre, 2) <= radius * radius) {
            mask[y * size + x] = 1;
            foreground += 1;
          }
        }
      }
    }

    var inv = 1 / Math.max(foreground, 1);
    var hueHist = new Array(HUE_BINS).fill(0), satHist = [0, 0, 0, 0, 0], valHist = [0, 0, 0, 0, 0];
    var sumR = 0, sumG = 0, sumB = 0, sumR2 = 0, sumG2 = 0, sumB2 = 0, hueX = 0, hueY = 0;
    var x0 = size, y0 = size, x1 = -1, y1 = -1;

    for (i = 0; i < count; i++) {
      if (!mask[i]) continue;
      var s = sat[i], v = val[i];
      hueHist[Math.floor(hue[i] * HUE_BINS) % HUE_BINS] += s * v;
      satHist[Math.min(4, Math.floor(s * 5))] += 1;
      valHist[Math.min(4, Math.floor(v * 5))] += 1;
      var angle = TWO_PI * hue[i];
      hueX += Math.cos(angle) * s;
      hueY += Math.sin(angle) * s;
      j = i * 3;
      var r = pixels[j] / 255, g = pixels[j + 1] / 255, b = pixels[j + 2] / 255;
      sumR += r; sumG += g; sumB += b;
      sumR2 += r * r; sumG2 += g * g; sumB2 += b * b;
      y = Math.floor(i / size); x = i % size;
      if (x < x0) x0 = x;
      if (x > x1) x1 = x;
      if (y < y0) y0 = y;
      if (y > y1) y1 = y;
    }

    var totalHue = 0;
    for (var hk = 0; hk < HUE_BINS; hk++) totalHue += hueHist[hk];
    if (!totalHue) totalHue = 1;
    hueHist = hueHist.map(function (h) { return h / totalHue; });
    satHist = satHist.map(function (c) { return c * inv; });
    valHist = valHist.map(function (c) { return c * inv; });

    var meanR = sumR * inv, meanG = sumG * inv, meanB = sumB * inv;
    var stdR = Math.sqrt(Math.max(0, sumR2 * inv - meanR * meanR));
    var stdG = Math.sqrt(Math.max(0, sumG2 * inv - meanG * meanG));
    var stdB = Math.sqrt(Math.max(0, sumB2 * inv - meanB * meanB));

    if (x1 < x0) { x0 = 0; y0 = 0; x1 = size - 1; y1 = size - 1; }
    var boxW = x1 - x0 + 1, boxH = y1 - y0 + 1;
    var aspect = boxW / boxH;
    var shape = [
      foreground / count,
      Math.min(aspect, 1 / aspect),
      aspect >= 1 ? 1 : 0,
      foreground / (boxW * boxH),
      boxW / size,
      boxH / size
    ];

    var grid = [];
    for (var gy = 0; gy < GRID; gy++) {
      var ay0 = y0 + Math.floor((boxH * gy) / GRID);
      var ay1 = Math.max(ay0 + 1, y0 + Math.floor((boxH * (gy + 1)) / GRID));
      for (var gx = 0; gx < GRID; gx++) {
        var ax0 = x0 + Math.floor((boxW * gx) / GRID);
        var ax1 = Math.max(ax0 + 1, x0 + Math.floor((boxW * (gx + 1)) / GRID));
        var cells = 0, cellX = 0, cellY = 0, cellS = 0, cellV = 0;
        for (y = ay0; y < ay1; y++) {
          var base = y * size;
          for (x = ax0; x < ax1; x++) {
            var k = base + x;
            if (!mask[k]) continue;
            cells += 1;
            var a = TWO_PI * hue[k];
            cellX += Math.cos(a) * sat[k];
            cellY += Math.sin(a) * sat[k];
            cellS += sat[k];
            cellV += val[k];
          }
        }
        if (cells) {
          var kk = 1 / cells, area = (ay1 - ay0) * (ax1 - ax0);
          grid.push(cellX * kk, cellY * kk, cellS * kk, cellV * kk, cells / area);
        } else {
          grid.push(0, 0, 0, 0, 0);
        }
      }
    }

    var edgeSum = 0, edgeCount = 0, strongEdges = 0;
    for (y = 1; y < size - 1; y++) {
      for (x = 1; x < size - 1; x++) {
        i = y * size + x;
        if (!mask[i]) continue;
        var magnitude = Math.abs(val[i + 1] - val[i - 1]) + Math.abs(val[i + size] - val[i - size]);
        edgeSum += magnitude;
        edgeCount += 1;
        if (magnitude > 0.25) strongEdges += 1;
      }
    }
    var texture = [edgeSum / Math.max(edgeCount, 1), strongEdges / Math.max(edgeCount, 1)];

    var coherence = Math.sqrt(hueX * hueX + hueY * hueY) * inv;
    var sumSat = 0, sumVal = 0;
    for (i = 0; i < count; i++) if (mask[i]) { sumSat += sat[i]; sumVal += val[i]; }

    return hueHist.concat(satHist, valHist,
      [meanR, meanG, meanB, stdR, stdG, stdB],
      shape, grid, texture,
      [coherence, sumSat * inv, sumVal * inv, stdR + stdG + stdB]);
  }

  /* --------------------------------------------------------------- layers */

  function conv2d(x, state, inC, outC, k, inH, inW) {
    var outH = inH - k + 1, outW = inW - k + 1;
    var planeIn = inH * inW, taps = inC * k * k, w = state.weights;
    var out = new Float64Array(outC * outH * outW);
    for (var co = 0; co < outC; co++) {
      var obase = co * outH * outW, bias = state.biases[co];
      for (var p = 0; p < outH * outW; p++) out[obase + p] = bias;
      var wbase = co * taps;
      for (var ci = 0; ci < inC; ci++) {
        var offset = ci * planeIn;
        for (var ky = 0; ky < k; ky++) {
          var rowBase = offset + ky * inW;
          for (var kx = 0; kx < k; kx++) {
            var weight = w[wbase + (ci * k + ky) * k + kx];
            if (weight === 0) continue;
            for (var oy = 0; oy < outH; oy++) {
              var src = rowBase + oy * inW + kx, dst = obase + oy * outW;
              for (var ox = 0; ox < outW; ox++) out[dst + ox] += weight * x[src + ox];
            }
          }
        }
      }
    }
    return { data: out, h: outH, w: outW };
  }

  function relu(x) {
    var out = new Float64Array(x.length);
    for (var i = 0; i < x.length; i++) out[i] = x[i] > 0 ? x[i] : 0;
    return out;
  }

  function maxPool(x, c, inH, inW) {
    var outH = Math.floor(inH / 2), outW = Math.floor(inW / 2);
    var out = new Float64Array(c * outH * outW), pos = 0;
    for (var ch = 0; ch < c; ch++) {
      var plane = ch * inH * inW;
      for (var oy = 0; oy < outH; oy++) {
        var base = plane + oy * 2 * inW;
        for (var ox = 0; ox < outW; ox++) {
          var i = base + ox * 2, value = x[i];
          if (x[i + 1] > value) value = x[i + 1];
          if (x[i + inW] > value) value = x[i + inW];
          if (x[i + inW + 1] > value) value = x[i + inW + 1];
          out[pos++] = value;
        }
      }
    }
    return { data: out, h: outH, w: outW };
  }

  function dense(x, state) {
    var n = x.length, outF = state.biases.length, w = state.weights;
    var out = new Float64Array(outF);
    for (var o = 0; o < outF; o++) {
      var base = o * n, total = 0;
      for (var i = 0; i < n; i++) total += w[base + i] * x[i];
      out[o] = total + state.biases[o];
    }
    return out;
  }

  function softmax(logits) {
    var top = -Infinity, i;
    for (i = 0; i < logits.length; i++) if (logits[i] > top) top = logits[i];
    var exps = [], total = 0;
    for (i = 0; i < logits.length; i++) { exps.push(Math.exp(logits[i] - top)); total += exps[i]; }
    return exps.map(function (e) { return e / total; });
  }

  /* ---------------------------------------------------------------- model */

  function cnnProbs(model, input) {
    var a = model.architecture, L = model.layers, size = a.img_size, k = a.kernel_size;
    // layers: conv, relu, pool, conv, relu, pool, dense, relu, dropout, dense
    var c1 = conv2d(input, L[0], a.channels, a.conv1_filters, k, size, size);
    var p1 = maxPool(relu(c1.data), a.conv1_filters, c1.h, c1.w);
    var c2 = conv2d(p1.data, L[3], a.conv1_filters, a.conv2_filters, k, p1.h, p1.w);
    var p2 = maxPool(relu(c2.data), a.conv2_filters, c2.h, c2.w);
    var hidden = relu(dense(p2.data, L[6]));
    return softmax(dense(hidden, L[9]));
  }

  function descriptorProbs(model, image) {
    var d = model.descriptor;
    if (d.features !== DESCRIPTOR_LENGTH) {
      throw new Error("Model descriptor has " + d.features + " features; this page expects " + DESCRIPTOR_LENGTH + ".");
    }
    var vec = describe(letterbox(image, d.size).pixels, d.size);
    var normalised = vec.map(function (v, i) {
      var s = Math.abs(d.std[i]) > 1e-6 ? d.std[i] : 1;
      return (v - d.mean[i]) / s;
    });
    // layers: dense, relu, dropout, dense
    var hidden = relu(dense(normalised, d.layers[0]));
    return softmax(dense(hidden, d.layers[3]));
  }

  /** Classify an RGB image. Returns { scores: {name: p}, prediction, confidence }. */
  function predictImage(model, image) {
    var cnn = cnnProbs(model, toModelInput(image, model.architecture.img_size));
    var probs = cnn;
    if (model.descriptor) {
      var desc = descriptorProbs(model, image);
      var wgt = model.cnn_weight;
      probs = desc.map(function (dv, i) { return (1 - wgt) * dv + wgt * cnn[i]; });
    }
    var scores = {}, best = 0;
    model.class_names.forEach(function (name, i) {
      scores[name] = probs[i];
      if (probs[i] > probs[best]) best = i;
    });
    return { scores: scores, prediction: model.class_names[best], confidence: probs[best] };
  }

  /** Convert canvas ImageData (RGBA) into the packed RGB image the model reads. */
  function fromImageData(imageData) {
    var src = imageData.data, n = imageData.width * imageData.height;
    var rgb = new Uint8Array(n * 3);
    for (var i = 0, j = 0; i < n; i++, j += 4) {
      rgb[i * 3] = src[j]; rgb[i * 3 + 1] = src[j + 1]; rgb[i * 3 + 2] = src[j + 2];
    }
    return { width: imageData.width, height: imageData.height, pixels: rgb };
  }

  var api = {
    DESCRIPTOR_LENGTH: DESCRIPTOR_LENGTH,
    resize: resize,
    letterbox: letterbox,
    toModelInput: toModelInput,
    describe: describe,
    predictImage: predictImage,
    fromImageData: fromImageData
  };

  if (typeof module === "object" && module.exports) module.exports = api;
  else root.FruitVisionModel = api;
})(typeof self !== "undefined" ? self : this);
