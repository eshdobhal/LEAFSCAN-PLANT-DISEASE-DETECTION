// LeafScan in the browser: runs the two CAFormer-S18 models with ONNX Runtime Web, entirely on this device.
// diagnose.js (shared with the Flask app) handles choosing, dropping, pasting, resizing and the camera;
// this file loads the models, diagnoses the chosen photo and draws the result.
(function () {
  "use strict";

  const VERSION = "v1.0.1";                         // bump with new model files to refresh visitors' caches
  const CACHE = "leafscan-models-" + VERSION;
  const MODELS = [                                  // same ensemble as the server: 224 px + 288 px
    { url: "models/leafscan_224.onnx", size: 224 },
    { url: "models/leafscan_288.onnx", size: 288 },
  ];
  const MEAN = [0.485, 0.456, 0.406];               // ImageNet normalisation, as in training
  const STD = [0.229, 0.224, 0.225];
  const THRESHOLD = 0.8;                            // below this, warn that the model isn't sure

  const $ = (selector) => document.querySelector(selector);
  const status = $("[data-model-status]");
  const statusText = $("[data-model-text]");
  const statusBar = $("[data-model-progress]");
  const statusHint = $("[data-model-hint]");
  const form = $("[data-upload-form]");
  const input = $("#image");
  const errorBox = $("[data-error]");
  const resultBox = $("[data-result]");
  let catalog = null;
  let sessions = null;
  let backend = "";

  // ---------------------------------------------------------------- loading
  // Download a model with a progress callback, keeping a copy in the browser's Cache Storage.
  async function fetchModel(url, onProgress) {
    let cache = null;
    try {
      cache = await caches.open(CACHE);
      const hit = await cache.match(url);
      if (hit) { onProgress(1); return await hit.arrayBuffer(); }
    } catch (e) { cache = null; /* private mode or file:// : just download */ }
    const response = await fetch(url);
    if (!response.ok) throw new Error("Could not download " + url + " (" + response.status + ")");
    const total = Number(response.headers.get("content-length")) || 49e6;
    const reader = response.body.getReader();
    const chunks = [];
    let received = 0;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      received += value.length;
      onProgress(Math.min(1, received / total));
    }
    const blob = new Blob(chunks);
    if (cache) { try { await cache.put(url, new Response(blob)); } catch (e) { /* quota: fine */ } }
    return blob.arrayBuffer();
  }

  // WebGPU (fast, recent Chrome/Edge) if it works, otherwise WebAssembly (every modern browser).
  async function createSession(buffer) {
    if (navigator.gpu) {
      try {
        const session = await ort.InferenceSession.create(buffer, { executionProviders: ["webgpu"] });
        backend = "WebGPU";
        return session;
      } catch (e) { console.warn("WebGPU unavailable, using WebAssembly:", e.message); }
    }
    backend = "WebAssembly";
    return ort.InferenceSession.create(buffer, { executionProviders: ["wasm"] });
  }

  async function load() {
    // Old caches from previous versions are deleted so they don't fill the visitor's disk.
    try {
      for (const name of await caches.keys()) if (name.startsWith("leafscan-models-") && name !== CACHE) await caches.delete(name);
    } catch (e) { /* no Cache Storage */ }
    catalog = await (await fetch("catalog.json")).json();
    const progress = MODELS.map(() => 0);
    const show = () => {
      const done = progress.reduce((a, b) => a + b, 0) / MODELS.length;
      statusBar.value = done;
      statusText.textContent = "Loading the AI model… " + Math.round(done * 100) + "%";
    };
    const buffers = await Promise.all(MODELS.map((m, i) => fetchModel(m.url, (p) => { progress[i] = p; show(); })));
    statusText.textContent = "Starting the AI model…";
    sessions = [];
    for (const buffer of buffers) sessions.push(await createSession(buffer));  // one at a time: less memory
    status.dataset.state = "ready";
    statusText.textContent = "Ready. The AI model runs on this device (" + backend + ").";
    statusHint.textContent = "Your photos are never uploaded.";
  }

  // ---------------------------------------------------------------- inference
  // Same preparation as the server: short side to size × 256/224, centre crop size × size,
  // ImageNet normalisation, laid out as [channel][row][column]. Returns the photo and its mirror image.
  function toTensor(bitmap, size) {
    const resized = Math.round(size * 256 / 224);
    const scale = resized / Math.min(bitmap.width, bitmap.height);
    const crop = size / scale;                       // the centre square, in original-photo pixels
    const canvas = document.createElement("canvas");
    canvas.width = canvas.height = size;
    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    ctx.imageSmoothingQuality = "high";
    ctx.drawImage(bitmap, (bitmap.width - crop) / 2, (bitmap.height - crop) / 2, crop, crop, 0, 0, size, size);
    const pixels = ctx.getImageData(0, 0, size, size).data;  // RGBA bytes
    const plane = size * size;
    const data = new Float32Array(2 * 3 * plane);            // batch of 2: photo, mirrored photo
    for (let y = 0; y < size; y++) {
      for (let x = 0; x < size; x++) {
        const src = (y * size + x) * 4;
        for (let c = 0; c < 3; c++) {
          const v = (pixels[src + c] / 255 - MEAN[c]) / STD[c];
          data[c * plane + y * size + x] = v;                          // photo
          data[3 * plane + c * plane + y * size + (size - 1 - x)] = v; // mirror image
        }
      }
    }
    return new ort.Tensor("float32", data, [2, 3, size, size]);
  }

  function softmaxInto(sum, logits, offset, n) {
    let max = -Infinity;
    for (let i = 0; i < n; i++) max = Math.max(max, logits[offset + i]);
    let total = 0;
    const exps = new Array(n);
    for (let i = 0; i < n; i++) { exps[i] = Math.exp(logits[offset + i] - max); total += exps[i]; }
    for (let i = 0; i < n; i++) sum[i] += exps[i] / total;
  }

  // Average the probabilities of 2 models × 2 views, exactly like model.py's Predictor.predict.
  async function diagnoseBitmap(bitmap) {
    const start = performance.now();
    const n = catalog.length;
    const sum = new Float64Array(n);
    let rows = 0;
    for (let m = 0; m < MODELS.length; m++) {
      const output = await sessions[m].run({ image: toTensor(bitmap, MODELS[m].size) });
      const logits = output.logits.data;
      for (let r = 0; r < 2; r++) { softmaxInto(sum, logits, r * n, n); rows++; }
    }
    const probs = Array.from(sum, (v) => v / rows);
    const top3 = probs.map((p, i) => [i, p]).sort((a, b) => b[1] - a[1]).slice(0, 3);
    return { index: top3[0][0], confidence: top3[0][1], top3: top3, ms: performance.now() - start };
  }

  // ---------------------------------------------------------------- result page
  function el(tag, attrs, ...children) {
    const node = document.createElement(tag);
    for (const [key, value] of Object.entries(attrs || {})) node.setAttribute(key, value);
    for (const child of children) if (child != null) node.append(child);  // strings become text, never HTML
    return node;
  }
  function icon(name) {
    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.setAttribute("class", "icon");
    svg.setAttribute("aria-hidden", "true");
    const use = document.createElementNS("http://www.w3.org/2000/svg", "use");
    use.setAttribute("href", "icons.svg#" + name);
    svg.append(use);
    return svg;
  }
  function list(lines) {
    return lines.length > 1 ? el("ul", {}, ...lines.map((line) => el("li", {}, line))) : el("p", {}, lines[0] || "");
  }
  function percent(p, digits) { return (p * 100).toFixed(digits) + "%"; }

  function renderResult(result, photoUrl, sampleLabel) {
    const c = catalog[result.index];
    const chip = c.is_background ? el("span", { class: "chip chip--neutral" }, icon("info"), "Not a leaf")
      : c.is_healthy ? el("span", { class: "chip chip--ok" }, icon("check"), "Healthy")
      : el("span", { class: "chip chip--bad" }, icon("alert"), "Disease detected");
    const again = el("button", { class: "btn btn--primary", type: "button" }, icon("refresh"), "Diagnose another leaf");
    again.addEventListener("click", () => { $("#diagnose").scrollIntoView(); $("label[for=image]").focus(); });
    const print = el("button", { class: "btn btn--ghost", type: "button" }, "Print / save as PDF");
    print.addEventListener("click", () => window.print());

    const card = el("div", { class: "card verdict__card" }, chip,
      el("h2", { class: "verdict__title", id: "result-title" },
        c.crop ? el("span", { class: "verdict__crop" }, c.crop) : null, c.condition),
      el("p", { class: "verdict__confidence" }, "Model confidence ", el("strong", {}, percent(result.confidence, 0))),
      result.confidence < THRESHOLD ? el("div", { class: "banner banner--warn", role: "status" }, icon("alert"),
        el("p", {}, el("strong", {}, "The model isn't sure."), " Retake in daylight with a single leaf filling the frame.")) : null,
      c.is_background ? el("p", {}, "We couldn't find a leaf in this photo. Try again with one leaf filling most of the frame.") : null,
      el("p", { class: "hint" }, "Diagnosed on this device in " + (result.ms / 1000).toFixed(1) + " s (" + backend + ")."),
      el("p", { class: "hint", "data-disclaimer": "" }, "An automated first opinion, not professional advice. Confirm with a local extension officer before treating."),
      el("div", { class: "cluster no-print" }, again, print));

    const bars = el("ol", { class: "bars" }, ...result.top3.map(([i, p], rank) =>
      el("li", { class: "bars__row" },
        el("span", { class: "bars__label", id: "bar-" + rank }, catalog[i].label),
        el("progress", { class: "bars__bar" + (rank === 0 ? " bars__bar--top" : ""), max: "1", value: p.toFixed(4), "aria-labelledby": "bar-" + rank }),
        el("span", { class: "bars__value" }, percent(p, 1)))));

    resultBox.replaceChildren(
      el("p", { class: "eyebrow" }, "Diagnosis" + (sampleLabel ? " · sample photo (" + sampleLabel.toLowerCase() + ")" : "")),
      el("div", { class: "verdict" }, el("img", { class: "verdict__photo", src: photoUrl, alt: "Your leaf photo" }), card),
      el("section", { class: "card", "aria-labelledby": "top3-title" }, el("h3", { id: "top3-title", class: "h3" }, "Top 3 possibilities"), bars));

    if (!c.is_background) {
      resultBox.append(el("div", { class: "result__grid" },
        el("section", { class: "card prose" }, el("h3", { class: "h3" }, c.is_healthy ? "Growing tips" : "About this condition"), list(c.description)),
        el("section", { class: "card prose" }, el("h3", { class: "h3" }, c.is_healthy ? "Benefits" : "What to do"), list(c.steps))));
      if (c.product_link) {
        resultBox.append(el("section", {},
          el("h3", { class: "h3" }, c.is_healthy ? "Suggested fertiliser" : "Suggested treatment"),
          el("article", { class: "product card" },
            el("img", { class: "product__img", src: "img/products/" + c.product_kind + ".svg", alt: "", width: "96", height: "96" }),
            el("div", { class: "product__body" },
              el("p", { class: "eyebrow" }, (c.is_healthy ? "Fertiliser" : "Treatment") + " · " + c.label),
              el("h4", { class: "product__name" }, c.product_name),
              el("a", { class: "btn btn--ghost btn--sm", href: c.product_link, target: "_blank", rel: "noopener noreferrer" },
                icon("shopping-bag"), "View product", el("span", { class: "visually-hidden" }, " (opens a third-party shop in a new tab)")))),
          el("p", { class: "hint" }, "Product links are third-party. Follow label instructions and local agricultural advice.")));
      }
    }
    resultBox.hidden = false;
    resultBox.scrollIntoView({ behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth" });
    resultBox.focus({ preventScroll: true });
  }

  // ---------------------------------------------------------------- wiring
  async function diagnoseBlob(blob, sampleLabel) {
    if (!sessions) { errorBox.textContent = "The AI model is still loading. Please wait a moment."; return; }
    errorBox.textContent = "";
    let bitmap;
    try { bitmap = await createImageBitmap(blob); }  // applies the phone's EXIF rotation
    catch (e) { errorBox.textContent = "We couldn't read that file as an image. Please try another photo."; return; }
    const result = await diagnoseBitmap(bitmap);
    renderResult(result, URL.createObjectURL(blob), sampleLabel);
  }

  form.addEventListener("submit", async (event) => {
    event.preventDefault();                 // nothing is uploaded: the diagnosis happens here
    if (!input.files.length) return;        // diagnose.js already showed "Please choose a photo"
    try { await diagnoseBlob(input.files[0]); }
    finally { window.dispatchEvent(new Event("pageshow")); }  // makes diagnose.js reset its loading state
  });

  document.querySelectorAll("[data-sample]").forEach((button) => button.addEventListener("click", async () => {
    button.setAttribute("aria-busy", "true");
    try {
      const blob = await (await fetch("img/samples/" + button.dataset.sample + ".jpg")).blob();
      await diagnoseBlob(blob, button.textContent.trim());
    } finally { button.removeAttribute("aria-busy"); }
  }));

  // Leaving the page aborts the model download; that is not a failure worth reporting.
  let leaving = false;
  addEventListener("pagehide", () => { leaving = true; });
  addEventListener("beforeunload", () => { leaving = true; });

  // Resolves to true when the models are ready, false if loading failed (the message is shown on the page).
  const ready = load().then(() => true, (error) => {
    if (leaving) return false;
    console.error(error);
    status.dataset.state = "error";
    statusText.textContent = "The AI model couldn't load. Check your connection and reload the page.";
    statusHint.textContent = String(error.message || error);
    return false;
  });

  // Used by the automated tests (web/test_web.py) to diagnose many photos without clicking.
  window.LeafScan = { ready: ready, diagnoseBitmap: (bitmap) => diagnoseBitmap(bitmap), backend: () => backend };
})();
