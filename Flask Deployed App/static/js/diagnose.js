// Diagnose page. The HTML form already works without this file; this script adds
// preview, drag-and-drop, paste, in-browser resizing, the camera, and a loading state.
(function () {
  "use strict";

  const MAX_BYTES = 8 * 1024 * 1024;  // same limit as the server (config.MAX_CONTENT_LENGTH)
  const MAX_SIDE = 1024;              // resize big photos so slow connections send ~200 KB, not 6 MB
  const TYPES = ["image/jpeg", "image/png", "image/webp"];

  const form = document.querySelector("[data-upload-form]");
  if (!form) return;
  const $ = (selector) => document.querySelector(selector);
  const input = $("#image");
  const dropzone = $("[data-dropzone]");
  const preview = $("[data-preview]");
  const previewImg = $("[data-preview-img]");
  const errorBox = $("[data-error]");
  const submitBtn = $("[data-submit]");
  const submitLabel = $("[data-submit-label]");
  let previewUrl = null;
  let submitting = false;

  const showError = (message) => { errorBox.textContent = message; };
  const formatSize = (bytes) =>
    bytes < 1024 * 1024 ? Math.max(1, Math.round(bytes / 1024)) + " KB" : (bytes / 1048576).toFixed(1) + " MB";

  // Put a File into the real <input type="file">, so a normal form submit uploads it.
  function setInputFile(file) {
    const transfer = new DataTransfer();
    transfer.items.add(file);
    input.files = transfer.files;
  }

  function clearFile() {
    input.value = "";
    preview.hidden = true;
    submitBtn.disabled = true;
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    previewUrl = null;
  }

  function showPreview(file) {
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    previewUrl = URL.createObjectURL(file);
    previewImg.src = previewUrl;
    $("[data-preview-name]").textContent = file.name;
    $("[data-preview-size]").textContent = formatSize(file.size) + " · ready";
    preview.hidden = false;
    submitBtn.disabled = false;
  }

  // Shrink the photo so its longest side is at most MAX_SIDE px, re-encoded as JPEG (quality 0.9).
  // createImageBitmap applies the phone's EXIF rotation for us. On any failure, send the original.
  async function downscale(file) {
    if (!window.createImageBitmap) return file;
    try {
      const bitmap = await createImageBitmap(file);
      const scale = Math.min(1, MAX_SIDE / Math.max(bitmap.width, bitmap.height));
      if (scale === 1 && file.size < 1024 * 1024) return file;  // already small: keep it untouched
      const canvas = document.createElement("canvas");
      canvas.width = Math.round(bitmap.width * scale);
      canvas.height = Math.round(bitmap.height * scale);
      const ctx = canvas.getContext("2d");
      ctx.fillStyle = "#fff";  // transparent PNG areas become white, not black
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(bitmap, 0, 0, canvas.width, canvas.height);
      bitmap.close();
      const blob = await new Promise((resolve) => canvas.toBlob(resolve, "image/jpeg", 0.9));
      if (!blob) return file;
      const name = file.name.replace(/\.[^.]+$/, "") + ".jpg";
      return new File([blob], name, { type: "image/jpeg" });
    } catch (e) {
      return file;
    }
  }

  // Every way of choosing a photo (picker, drop, paste, camera) ends up here.
  async function handleFile(file) {
    showError("");
    if (!file) return;
    if (!TYPES.includes(file.type)) {
      clearFile();
      showError("That file type isn't supported. Please use a JPG, PNG or WebP photo.");
      return;
    }
    submitBtn.disabled = true;
    submitLabel.textContent = "Preparing photo…";
    const ready = await downscale(file);
    submitLabel.textContent = "Diagnose leaf";
    if (ready.size > MAX_BYTES) {
      clearFile();
      showError("That photo is " + formatSize(ready.size) + ". The limit is 8 MB. Please choose a smaller photo.");
      return;
    }
    if (ready !== file || input.files[0] !== file) setInputFile(ready);
    showPreview(ready);
  }

  // --- Choosing, dropping and pasting ---
  input.addEventListener("change", () => handleFile(input.files[0]));

  $("[data-remove]").addEventListener("click", () => {
    clearFile();
    input.focus();  // focus lands on the (visually hidden) input, whose label shows the focus ring
  });

  ["dragenter", "dragover"].forEach((type) =>
    dropzone.addEventListener(type, (event) => {
      event.preventDefault();
      dropzone.classList.add("is-dragover");
    }));
  ["dragleave", "drop"].forEach((type) =>
    dropzone.addEventListener(type, () => dropzone.classList.remove("is-dragover")));
  dropzone.addEventListener("drop", (event) => {
    event.preventDefault();
    handleFile(event.dataTransfer.files[0]);
  });
  // Dropping a file anywhere else would make the browser open it and leave the page.
  window.addEventListener("dragover", (event) => event.preventDefault());
  window.addEventListener("drop", (event) => event.preventDefault());

  document.addEventListener("paste", (event) => {
    const item = [...(event.clipboardData ? event.clipboardData.items : [])]
      .find((i) => i.kind === "file" && i.type.startsWith("image/"));
    if (item) handleFile(item.getAsFile());
  });

  // --- Submitting: show progress and block double submits ---
  form.addEventListener("submit", (event) => {
    if (!input.files.length) {
      event.preventDefault();
      showError("Please choose a photo of a leaf first.");
      return;
    }
    if (submitting) {
      event.preventDefault();
      return;
    }
    submitting = true;
    submitBtn.setAttribute("aria-busy", "true");
    submitBtn.disabled = true;
    submitLabel.textContent = "Analysing leaf…";
  });

  // Coming back with the Back button restores the page from cache: undo the loading state.
  window.addEventListener("pageshow", () => {
    submitting = false;
    submitBtn.removeAttribute("aria-busy");
    submitLabel.textContent = "Diagnose leaf";
    submitBtn.disabled = !input.files.length;
  });

  // Start state: nothing chosen yet (or a file the browser kept after Back).
  if (input.files.length) handleFile(input.files[0]);
  else submitBtn.disabled = true;

  // --- Camera ---
  const dialog = $("[data-camera]");
  const openBtn = $("[data-camera-open]");
  if (!dialog || typeof dialog.showModal !== "function") return;  // very old browser: no camera button
  const video = $("[data-camera-video]");
  const canvas = $("[data-camera-canvas]");
  const status = $("[data-camera-status]");
  const captureBtn = $("[data-camera-capture]");
  const retakeBtn = $("[data-camera-retake]");
  const useBtn = $("[data-camera-use]");
  let stream = null;

  function showStep(captured) {
    video.hidden = captured;
    canvas.hidden = !captured;
    captureBtn.hidden = captured;
    retakeBtn.hidden = !captured;
    useBtn.hidden = !captured;
  }

  function stopCamera() {
    if (stream) stream.getTracks().forEach((track) => track.stop());  // turns the camera light off
    stream = null;
    video.srcObject = null;
  }

  async function openCamera() {
    showStep(false);
    captureBtn.disabled = true;
    dialog.showModal();
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      status.textContent = "The live camera isn't available here (it needs HTTPS). Use Choose photo: on a phone it can open the camera too.";
      return;
    }
    status.textContent = "Starting camera…";
    try {
      // "environment" = the back camera on phones.
      stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment" }, audio: false });
      if (!dialog.open) return stopCamera();  // closed while the permission prompt was showing
      video.srcObject = stream;
      await video.play();
      captureBtn.disabled = false;
      status.textContent = "Fill the frame with one leaf, then press Capture.";
    } catch (error) {
      const name = error && error.name;
      status.textContent =
        name === "NotAllowedError" ? "Camera permission was denied. Allow it in your browser settings, or use Choose photo instead."
        : name === "NotFoundError" || name === "OverconstrainedError" ? "No camera was found on this device. Use Choose photo instead."
        : "Couldn't start the camera. Use Choose photo instead.";
    }
  }

  openBtn.hidden = false;
  openBtn.addEventListener("click", openCamera);
  $("[data-camera-close]").addEventListener("click", () => dialog.close());
  dialog.addEventListener("close", stopCamera);  // also fires when Escape closes the dialog

  captureBtn.addEventListener("click", () => {
    if (!video.videoWidth) return;  // camera not ready yet
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext("2d").drawImage(video, 0, 0);
    showStep(true);
    status.textContent = "Happy with it? Use the photo, or retake.";
    useBtn.focus();
  });

  retakeBtn.addEventListener("click", () => {
    showStep(false);
    status.textContent = "Fill the frame with one leaf, then press Capture.";
    captureBtn.focus();
  });

  useBtn.addEventListener("click", () => {
    canvas.toBlob(async (blob) => {
      dialog.close();
      if (blob) await handleFile(new File([blob], "camera-photo.jpg", { type: "image/jpeg" }));
      submitBtn.focus();  // next step for keyboard users: submit
    }, "image/jpeg", 0.92);
  });
})();
