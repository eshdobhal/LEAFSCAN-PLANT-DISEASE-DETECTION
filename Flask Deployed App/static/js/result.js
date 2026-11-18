// Result page: print button, animated confidence bars, and hiding a broken reference photo.
(function () {
  "use strict";

  const printBtn = document.querySelector("[data-print]");
  if (printBtn) printBtn.addEventListener("click", () => window.print());

  // The reference photo is hot-linked from another site and may have disappeared.
  // (An inline onerror="" attribute would be blocked by our Content-Security-Policy.)
  const ref = document.querySelector("[data-ref]");
  const refImg = document.querySelector("[data-ref-img]");
  if (ref && refImg) {
    const hide = () => { ref.hidden = true; };
    refImg.addEventListener("error", hide);
    if (refImg.complete && refImg.naturalWidth === 0 && refImg.currentSrc) hide();  // failed before we got here
  }

  // Grow each <progress> bar from 0 to its value once, unless the user prefers reduced motion.
  if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
  const bars = [...document.querySelectorAll(".bars__bar")];
  const targets = bars.map((bar) => Number(bar.value));
  const DURATION = 600;  // ms
  let start = null;
  function frame(now) {
    if (start === null) start = now;
    const t = Math.min(1, (now - start) / DURATION);
    const eased = 1 - Math.pow(1 - t, 3);  // ease-out
    bars.forEach((bar, i) => { bar.value = targets[i] * eased; });
    if (t < 1) requestAnimationFrame(frame);
  }
  bars.forEach((bar) => { bar.value = 0; });
  requestAnimationFrame(frame);
})();
