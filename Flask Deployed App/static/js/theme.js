// Runs on every page: colour-theme toggle and the phone menu button.
(function () {
  "use strict";
  var root = document.documentElement;

  // --- Theme: cycles system -> light -> dark -> system. Choice saved in localStorage. ---
  var toggle = document.querySelector(".theme-toggle");
  var ORDER = ["system", "light", "dark"];

  function currentMode() {
    return root.dataset.theme || "system";
  }

  function apply(mode) {
    if (mode === "system") delete root.dataset.theme;
    else root.dataset.theme = mode;
    try {
      if (mode === "system") localStorage.removeItem("theme");
      else localStorage.setItem("theme", mode);
    } catch (e) { /* storage blocked (private mode): the theme still applies for this page */ }
    if (toggle) {
      toggle.dataset.mode = mode;
      toggle.setAttribute("aria-label", "Theme: " + mode + ". Click to change.");
    }
  }

  if (toggle) {
    apply(currentMode());
    toggle.addEventListener("click", function () {
      var next = ORDER[(ORDER.indexOf(currentMode()) + 1) % ORDER.length];
      apply(next);
    });
  }

  // --- Phone menu: the button shows/hides the nav (CSS does the hiding). ---
  var menu = document.querySelector(".menu-toggle");
  if (menu) {
    menu.addEventListener("click", function () {
      var open = menu.getAttribute("aria-expanded") === "true";
      menu.setAttribute("aria-expanded", String(!open));
    });
    // Escape closes the menu and returns focus to the button.
    document.addEventListener("keydown", function (event) {
      if (event.key === "Escape" && menu.getAttribute("aria-expanded") === "true") {
        menu.setAttribute("aria-expanded", "false");
        menu.focus();
      }
    });
  }
})();
