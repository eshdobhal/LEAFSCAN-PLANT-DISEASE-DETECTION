// Runs before the page paints (loaded without defer in <head>): mark that JavaScript is on and apply a
// saved colour theme, so dark-mode users never see a white flash. The Flask app does the same inline.
document.documentElement.classList.add("js");
try {
  var theme = localStorage.getItem("theme");
  if (theme === "light" || theme === "dark") document.documentElement.dataset.theme = theme;
} catch (e) { /* storage blocked: system theme */ }
