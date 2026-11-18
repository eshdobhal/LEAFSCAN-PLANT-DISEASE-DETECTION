// Treatments page: filter product cards by search text, crop and product type.
// Each card carries data-search, data-crop and data-kind attributes (see partials/macros.html).
(function () {
  "use strict";

  const filters = document.querySelector("[data-filters]");
  if (!filters) return;
  const cards = [...document.querySelectorAll("[data-products] > .product")];
  const search = filters.querySelector("[data-search]");
  const count = filters.querySelector("[data-count]");
  const empty = document.querySelector("[data-empty]");
  const state = { text: "", crop: "all", kind: "all" };

  function update() {
    let shown = 0;
    cards.forEach((card) => {
      const match =
        (state.crop === "all" || card.dataset.crop === state.crop) &&
        (state.kind === "all" || card.dataset.kind === state.kind) &&
        card.dataset.search.includes(state.text);
      card.hidden = !match;
      if (match) shown += 1;
    });
    count.textContent = shown + (shown === 1 ? " product" : " products");  // read out by aria-live
    empty.hidden = shown > 0;
  }

  // A group of toggle buttons where exactly one is pressed (aria-pressed tells screen readers).
  function buttonGroup(selector, key) {
    const buttons = [...filters.querySelectorAll(selector)];
    buttons.forEach((button) =>
      button.addEventListener("click", () => {
        buttons.forEach((b) => b.setAttribute("aria-pressed", String(b === button)));
        state[key] = button.dataset[key];
        update();
      }));
  }

  buttonGroup("[data-kind]", "kind");
  buttonGroup("[data-crop]", "crop");
  search.addEventListener("input", () => {
    state.text = search.value.trim().toLowerCase();
    update();
  });
})();
