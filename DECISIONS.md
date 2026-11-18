# LeafScan — design & engineering decisions

This file records every non-obvious choice made while rebuilding the original
Plant-Disease-Detection Flask app as **LeafScan**. It is filled in phase by phase.

## Plan (phase 1)

1. Baseline commit of the untouched original on `master`; all work on branch `redesign`.
2. Rewrite `requirements.txt` as UTF-8 with current, CPU-only PyTorch; add `requirements-dev.txt` (pytest, Playwright).
3. Split `app.py` into `config.py`, `catalog.py`, `model.py` and a small `app.py` with `create_app()`.
4. `catalog.py`: read both CSVs once (cp1252 → str), build `Condition` dataclasses with derived `is_healthy` / `is_background`, list-ified text, display-name fixes.
5. `model.py`: load weights once (`map_location="cpu"`, `weights_only=True`), training-identical preprocessing, softmax + top-3, `inference_mode`, warm-up; demo mode with random weights if the file is missing.
6. Uploads handled in memory only, validated by extension + Pillow, 8 MB cap, result photo returned as a ≤512 px data URL.
7. Routes: `/`, `/diagnose` (GET/POST), `/treatments`, `/about`, `/api/predict`, `/healthz`; 301s for `/index`, `/market`, `/contact`; `/submit` kept as a compatibility shim.
8. Security headers (CSP with per-request nonce), long static caching with `?v=` busting, `logging` instead of `print`.
9. pytest suite covering pages, redirects, uploads, API schema, error paths.
10. One hand-written CSS file with design tokens, light/dark themes, fluid type, self-hosted Fraunces + Inter.
11. Jinja `base.html` + partials; mobile-first header with progressive-enhancement menu; theme toggle.
12. Diagnose page: drag-drop, paste, camera dialog, client-side downscale, loading state — all degrading to a plain form without JS.
13. Result, Treatments (vanilla-JS filters), About, error pages; local SVG crop/product art, favicons, manifest, OG image.
14. Playwright screenshots (375/1440 px × light/dark) → self-review → fixes; accuracy + latency benchmarks.
15. README, DECISIONS, CODE_WALKTHROUGH.

## Environment notes

- **Python 3.13 instead of 3.11.** This machine only has Python 3.10 and 3.13 installed
  (`py -0`). Everything here is plain Python and works on 3.11–3.13; the virtualenv used
  for development and testing is 3.13.
- **Model weights were not present** in the repo or anywhere under `Downloads`. See
  "Demo mode" below.
- **Node.js is not installed**, so `npx lighthouse` could not be run.
