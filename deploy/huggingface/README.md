---
title: LeafScan
emoji: 🌿
colorFrom: green
colorTo: green
sdk: docker
app_port: 7860
license: mit
short_description: Leaf disease diagnosis from one photo
---

# LeafScan on Hugging Face Spaces

> **Note (November 2026):** Hugging Face now requires a PRO subscription for Docker and Gradio
> Spaces; only static Spaces are free. The free live demo is the browser version on GitHub Pages
> (`web/`). These files are for running the full Flask app on Spaces with PRO. `app.py` +
> `requirements.txt` are the no-Docker alternative for a Gradio Space.

This folder holds the two files of the LeafScan Space: this `README.md` (the Space settings in the
front matter above) and the `Dockerfile`.

The Dockerfile clones a tagged release of
[eshdobhal/LEAFSCAN-PLANT-DISEASE-DETECTION](https://github.com/eshdobhal/LEAFSCAN-PLANT-DISEASE-DETECTION)
(weights included), installs CPU-only PyTorch and runs the app with gunicorn on port 7860.

**To deploy a new version:** tag a release on GitHub, change `LEAFSCAN_REF` in the Space's Dockerfile to
that tag, and commit. The Space rebuilds automatically.

Settings used on Spaces (set in the Dockerfile):

| Variable | Value | Why |
|---|---|---|
| `REQUIRE_MODEL` | `1` | Never start in demo mode |
| `TRUST_PROXY` | `1` | Spaces sit behind an HTTPS proxy: real client IP and https URLs |
| `FRAME_ANCESTORS` | `https://huggingface.co` | The Space page shows the app in a frame |
| `RATE_LIMIT_PER_MINUTE` | `60` | Close to what 2 vCPUs can serve anyway |

Author: Esh Dobhal · MIT License
