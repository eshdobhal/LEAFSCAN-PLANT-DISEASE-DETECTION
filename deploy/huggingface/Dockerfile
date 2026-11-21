# LeafScan on Hugging Face Spaces (Docker SDK, free CPU: 2 vCPU, 16 GB RAM).
# Builds the app from a tagged GitHub release, so the Space always runs a tested version.
FROM python:3.11-slim

ARG LEAFSCAN_REF=v1.0.1
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

# Only the web app (trained weights included), not the training code, screenshots or test images.
RUN git clone --depth 1 --branch "$LEAFSCAN_REF" https://github.com/eshdobhal/LEAFSCAN-PLANT-DISEASE-DETECTION.git /tmp/src && mv "/tmp/src/Flask Deployed App" /app && rm -rf /tmp/src

WORKDIR /app
RUN pip install -r requirements.txt

# Spaces run containers as user 1000.
RUN useradd -m -u 1000 user
USER user

# REQUIRE_MODEL: never start in demo mode. TRUST_PROXY: Spaces sit behind an HTTPS proxy.
# FRAME_ANCESTORS: the Space page shows the app in a frame on huggingface.co.
ENV REQUIRE_MODEL=1 TRUST_PROXY=1 FRAME_ANCESTORS=https://huggingface.co RATE_LIMIT_PER_MINUTE=60
EXPOSE 7860
CMD ["gunicorn", "app:create_app()", "--bind", "0.0.0.0:7860", "--workers", "1", "--threads", "4", "--timeout", "120"]
