---
title: Assert Real
emoji: 🔍
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# Assert Real

AI-powered deepfake detection with explainability. Upload an image and get a
prediction with visual evidence (GradCAM heatmaps, ELA, evidence crops) and
optional natural-language explanations from a VLM.

## How it works

1. Upload a face image (JPEG, PNG, or WebP, max 25 MB).
2. The EfficientNet-B4 model classifies it as real or fake.
3. GradCAM highlights the regions the model focused on.
4. Optionally, a VLM explains the result in plain language.

## Architecture

Single-container deployment: FastAPI serves both the React frontend and the
detection API on one port. Model weights are baked into the image at build time
so cold starts after the Space wakes from sleep do not require network access.

## Limitations

- Free CPU tier (2 vCPU, 16 GB RAM). Inference is slower than GPU.
- Space sleeps after 48 hours of inactivity. First request after sleep takes
  ~2 minutes while the container restarts.
- GradCAM outputs are stored in /tmp and lost on restart.
- VLM explanations require API keys to be set in the Space secrets.
