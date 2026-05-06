# XADE Backend — Deployment Guide (Railway)

## Overview

The XADE backend is a FastAPI app containerised with Docker. It runs the full
grounded pipeline: EfficientNet-B4 detection → LayerCAM → BiSeNet face parsing
→ forensic features → VLM explanation. All ML weights are lazy-downloaded on
first request; no weights are baked into the image.

**Hosting choice: Railway** (~$5–10/mo, shared CPU, 2 GB RAM)
Hugging Face Spaces is the fallback if cost becomes an issue (free, ~30 s cold
start after idle).

---

## First Deploy

### 1. Create a Railway project

1. Go to [railway.app](https://railway.app) → **New Project** → **Deploy from
   GitHub repo** → select `XADE`.
2. Set the **Root Directory** to `backend/`.
3. Railway auto-detects the `Dockerfile` — no extra build config needed.

### 2. Set environment variables in Railway

In the Railway project → **Variables**, add:

| Variable | Value |
|---|---|
| `ANTHROPIC_API_KEY` | your Anthropic key |
| `OPENAI_API_KEY` | your OpenAI key |
| `GOOGLE_GEMINI_API_KEY` | your Gemini key |
| `SUPABASE_URL` | your Supabase project URL |
| `SUPABASE_KEY` | Supabase **service role** key (not the anon key) |
| `CORS_ORIGINS` | comma-separated Vercel URLs (see below) |
| `VLM_DEFAULT_PROVIDER` | `google` |

**CORS_ORIGINS example** (fill in your real Vercel domain once known):
```
https://xade.vercel.app,https://xade-git-main-viktorahnstrom.vercel.app
```
All `https://xade*.vercel.app` preview URLs are also allowed automatically via
regex — no need to add every PR preview individually.

### 3. Deploy

Railway deploys automatically on every push to `main`. The first deploy
downloads BiSeNet weights (~50 MB) and the MediaPipe model (~6 MB) at startup.
Allow ~3–5 min for the first cold build; subsequent deploys are faster.

### 4. Record the URL

Once deployed, copy the Railway public URL (e.g.
`https://xade-backend-production.up.railway.app`) and:

- Add it to `docs/user_study_plan.md`.
- Set `VITE_API_BASE_URL` to this URL in the Vercel project environment
  variables so the frontend hits the live backend for post-study product trials.

---

## Verifying a deploy

```bash
# Should return 200 within 2 s on a warm container
curl https://<your-railway-url>/api/v1/health
```

Expected response:
```json
{
  "status": "healthy",
  "database": "healthy",
  "detection_model": "loaded",
  "vlm_service": "initialized"
}
```

---

## Redeploy

Railway redeploys automatically on every push to `main`. To trigger a manual
redeploy (e.g. after rotating a secret): Railway dashboard → **Deployments** →
**Redeploy**.

---

## Secret rotation

1. Update the variable value in Railway → **Variables**.
2. Click **Redeploy** to pick up the new value.
3. Never commit secrets to the repo — `.env` is gitignored.

---

## Switching to Hugging Face Spaces (fallback)

If Railway cost is a problem after the study:

1. Create a new Space at [huggingface.co/spaces](https://huggingface.co/spaces)
   → **Docker** SDK.
2. Push the `backend/` directory as the Space repo.
3. Set the same environment variables in the Space **Settings → Repository
   secrets**.
4. Update `VITE_API_BASE_URL` in Vercel to the new Space URL.

Note: HF Spaces free tier has a ~30 s cold start after idle. For the thesis
defence demo this is acceptable; for an active study window consider the
persistent paid tier ($9/mo).

---

## Rebuilding study assets

If the 12 study images or VLM explanations change, regenerate and redeploy:

```bash
# From backend/ with venv active and backend running locally
curl -X POST http://localhost:8000/api/v1/study/precompute

# Commit the updated files
git add desktop/public/study-analyses.json desktop/public/quiz-heatmaps/
git commit -m "chore: regenerate study assets"
git push
```

Vercel redeploys automatically; Railway is not involved (study assets are
served as static files from the Vercel bundle).
