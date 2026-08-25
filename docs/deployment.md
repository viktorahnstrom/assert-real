# Deployment

Assert Real supports two deployment targets. Both use the same backend code and
model weights; they differ in how TLS, static files, and the frontend are served.

---

## 1. Hugging Face Spaces (current, free)

Single Docker container running on HF's free CPU tier (2 vCPU, 16 GB RAM).
FastAPI serves both the React frontend and the API on one port. HF terminates
TLS.

### What it is

A public demo. The Space sleeps after 48 hours of inactivity; the first
request after sleep takes ~2 minutes while the container restarts. GradCAM
outputs are stored in /tmp and lost on restart.

### Setup

1. Create a new Space on Hugging Face with `sdk: docker`.

2. Copy `Dockerfile.spaces` to `Dockerfile` in the Space repo root, and copy
   `README.spaces.md` to `README.md`. Copy the `backend/`, `desktop/`,
   `shared/`, `mobile/package.json`, `package.json`, `package-lock.json`, and
   `tsconfig.json` into the Space repo.

3. Set these secrets in the Space's Settings UI:

   | Secret | Description |
   |--------|-------------|
   | `SUPABASE_URL` | Your Supabase project URL |
   | `SUPABASE_ANON_KEY` | Supabase anon (public) key |
   | `VITE_SUPABASE_URL` | Same as SUPABASE_URL (passed to frontend build) |
   | `VITE_SUPABASE_ANON_KEY` | Same as SUPABASE_ANON_KEY (passed to frontend build) |

   Optional (VLM explanations):

   | Secret | Description |
   |--------|-------------|
   | `GOOGLE_GEMINI_API_KEY` | Gemini API key |
   | `VLM_DEFAULT_PROVIDER` | `google`, `openai`, `anthropic`, or `mock` |
   | `OPENAI_API_KEY` | OpenAI API key (if using OpenAI) |
   | `ANTHROPIC_API_KEY` | Anthropic API key (if using Anthropic) |

   **Never set `SUPABASE_SERVICE_ROLE_KEY` in the Space.** Only the anon key
   should reach the image.

4. Push. HF builds the image from `Dockerfile` and starts the container.

### How it works

- `Dockerfile.spaces` is a three-stage build: frontend (Vite), Python deps
  (CPU-only torch), and runtime.
- The container runs as UID 1000 (HF requirement).
- `SERVE_FRONTEND=true` tells FastAPI to serve the React build from
  `/app/frontend` with SPA fallback. API routes (`/api/*`, `/gradcam/*`,
  `/health`) take priority over the catch-all.
- All model weights are baked in at build time via `prefetch_models.py`.
  The container never needs network access to serve a request.
- Cache directories point to `/tmp` (disposable). `/data` persists across
  restarts if a storage bucket is attached, but is not used by default.

### Rate limiting

HF puts a reverse proxy in front of the container. The backend runs uvicorn
with `--proxy-headers --forwarded-allow-ips '*'`, which rewrites
`request.client.host` from `X-Forwarded-For`. slowapi's `get_remote_address`
reads `request.client.host`, so per-IP rate limiting works correctly.

### CORS

Same-origin deployment (frontend and API on the same port), so the default
localhost dev origins are all that's needed. No extra `CORS_ORIGINS` required.

---

## 2. Single VPS with Caddy (ready, needs a paid host)

Three containers behind Caddy: backend (FastAPI), web (nginx serving the React
build), and Caddy for TLS termination and reverse proxying.

### What it is

The production deployment path. Caddy auto-provisions TLS via Let's Encrypt.
The backend and web containers are internal-only; only Caddy publishes ports
80/443.

### Setup

1. Provision a VPS (4 vCPU, 8 GB RAM recommended). Run the bootstrap script:

   ```bash
   ssh root@<server> 'bash -s' < scripts/bootstrap.sh
   ```

   This creates a `deploy` user, hardens SSH, configures ufw, sets up swap,
   and installs Docker.

2. Copy files to the server:

   ```bash
   scp .env docker-compose.prod.yml Caddyfile deploy@<server>:/opt/assert-real/
   ```

3. Create `.env` from `backend/.env.example`. Set `DOMAIN`, `ACME_EMAIL`,
   and all Supabase/VLM keys.

4. Start the stack:

   ```bash
   ssh deploy@<server> 'cd /opt/assert-real && docker compose -f docker-compose.prod.yml up -d'
   ```

### How it works

- `docker-compose.prod.yml` defines three services: `caddy`, `backend`, `web`.
- Caddy handles TLS, security headers, 25 MB upload limit, and proxies
  `/api/*` and `/gradcam/*` to the backend. Everything else goes to the web
  container (nginx with SPA fallback).
- Images are built and pushed to GHCR by `.github/workflows/deploy.yml` on
  merge to main, then pulled on the VPS via SSH.
- The backend Dockerfile (`backend/Dockerfile`) bakes in all model weights,
  runs as a non-root user (UID 10001), and includes a Docker HEALTHCHECK.
- `SERVE_FRONTEND` is not set (defaults to false) — nginx handles the
  frontend.

### CI/CD

- `.github/workflows/ci.yml` — lint, typecheck, pytest, Docker build check.
- `.github/workflows/deploy.yml` — build + push to GHCR, SSH deploy, health
  check with retries.

Configure these GitHub repo secrets for deploy:

| Secret | Description |
|--------|-------------|
| `DEPLOY_HOST` | VPS IP or hostname |
| `DEPLOY_USER` | SSH user (default: `deploy`) |
| `DEPLOY_SSH_KEY` | Private SSH key for the deploy user |
| `VITE_SUPABASE_URL` | Passed to frontend build |
| `VITE_SUPABASE_ANON_KEY` | Passed to frontend build |
