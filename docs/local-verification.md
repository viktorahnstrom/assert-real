# Local Verification Runbook

Run the full production stack locally against the GHCR images CI already built.

> **Apple Silicon limitation:** The GHCR images are linux/amd64 only.
> `docker-compose.local.yml` sets `platform: linux/amd64` so Docker Desktop
> runs them under Rosetta/QEMU emulation. However, **MediaPipe's native
> binary requires AVX instructions**, which emulation does not provide. The
> backend process dies with SIGILL during startup (exit code 132). This is
> not a code bug and will not happen on a real amd64 VPS.
>
> **Consequence:** On Apple Silicon, checks 2-11 (anything requiring the
> backend to be running) can only be verified against a real deployment.
> Check 1 (prefetch with `--network none`) also fails for the same reason.
> Only the web container and Caddy routing can be tested locally.
>
> The checks below are written for a real amd64 host. Run them on the VPS
> after provisioning, or on an amd64 CI runner.

---

## Prerequisites

### Authenticate with GHCR

The packages are private. Create a GitHub PAT with `read:packages` scope, then:

```bash
echo "<YOUR_PAT>" | docker login ghcr.io -u viktorahnstrom --password-stdin
```

### Set GitHub repo secrets for the web image

The web image bakes `VITE_SUPABASE_URL` and `VITE_SUPABASE_ANON_KEY` at build
time. If these secrets are not set in the GitHub repo, the frontend will load
but Supabase auth will fail silently (empty config).

Go to repo Settings > Secrets and variables > Actions, and add:

- `VITE_SUPABASE_URL` — your Supabase project URL
- `VITE_SUPABASE_ANON_KEY` — your Supabase anon key

Then re-run the Deploy workflow (or push a commit) so CI rebuilds the web
image with the values baked in. Pull the new image before testing:

```bash
docker pull ghcr.io/viktorahnstrom/assert-real-web:latest
```

### Create .env.local

Create `.env.local` in the repo root (gitignored):

```bash
# ── Supabase ─────────────────────────────────────────────────────────────────
# Get these from: https://supabase.com/dashboard/project/<id>/settings/api
SUPABASE_URL=https://<your-project-id>.supabase.co
SUPABASE_ANON_KEY=<your-anon-key>

# ── VLM (optional — set mock to skip) ───────────────────────────────────────
VLM_DEFAULT_PROVIDER=mock

# ── Rate limiting (low values for easy testing) ─────────────────────────────
RATE_LIMIT_ANALYSES=3/minute
RATE_LIMIT_DEFAULT=60/hour

# ── CORS ─────────────────────────────────────────────────────────────────────
# Caddy proxies everything on localhost, so the browser sees same-origin.
# But add https://localhost in case any direct calls bypass the proxy.
CORS_ORIGINS=https://localhost
```

**Differences from VPS .env:**

| Key | Local | VPS |
|-----|-------|-----|
| `DOMAIN` | Not needed (Caddyfile.local hardcodes `localhost`) | Required |
| `ACME_EMAIL` | Not needed (internal CA) | Required |
| `CORS_ORIGINS` | `https://localhost` | `https://assertreal.dev` |
| `RATE_LIMIT_ANALYSES` | `3/minute` (easy to test) | `10/hour` |
| `VLM_DEFAULT_PROVIDER` | `mock` (no API keys needed) | `google` or other |

The service role key must not appear in this file.

### Accept the self-signed cert

Caddy's internal CA issues a cert for `localhost`. Your browser will show a
security warning on first visit. In Chrome: click "Advanced" > "Proceed to
localhost (unsafe)". In Firefox: click "Advanced" > "Accept the Risk and
Continue". This is expected.

Supabase JS client calls go to your Supabase project URL (not localhost), so
the self-signed cert does not affect auth or database calls.

---

## Start the stack

```bash
docker compose -f docker-compose.local.yml up -d
```

Wait ~2 minutes for the backend model loading (120s HEALTHCHECK start period).

---

## Checks

### 1. Prefetch: model loaders work offline

Run before starting the stack (or in a separate terminal). This verifies the
baked-in weights are in the paths the app reads from.

```bash
# Detection model
docker run --rm --network none \
  ghcr.io/viktorahnstrom/assert-real-backend:latest \
  python -c "from app.utils.model_loader import load_model_checkpoint; load_model_checkpoint(); print('OK')"

# Face parser (BiSeNet)
docker run --rm --network none \
  ghcr.io/viktorahnstrom/assert-real-backend:latest \
  python -c "from facexlib.parsing import init_parsing_model; init_parsing_model(model_name='bisenet', device='cpu'); print('OK')"

# Face category mapper (MediaPipe)
docker run --rm --network none \
  ghcr.io/viktorahnstrom/assert-real-backend:latest \
  python -c "from app.services.face_category_mapper import FaceCategoryMapper; m = FaceCategoryMapper(); m.close(); print('OK')"
```

**Expected:** Each prints `OK` and exits 0.

**Failure:** `ConnectionError`, `OSError`, or download progress bars. Means the
prefetch wrote to a different cache path than the app reads from. Check
`HF_HOME`, `TORCH_HOME`, and `XDG_CACHE_HOME` in the Dockerfile vs what the
loader expects.

### 2. Container HEALTHCHECK reports healthy

```bash
# Wait for the start period, then check
sleep 130
docker ps --format 'table {{.Names}}\t{{.Status}}' | grep assert-real
```

**Expected:** Backend container shows `(healthy)`.

**Failure:** `(unhealthy)` or `(health: starting)` after 3+ minutes. Check
backend logs: `docker compose -f docker-compose.local.yml logs backend --tail 50`

### 3. Liveness probe (/health)

```bash
curl -ksS https://localhost/health
```

**Expected:** `{"status":"ok"}` — 200, no database contact.

**Failure:** 502 means Caddy can't reach the backend. Check the backend
container is running and on the `edge` network.

### 4. Readiness check (/api/v1/health)

```bash
curl -ksS https://localhost/api/v1/health | python3 -m json.tool
```

**Expected:**

```json
{
    "status": "healthy",
    "database": "healthy",
    "detection_model": "loaded",
    "vlm_service": "initialized"
}
```

**Failure:** `"database": "unhealthy: ..."` means `SUPABASE_URL` or
`SUPABASE_ANON_KEY` is wrong in `.env.local`. `"detection_model": "not_loaded"`
means prefetch failed (see check 1).

### 5. Frontend loads with SPA fallback

```bash
# Homepage loads
curl -ksS -o /dev/null -w '%{http_code}' https://localhost/
# Expected: 200

# Deep link returns HTML (SPA fallback), not 404
curl -ksS -o /dev/null -w '%{http_code}' https://localhost/some/deep/route
# Expected: 200

# Verify it's HTML, not JSON
curl -ksS https://localhost/some/deep/route | head -1
# Expected: <!doctype html> or <!DOCTYPE html>
```

Then open `https://localhost` in the browser. Accept the cert warning. You
should see the product UI (login page), not the study.

**Failure:** 404 on deep links means nginx SPA fallback isn't working. Check
`desktop/nginx.conf` has the `try_files $uri $uri/ /index.html` rule.
If the page is blank, open DevTools console — if you see
`VITE_SUPABASE_URL is not set`, the web image was built without secrets
(see Prerequisites).

### 6. Unauthenticated requests return 401

```bash
for pair in \
  "POST /api/v1/images/upload" \
  "GET /api/v1/images/" \
  "GET /api/v1/images/fake-id" \
  "DELETE /api/v1/images/fake-id" \
  "POST /api/v1/analyses/" \
  "GET /api/v1/analyses/" \
  "GET /api/v1/analyses/fake-id" \
  "DELETE /api/v1/analyses/fake-id" \
  "GET /api/v1/analyses/image/fake-id" \
  "POST /api/detect"; do
  method=$(echo "$pair" | awk '{print $1}')
  path=$(echo "$pair" | awk '{print $2}')
  code=$(curl -ksS -o /dev/null -w '%{http_code}' -X "$method" "https://localhost$path")
  printf "%-7s %-40s %s\n" "$method" "$path" "$code"
done
```

**Expected:** Every line shows `401`.

**Failure:** `500` means the auth dependency is crashing instead of returning
a clean error. `422` on POST endpoints is acceptable (missing body, but auth
was checked first — verify by checking the response body for "Not authenticated"
vs a validation error). `404` means Caddy isn't routing to the backend.

### 7. Sign up, log in, upload end to end

Do this in the browser at `https://localhost`:

1. Sign up with an email/password.
2. Confirm the email (check your inbox or Supabase dashboard).
3. Log in.
4. Upload a JPEG image.
5. Verify it appears in the image list.

**Failure at step 1:** "Failed to fetch" or CORS error in DevTools. See
"What to expect to break" section below — likely a Supabase auth redirect
URL issue.

### 8. Upload storage path is {user_id}/{uuid}.{ext}

After uploading in step 7, check the response in DevTools Network tab, or
query the database:

```bash
# From the Supabase SQL editor or via curl:
# Look at the storage_path column in the images table.
# It should be: <uuid-of-user>/<uuid-of-file>.jpg
```

**Expected:** Path starts with a UUID (the user ID), not `uploads/`.

**Failure:** Path starts with `uploads/` means the old code is still in the
image. Re-check that CI rebuilt after the storage path commit.

### 9. Rate limit returns 429

Set `RATE_LIMIT_ANALYSES=3/minute` in `.env.local`. Restart the backend if
you changed it after boot.

```bash
# Get a token first (sign in via the UI, copy from DevTools or use curl)
TOKEN="<your-access-token>"

for i in 1 2 3 4 5; do
  code=$(curl -ksS -o /dev/null -w '%{http_code}' -X POST \
    -H "Authorization: Bearer $TOKEN" \
    -F "file=@test.jpg" \
    "https://localhost/api/detect")
  echo "Request $i -> $code"
done
```

**Expected:** First 3 return 200 (or 503 if no model), 4th and 5th return 429.

**Failure:** All 5 return the same code. See "What to expect to break"
below — Caddy may be masking client IPs.

### 10. Upload validation

```bash
TOKEN="<your-access-token>"

# Oversized file -> 413
dd if=/dev/zero bs=1M count=200 2>/dev/null | \
  curl -ksS -o /dev/null -w '%{http_code}' -X POST \
    -H "Authorization: Bearer $TOKEN" \
    -F "file=@-;filename=big.jpg;type=image/jpeg" \
    "https://localhost/api/v1/images/upload"
# Expected: 413 (Caddy rejects at 25MB before FastAPI sees it)

# Wrong MIME type -> 400
echo "not an image" > /tmp/fake.txt
curl -ksS -o /dev/null -w '%{http_code}' -X POST \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@/tmp/fake.txt;type=text/plain" \
  "https://localhost/api/v1/images/upload"
# Expected: 400
```

**Failure:** 413 returning as 502 means Caddy closed the connection and the
backend never saw the request (still correct behavior, the upload is rejected).
200 on the oversized file means the size limit isn't working.

### 11. GradCAM overlays render

After running an analysis (upload + detect in the UI), check if GradCAM
heatmap images load. They are served from `/gradcam/<filename>`.

```bash
# If you know a filename from the analysis response:
curl -ksS -o /dev/null -w '%{http_code}' https://localhost/gradcam/<filename>.jpg
# Expected: 200
```

**Failure:** 404 means `GRADCAM_DIR` inside the container doesn't match what
the static mount serves. 502 means Caddy isn't routing `/gradcam/*` to the
backend.

---

## Tear down

```bash
docker compose -f docker-compose.local.yml down -v
```

The `-v` flag removes the named volumes (caddy data, app data). Omit it if
you want to keep state across restarts.

---

## What to expect to break

### Supabase auth redirect URLs

Supabase auth requires the redirect URL to be allowlisted in the dashboard.
Go to Supabase > Authentication > URL Configuration and add:

```
https://localhost
```

to the "Redirect URLs" list. Without this, sign-up confirmation and OAuth
flows will fail with a redirect error. Email/password sign-in may still work
since it doesn't redirect.

### Rate limiting behind Caddy

slowapi's `get_remote_address` reads `request.client.host`. uvicorn's
`--proxy-headers` rewrites this from `X-Forwarded-For`. Caddy sets
`X-Forwarded-For` to the real client IP for external requests.

**However**, locally, all containers are on the same Docker bridge network.
Caddy sees the Docker gateway IP (typically `172.x.0.1`) as the client for
requests from the host. Every browser request will appear to come from the
same IP — which is actually correct behavior (there is only one client
locally). On a real VPS with multiple visitors, each will have a distinct IP.

If you want to verify per-IP isolation works, make requests from inside a
different container on the same network vs from the host, and confirm they
get independent rate limit buckets.

### Self-signed cert and Supabase/API calls

The self-signed cert only affects the browser's connection to `localhost`.
It does **not** affect:

- **Supabase auth/DB calls** from the frontend: these go directly to
  `https://<project>.supabase.co` (a real cert).
- **Supabase calls from the backend**: these also go to the real Supabase
  URL over the internet.
- **Backend-to-backend calls**: the HEALTHCHECK uses `http://localhost:8000`
  inside the container (no TLS involved).

The only thing the self-signed cert breaks is the browser warning on first
visit. Accept it once and it won't appear again for that session.

### Empty Supabase config in the web image

If `VITE_SUPABASE_URL` and `VITE_SUPABASE_ANON_KEY` were not set as GitHub
repo secrets before CI built the web image, the frontend JS bundle has empty
strings for these values. The page will load but:

- Login/signup will fail silently or show "VITE_SUPABASE_URL is not set"
- No API calls will work from the frontend

**Fix:** Set the secrets in GitHub, re-run the deploy workflow, then
`docker pull ghcr.io/viktorahnstrom/assert-real-web:latest`.
