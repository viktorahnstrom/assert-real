# Pre-deploy Verification Report

**Date:** 2026-08-25
**Branch:** `main` (20 commits ahead of `origin/main`, not pushed)

---

## Automated Checks

### 1. Test suite (`VLM_DEFAULT_PROVIDER=mock`)

**BLOCKED** — No local Python venv with project dependencies. 8 test files exist
in `backend/tests/`. Must run inside Docker:

```bash
docker run --rm --platform linux/amd64 assert-real-backend:test \
  python -m pytest -v --tb=short
```

### 2. ruff format --check + ruff check

| Check           | Result |
|-----------------|--------|
| `ruff format`   | **PASS** — 67 files already formatted |
| `ruff check`    | **PASS** — all checks passed |

One fix applied before passing: `analyses.py:810` — ruff wanted a multi-line
function signature collapsed to one line (fits within line length). Formatting
only, no logic change. **Not yet committed.**

### 3. TypeScript typecheck

**PASS** — `tsc --noEmit` exited 0.

### 4. ESLint

**PASS (1 pre-existing warning)** — Unused variable `user` at `App.tsx:513`.
Zero errors. This warning predates the Phase 2 changes.

---

## Docker Checks (waiting on local Docker builds)

### 4b. Image builds + sizes

**TODO** — Build both images locally and record sizes:

```bash
# Use --platform linux/amd64 on Apple Silicon — the VPS is amd64.
# CI (ubuntu-latest) is already amd64 natively.
docker build --platform linux/amd64 -t assert-real-backend:test ./backend
docker build --platform linux/amd64 -t assert-real-web:test -f desktop/Dockerfile .
docker images --format '{{.Repository}}:{{.Tag}}\t{{.Size}}' | grep assert-real
```

### 5. Prefetch check (network-disabled)

**TODO** — Run each model loader with `--network none`:

```bash
# Detection model
docker run --rm --platform linux/amd64 --network none assert-real-backend:test \
  python -c "from app.utils.model_loader import load_model_checkpoint; load_model_checkpoint()"

# Face parser (BiSeNet)
docker run --rm --platform linux/amd64 --network none assert-real-backend:test \
  python -c "from facexlib.parsing import init_parsing_model; init_parsing_model(model_name='bisenet', device='cpu')"

# Face category mapper (MediaPipe)
docker run --rm --platform linux/amd64 --network none assert-real-backend:test \
  python -c "from app.services.face_category_mapper import FaceCategoryMapper; m = FaceCategoryMapper(); m.close()"
```

If any reach for the network, the prefetch wrote to a different cache directory
than the one the app reads at runtime.

---

## Full Stack Locally (docker-compose.prod.yml)

### 6. Endpoint smoke tests

**TODO** — Bring up the stack and verify:

```bash
# Use a local domain override
DOMAIN=localhost ACME_EMAIL=test@test.com docker compose -f docker-compose.prod.yml up -d

# Liveness probe (no DB contact)
curl -fsS http://localhost/health

# Readiness check (full dependency report)
curl -fsS http://localhost/api/v1/health

# SPA fallback (should return index.html, not 404)
curl -fsS -o /dev/null -w '%{http_code}' http://localhost/some/deep/route

# Caddy forwarding
curl -fsS http://localhost/api/model-info   # → backend
curl -fsS http://localhost/gradcam/          # → backend static files
```

### 7. Docker HEALTHCHECK

**TODO** — Confirm container health status:

```bash
docker ps --format 'table {{.Names}}\t{{.Status}}'
```

Expect the backend container to show `(healthy)` after ~120s start period.

---

## Auth & Limits (against running stack)

### 8. Unauthenticated requests → 401

**TODO** — Hit every protected endpoint without a Bearer token:

| Endpoint | Method | Expected |
|----------|--------|----------|
| `/api/v1/images/upload` | POST | 401 |
| `/api/v1/images/` | GET | 401 |
| `/api/v1/images/{id}` | GET | 401 |
| `/api/v1/images/{id}` | DELETE | 401 |
| `/api/v1/analyses/` | POST | 401 |
| `/api/v1/analyses/` | GET | 401 |
| `/api/v1/analyses/{id}` | GET | 401 |
| `/api/v1/analyses/{id}` | DELETE | 401 |
| `/api/v1/analyses/image/{id}` | GET | 401 |
| `/api/detect` | POST | 401 |

```bash
for path in \
  "POST /api/v1/images/upload" \
  "GET  /api/v1/images/" \
  "GET  /api/v1/images/fake-id" \
  "DELETE /api/v1/images/fake-id" \
  "POST /api/v1/analyses/" \
  "GET  /api/v1/analyses/" \
  "GET  /api/v1/analyses/fake-id" \
  "DELETE /api/v1/analyses/fake-id" \
  "GET  /api/v1/analyses/image/fake-id" \
  "POST /api/detect"; do
  method=$(echo $path | awk '{print $1}')
  url=$(echo $path | awk '{print $2}')
  code=$(curl -s -o /dev/null -w '%{http_code}' -X "$method" "http://localhost$url")
  echo "$method $url → $code"
done
```

Any 500 instead of 401 is a bug.

### 9. SUPABASE_SERVICE_ROLE_KEY audit

| File | Line | Status |
|------|------|--------|
| `.env.example:14` | `SUPABASE_SERVICE_ROLE_KEY=` | **OK** — documentation placeholder |
| `scripts/show_study_results.py:57` | Reads key from `.env` | **OK** — offline analysis script, not a request handler. Needs admin access to read all study results across users. Never runs in production. |
| `app/db/database.py:22` | Stored on `Settings` dataclass | **DEAD CODE** — `get_postgrest_admin_client()` is defined and exported but has **zero callers** outside `database.py`. No router or service imports it. |

**Verdict:** No request handler touches the service role key. The only live usage
is the offline `show_study_results.py` script.

### 10. Upload storage path format

**FIXED** — Was `uploads/{uuid}.{ext}`, now `{user_id}/{uuid}.{ext}`.
`user_id` comes from the validated JWT (`user.id` via `Depends(require_auth)`).
Extension is derived from the validated MIME type, not client filename.
Committed as `6b01b89`.

All downstream reads (delete, signed URL generation, analysis image download)
read `storage_path` from the DB column and are unaffected by the prefix change.

### 11. Frontend image display method

**PASS — uses `createSignedUrl`** (not `getPublicUrl`).

`desktop/src/lib/api.ts:281`: `supabase.storage.from('images').createSignedUrls(storagePaths, 3600)`

Signed URLs will continue working after storage policies are tightened.
No change needed.

### 12. Rate limit (429)

**TODO** — Exceed the rate limit on `/api/detect`:

```bash
for i in $(seq 1 12); do
  code=$(curl -s -o /dev/null -w '%{http_code}' -X POST \
    -H "Authorization: Bearer $TOKEN" \
    -F "file=@test.jpg" \
    "http://localhost/api/detect")
  echo "Request $i → $code"
done
```

Expect 429 after 10 requests (default `RATE_LIMIT_ANALYSES=10/hour`).

### 13. Upload validation

**TODO** — Test rejection cases:

```bash
# Oversized file (200MB) → expect 413
dd if=/dev/zero bs=1M count=200 | \
  curl -s -o /dev/null -w '%{http_code}' -X POST \
    -H "Authorization: Bearer $TOKEN" \
    -F "file=@-;filename=big.jpg;type=image/jpeg" \
    "http://localhost/api/v1/images/upload"

# Wrong MIME type (.txt as .jpg) → expect 400
echo "not an image" > fake.txt
curl -s -o /dev/null -w '%{http_code}' -X POST \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@fake.txt;type=text/plain" \
  "http://localhost/api/v1/images/upload"
```

---

## Changes Made During Verification

| File | Change | Reason |
|------|--------|--------|
| `backend/app/routers/analyses.py:810` | Collapsed multi-line function signature to one line | `ruff format` required it. Formatting only. Committed as `dd48c7f`. |
| `backend/app/routers/images.py:128` | `uploads/{uuid}` → `{user.id}/{uuid}` | Storage RLS requires user-scoped path prefix. Committed as `6b01b89`. |

---

## Summary

| # | Check | Status |
|---|-------|--------|
| 1 | pytest (mock VLM) | **BLOCKED** — no local venv |
| 2 | ruff format | **PASS** (1 fix applied) |
| 2 | ruff check | **PASS** |
| 3 | tsc --noEmit | **PASS** |
| 4 | eslint | **PASS** (1 pre-existing warning) |
| 4b | Docker image builds | **TODO** |
| 5 | Prefetch network check | **TODO** |
| 6 | Full stack smoke test | **TODO** |
| 7 | HEALTHCHECK status | **TODO** |
| 8 | 401 on unauthed requests | **TODO** |
| 9 | Service role key audit | **PASS** — no request handler uses it |
| 10 | Upload storage path | **FIXED** — `{user_id}/{uuid}.{ext}` (`6b01b89`) |
| 11 | Image display method | **PASS** — uses `createSignedUrls`, not `getPublicUrl` |
| 12 | Rate limit 429 | **TODO** |
| 13 | Upload validation 413/400 | **TODO** |
