# Phase 1 Audit Report

## 1. VITE_STUDY_ONLY Confirmation

**Default is effectively `true`.** The check is `import.meta.env.VITE_STUDY_ONLY !== 'false'`, so any value other than the literal string `"false"` (including undefined) means study-only mode.

When `true`: users see only the multi-phase study flow (classify images, view explanations, retest, survey, thank-you screen). No auth, no upload, no product.

When `false`: after completing the study, users proceed to `AuthPage` then the full product (`UploadView`, `ResultView`, `HistoryView`, `StatisticsView`).

**The product path has never run in production.** This deployment will be its first time. The `localStorage` key `xade-test-completed` gates the transition -- that key name is load-bearing for returning users but should be renamed to avoid confusion (cosmetic category).

---

## 2. True Route Map

### Backend endpoints (final resolved paths)

| Router file | Mount prefix (main.py) | Router's own prefix | Final path |
|---|---|---|---|
| `auth.py` | (none) | `/api/v1/auth` | `/api/v1/auth/signup`, `/login`, `/logout`, `/forgot-password`, `/me` |
| `images.py` | (none) | `/api/v1/images` | `/api/v1/images/upload`, `/`, `/{image_id}`, `/delete/{image_id}` |
| `analyses.py` | (none) | `/api/v1/analyses` | `/api/v1/analyses/`, `/{analysis_id}`, `/delete/{analysis_id}`, `/image/{image_id}` |
| `study.py` | (none) | `/api/v1/study` | `/api/v1/study/analyze`, `/results`, `/precompute` |
| `detect.py` | `/api` | (none) | `/api/detect`, `/api/model-info` |
| `vlm.py` | `/api` | (none) | `/api/vlm-providers`, `/api/vlm-usage`, `/api/vlm-health/{provider_id}` |
| main.py direct | - | - | `/`, `/health`, `/api/v1/health` |
| StaticFiles | - | - | `/gradcam/*` (temp dir) |

**No double-prefixing.** The four `routers/` files declare their own full `/api/v1/...` prefix and are mounted without a prefix. The two `api/` files declare no prefix and are mounted at `/api`. It works, but the split convention is confusing and should be unified.

### Frontend calls (desktop/src/lib/api.ts)

Base URL: `VITE_API_BASE_URL` (default `http://localhost:8000`)

| Function | Method | Path |
|---|---|---|
| `detectDeepfake()` | POST | `/api/detect` |
| `uploadImage()` | POST | `/api/v1/images/upload?user_id=...` |
| `createAnalysis()` | POST | `/api/v1/analyses/` |
| `fetchUserAnalyses()` | GET | `/api/v1/analyses/?user_id=...` |
| `fetchUserImages()` | GET | `/api/v1/images/?user_id=...` |
| `deleteAnalysis()` | DELETE | `/api/v1/analyses/{id}` |
| `studyAnalyzeImage()` | POST | `/api/v1/study/analyze` |
| `fetchVLMProviders()` | GET | `/api/vlm-providers` |

### Mismatches

- `uploadImage()` sends `user_id` as a query param, but the backend endpoint accepts it as an optional parameter that is never used ("Will come from auth token later"). Harmless but dead code.
- `saveStudyResults()` writes directly to Supabase (`study_results` table), bypassing the backend `/api/v1/study/results` endpoint entirely.
- `/api/model-info`, `/api/vlm-usage`, `/api/vlm-health/{provider_id}` exist in backend but are never called by the frontend.

---

## 3. Environment Variable Inventory

### Backend variables

| Variable | Where read | Default | In `.env.example` | Notes |
|---|---|---|---|---|
| `SUPABASE_URL` | database.py, auth.py, images.py, analyses.py, auth dep | `""` | Yes | |
| `SUPABASE_ANON_KEY` | same files | `""` | Yes | |
| `SUPABASE_SERVICE_ROLE_KEY` | database.py, images.py, analyses.py | `""` | Yes | |
| `ENVIRONMENT` | database.py | `"development"` | Yes | |
| `CORS_ORIGINS` | main.py:107 | `""` | **No** | Used but undocumented |
| `VLM_DEFAULT_PROVIDER` | vlm/config.py:43 | `"google"` | Yes | |
| `VLM_MAX_REQUESTS_PER_DAY` | vlm/config.py:44 | `500` | Yes (100) | Code default != example |
| `VLM_MAX_MONTHLY_COST_USD` | vlm/config.py:45 | `5.00` | Yes (10.00) | Code default != example |
| `GOOGLE_GEMINI_API_KEY` | vlm/config.py:49 | `None` | Yes | |
| `GOOGLE_GEMINI_MODEL` | vlm/config.py:52 | `"gemini-2.5-flash"` | Yes (`gemini-2.0-flash`) | Mismatch |
| `GOOGLE_GEMINI_TIMEOUT_SECONDS` | vlm/config.py:54 | `30` | Yes as `GOOGLE_GEMINI_TIMEOUT` | **Name mismatch** |
| `OPENAI_API_KEY` | vlm/config.py:58 | `None` | Yes | |
| `OPENAI_MODEL` | vlm/config.py:61 | `"gpt-4o-mini"` | Yes | |
| `OPENAI_TIMEOUT_SECONDS` | vlm/config.py:63 | `30` | Yes as `OPENAI_TIMEOUT` | **Name mismatch** |
| `ANTHROPIC_API_KEY` | vlm/config.py:67 | `None` | Yes | |
| `ANTHROPIC_MODEL` | vlm/config.py:70 | `"claude-haiku-4-5-20251001"` | Yes | |
| `ANTHROPIC_TIMEOUT_SECONDS` | vlm/config.py:72 | `30` | Yes | |
| `CAM_METHOD` | gradcam_service.py:33 | `"layercam"` | **No** | |
| `FACE_PARSER_INFERENCE_SIZE` | face_parser.py:130 | `512` | **No** | |
| `FACE_PARSER_DEVICE` | face_parser.py:131 | `"cpu"` | **No** | |
| `REGION_RANKER_ALPHA` | region_ranker.py:69 | `0.5` | **No** | |
| `REGION_RANKER_THRESHOLD` | region_ranker.py:70 | `0.35` | **No** | |
| `REGION_RANKER_MAX_REGIONS` | region_ranker.py:71 | `3` | **No** | |
| `REGION_RANKER_MIN_REGIONS` | region_ranker.py:72 | `1` | **No** | |

### Documented but not used in code (aspirational)

| Variable | In `.env.example` | Status |
|---|---|---|
| `LOG_LEVEL` | Yes | Never read |
| `MAX_UPLOAD_BYTES` | Yes (25 MB) | Never read; hardcoded as 10 MB in images.py |
| `RATE_LIMIT_ANALYSES` | Yes (10/hour) | Never read |
| `RATE_LIMIT_DEFAULT` | Yes (60/hour) | Never read |
| `ENABLE_STUDY_ROUTER` | Yes (false) | Never read |

### Desktop variables

| Variable | Where read | Default | In `.env.example` |
|---|---|---|---|
| `VITE_SUPABASE_URL` | supabase.ts:3 | (none -- required) | Yes |
| `VITE_SUPABASE_ANON_KEY` | supabase.ts:4 | (none -- required) | Yes |
| `VITE_API_BASE_URL` | api.ts:1 | `"http://localhost:8000"` | Yes |
| `VITE_STUDY_ONLY` | App.tsx:1082, DeepfakeTest.tsx:810 | effectively `true` | Yes (`false`) |

### Dockerfile variables (backend)

`PYTHONUNBUFFERED`, `PYTHONDONTWRITEBYTECODE`, `HF_HOME`, `TORCH_HOME`, `XDG_CACHE_HOME`, `OMP_NUM_THREADS`, `PORT` -- all hardcoded in the Dockerfile. `PORT` is a Railway leftover.

### CI variables (.github/workflows/ci.yml)

`ELECTRON_SKIP_BINARY_DOWNLOAD=1`, `VLM_DEFAULT_PROVIDER=mock`, `VITE_API_BASE_URL=/api`

### docker-compose.prod.yml / Caddyfile

`DOMAIN`, `ACME_EMAIL`, `IMAGE_TAG` -- all documented in the root `.env.example`.

---

## 4. VLM Cost Controls

**Enforced, but in-memory only.** `usage_tracker.py` checks daily request count and monthly cost ceiling before every VLM call. If exceeded, it returns a graceful fallback explanation instead of an error. However:

- Counters reset on every server restart (no persistence to DB or Redis).
- The daily reset logic tracks only `last_reset_day` (day-of-month), not the full date. If the server runs for 31+ days without restart, it will silently skip a reset at the month boundary. Not a real risk with single-worker restarts, but worth noting.

---

## 5. Rate Limiting

**None.** No `slowapi`, no middleware, no decorators. The env vars `RATE_LIMIT_ANALYSES` and `RATE_LIMIT_DEFAULT` are documented in `.env.example` but never read. Any IP can hit `/api/detect` or `/api/v1/study/analyze` as fast as it wants. The only protection is the VLM cost cap, which doesn't prevent CPU-bound abuse (detection model inference).

---

## 6. Upload Validation

**Partial, with a critical ordering bug.**

In `images.py:75-110`:
- MIME type is checked against a whitelist (`image/jpeg`, `image/png`, `image/webp`) -- good.
- File size is checked against 10 MB -- but only **after** `await file.read()` buffers the entire file into memory (line 94). A 1 GB file with a spoofed `image/jpeg` content-type will be fully loaded before rejection.
- File extension is extracted from the client-supplied filename with no validation (line 104). Attacker could name a file `payload.php`.
- The `MAX_UPLOAD_BYTES` env var (25 MB) in `.env.example` is never used; the 10 MB limit is hardcoded.

---

## 7. GradCAM Storage

**Temp dir, no cleanup, unbounded growth.**

- Storage path: `tempfile.gettempdir() / "xade_gradcam"` (gradcam_storage.py:14, main.py:121)
- Each analysis writes ~500 KB-1 MB (GradCAM overlay + ELA overlay + 3 region crops).
- No TTL, no cleanup cron, no size cap. On a Docker container, `/tmp` is ephemeral but grows until the container restarts or the disk fills.
- Served via FastAPI `StaticFiles` at `/gradcam`.

---

## 8. Model Loading

`model_loader.py` downloads from `viktorahnstrom/xade-deepfake-detector` on HuggingFace Hub into `backend/checkpoints/best_model.pt`. It checks if the file exists first; if not, it downloads (blocking, no timeout). BiSeNet (face parser) and MediaPipe (face mesh) also fetch weights on first use.

In Docker, the checkpoints directory is inside the image layer only if baked in at build time. Currently it is not -- every container restart triggers a fresh download.

---

## 9. Test Coverage

**8 test files in `backend/tests/`:**

| File | What it covers |
|---|---|
| `test_attach_region_comments.py` | Region comment attachment logic |
| `test_categories.py` | Face category dataclass invariants |
| `test_face_parser.py` | BiSeNet smoke test (requires model download) |
| `test_forensics.py` | ELA, sharpness, spectrum, forensic report |
| `test_prompt_builder.py` | VLM prompt construction and parsing |
| `test_region_ranker.py` | Region ranking fusion logic |
| `test_structured_schema.py` | VLM structured output parsing |
| `test_zscore.py` | Z-score computation |

**Not covered:** No tests for any router/endpoint, no integration tests, no auth tests, no upload tests.

**No pytest config** in `pyproject.toml` or `pytest.ini`. CI workflow references pytest but does not currently run it.

---

## 10. Auth in Production

- **No JWT audience/issuer validation.** The backend's `require_auth` dependency (dependencies/auth.py) forwards the token to Supabase's `/auth/v1/user` endpoint and trusts whatever comes back. It doesn't validate claims locally.
- **No cookie configuration.** Session handling is entirely Supabase client-side (localStorage-based `access_token`). No `HttpOnly` cookie is set by the backend.
- **CORS regex is too broad.** `r"https://xade.*\.vercel\.app"` matches `xade-anything-at-all.vercel.app` because `.*` is greedy. An attacker could create `xade-evil.vercel.app` and it would pass CORS.
- **Supabase redirect URLs** are configured in the Supabase dashboard, not in code. You'll need to add the production domain there for OAuth callbacks.
- **No hardcoded secrets in code.** All keys come from env vars.

---

## 11. XADE/xade References

### (a) Cosmetic -- safe to rename to assert-real

- `main.py:1`: docstring `"XADE Backend API"`
- `main.py:24`, `main.py:85`: emoji print statements (`Starting XADE Backend...`, `Shutting down XADE backend...`)
- `main.py:91`: FastAPI `title="XADE Backend API"`
- `main.py:137`: root endpoint response `"name": "XADE Backend API"`
- `api.ts:131`: error message `"Cannot reach the XADE backend"`
- `api.ts`: console log prefixes `[XADE]`, `[XADE study]`
- Module docstrings in auth.py, images.py, model_loader.py, schemas/models.py, services/__init__.py, gradcam_storage.py, gradcam_service.py, dependencies/auth.py, utils/__init__.py
- `README.md:1`: `# XADE`
- `package.json:2`: `"name": "xade"` -- cosmetic (private package, not published), but update it
- `package.json:3`: `"description": "eXplainable Automated Deepfake Evaluation"`
- `localStorage` key: `xade-test-completed` (App.tsx, DeepfakeTest.tsx)
- Various docs files (deploy.md, documentation-checklist.md, xade-schema.sql header, technology-stack.md, user-study-deployment-roadmap.md)

### (b) Load-bearing -- do NOT rename

- `model_loader.py:17`: `HF_REPO_ID = "viktorahnstrom/xade-deepfake-detector"` -- this is the HuggingFace repo ID. Renaming breaks model downloads unless you also rename/fork the HF repo.
- `main.py:121` and `gradcam_storage.py:14`: temp dir path `xade_gradcam` -- the name doesn't matter functionally (it's a temp dir), but both files must use the same name. Can be renamed as a pair.
- Supabase table names (`profiles`, `images`, `analyses`, `study_results`, `user_preferences`, `api_logs`) are NOT prefixed with "xade" -- safe, no action needed.

---

## 12. Railway and Vercel Leftovers

| Location | What | Status |
|---|---|---|
| `backend/Dockerfile:48` | `ENV PORT=8000` | Railway convention. Uvicorn should bind to a fixed port, not `$PORT`. |
| `vercel.json` | Vercel build config | Still needed if you keep the Vercel study deployment. Not needed for self-hosted. |
| `main.py:114` | `allow_origin_regex=r"https://xade.*\.vercel\.app"` | Hardcoded Vercel regex. Replace with env-driven CORS. |
| `main.py:105-106` | Comment referencing Railway/Vercel URLs | Dead comment. |
| `desktop/.env.example:16` | Comment: `https://xade-backend.up.railway.app` | Dead Railway URL in comment. |

---

## 13. Corrections to the Phase 2 Plan

A few things I'd adjust based on what I found:

1. **Router prefix unification (Phase 2, step 1):** The current routing actually works without double-prefixing. The `routers/` files use `/api/v1/...` prefixes internally; the `api/` files use no prefix and get `/api` from the mount. If you want everything under `/api`, the simplest change is to move `detect.py` and `vlm.py` to use `/api/v1/...` prefixes like the others, and drop the mount prefix. The frontend calls `/api/detect` (no `/v1/`), so you'd need to update those too.

2. **Health check split (step 2):** Current `/health` and `/api/v1/health` both hit the same handler that queries the database. Your plan to split liveness from readiness is correct. Just note that the DB query is `profiles.select("id").limit(1)` -- if the `profiles` table is empty, this still succeeds (returns empty list), so it's a valid connectivity check.

3. **GradCAM dir naming (step 3):** The temp dir is currently named `xade_gradcam`. When you make it configurable via `GRADCAM_DIR`, that implicitly handles the rename.

4. **Upload validation (step 5):** Beyond rate limiting, fix the read-before-check ordering in `images.py`. Use FastAPI's `UploadFile` with a streaming size check, or set `max_request_size` at the ASGI level.

5. **CORS env var (step 7):** `CORS_ORIGINS` already exists as an env var but is undocumented. You just need to remove the hardcoded regex and document the env var.
