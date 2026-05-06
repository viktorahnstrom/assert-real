# User Study & Deployment Roadmap — GitHub Issues

This document contains six ready-to-paste GitHub issues that take XADE from
"the grounded pipeline is shipped" to "the user study is live on the web and
participants can finish their session by trying the product themselves."

It is the operational successor to
[vlm-grounding-roadmap.md](vlm-grounding-roadmap.md): that roadmap built the
forensic-grounding pipeline; this one ships it to participants.

## One-line vision

Deploy the desktop app as a web build to Vercel and the FastAPI backend to a
small CPU host so participants can run the user study from a URL and, when
they finish, opt into trying the live product without installing anything.

## Pipeline after these changes

```
Vercel (frontend, web build)                 Railway / HF Spaces (backend)
 ├── /                static study landing    ├── /api/v1/analyses/   live
 ├── /study           Phase 1 + Phase 2       ├── /api/v1/study/results
 │   └─ loads precomputed                     ├── /api/v1/auth/*       Supabase
 │      study-analyses.json                   └── EfficientNet-B4 + LayerCAM +
 │      + quiz-heatmaps/                          BiSeNet + forensics + VLM
 └── /app             post-study product
     └─ uploads → backend → grounded
        explanation
```

## Scope principles

- **No installer for the study.** Strangers from a uni mailing list will not
  download an unsigned executable. A URL that opens in their browser
  is the only realistic recruitment channel.
- **Same React code, two build targets.** Vite already supports both web and
  Electron builds. Keep the Electron installer as a bonus track for users
  who want a "real app" experience after the study.
- **Cheapest credible deploy.** Vercel hobby tier is free; backend should
  cost ≤ $10/month for the study window and shut down when grading is done.
  Hugging Face Spaces is the $0 fallback if cost matters more than UX.
- **Backend stays optional during the study itself.** Phase 1 and Phase 2
  are static — `study-analyses.json` and `quiz-heatmaps/` are bundled with
  the frontend build. The backend only matters for `/study/results`
  submission and for the post-study product trial.

## How to use this file

Each issue is delimited by `===` rulers. For every issue:

- **Title** — copy into the GitHub "Title" field.
- **Body** — copy everything between `--- BODY START ---` and `--- BODY END ---`
  into the GitHub "Description" field.
- **Labels** — suggested labels to apply.
- **Estimate** — rough working-day estimate.

---
===============================================================================
---

## Issue 1

**Title:** Smoke-test the full grounded pipeline on the 12 study images

**Labels:** `enhancement`, `backend`, `qa`, `user-study`
**Estimate:** 0.5 day
**Depends on:** —

--- BODY START ---
## Context
All four roadmap issues from
[vlm-grounding-roadmap.md](../docs/vlm-grounding-roadmap.md) (#84, #85, #86,
#88) are merged. Before any user-study activity we need to verify end-to-end
that the 12 curated images under
[study_images/selected/](../study_images/selected/) actually produce valid
grounded explanations across all four arms — not silently falling back to the
legacy free-text parser in
[prompt_builder.py](../backend/app/services/vlm/prompt_builder.py).

## Approach
- Run [`/study/precompute`](../backend/app/routers/study.py) over both
  `study_images/selected/real/` and `study_images/selected/fake/` for all
  four providers (`openai`, `google`, `anthropic`, `rule_based`).
- Cache the precompute output so we don't pay 4× provider cost twice.
- Save the rendered ELA overlays + forensic z-score strips per image to
  `docs/study_smoke_test/` as PNGs for visual review.
- Hand-spot-check 4 explanations (1 per provider): are the cited z-scores
  actually present in the `[FORENSIC EVIDENCE]` block? Do
  `evidence_type=heatmap` claims point to real GradCAM/ELA peaks? Log any
  that read as confabulated.

## Acceptance criteria
- [ ] `desktop/public/study-analyses.json` regenerated and committed for the
      final 12-image set.
- [ ] For each of the 48 (12 × 4) generated explanations, assert
      `vlm_explanation.structured_regions is not None` and every entry has a
      non-empty `evidence_ref`.
- [ ] At least one rendered ELA overlay + forensic strip per image saved to
      `docs/study_smoke_test/` for visual review.
- [ ] Hand-spot-check log committed to `docs/study_smoke_test/spotcheck.md`
      noting any obviously confabulated claims and which provider produced
      them.
- [ ] Confirm Phase 1 classification accuracy distribution is not at
      ceiling/floor on a 2-person internal sanity check (we want some
      misclassified images so Phase 2 actually fires).

## Notes
Use the precompute endpoint, not single `/analyze` calls — we want the same
JSON the participants will see. If a provider's structured-output path falls
back to the legacy free-text parser more than once across the 12 images,
that's a real bug; file it as a follow-up rather than accepting silently.
--- BODY END ---

---
===============================================================================
---

## Issue 2

**Title:** Finalize user-study design — recruitment, consent, preference dimensions

**Labels:** `user-study`, `documentation`, `thesis`
**Estimate:** 0.5–1 day (mostly writing)
**Depends on:** —

--- BODY START ---
## Context
The 4-arm within-subjects flow (#11 / merged via #88) is implemented in
[study.py](../backend/app/routers/study.py) and
[DeepfakeTest.tsx](../desktop/src/components/auth/DeepfakeTest.tsx) but the
surrounding study scaffolding — target N, recruitment channel, consent +
debrief copy, what we ask participants — still needs to be locked before
piloting. Locking the wording up front prevents the `/study/results` JSONL
schema from changing mid-study.

## Approach
- Document target sample size and justification (rule of thumb for
  within-subjects 4-arm preference: 20–30 participants).
- Write the recruitment plan into a new `docs/user_study_plan.md` covering
  channel (uni mailing list / social / lab), incentive (if any), and
  inclusion criteria.
- Draft consent + debrief screens in
  [DeepfakeTest.tsx](../desktop/src/components/auth/DeepfakeTest.tsx) (or
  small new sub-components). Cover: data collected, anonymity, right to
  withdraw, contact email.
- Lock the Phase 2 preference dimensions — current likely set: *clarity*,
  *trustworthiness*, *helpfulness*, plus a free-form "why this one?" text
  field. Decide if optional demographics (age band, AI/CS background y/n)
  go before Phase 1.
- IRB / ethics check with Neziha — does Jönköping need anything formal for
  an anonymous within-subjects preference study with no personal data
  beyond response logs?

## Acceptance criteria
- [ ] `docs/user_study_plan.md` exists and covers: target N, recruitment
      channel, incentive, inclusion criteria, IRB status.
- [ ] Consent screen drafted in
      [DeepfakeTest.tsx](../desktop/src/components/auth/DeepfakeTest.tsx)
      (or sub-component) and gated before Phase 1.
- [ ] Debrief screen drafted and shown after Phase 2 completion.
- [ ] Phase 2 preference dimensions and free-form question wording locked;
      reflected in the `/study/results` JSONL schema in
      [study.py](../backend/app/routers/study.py).
- [ ] Optional demographics decision documented (collect / skip).
- [ ] IRB / ethics confirmation from Neziha recorded in the plan doc.

## Notes
Don't wait for Issue 1 — these can run in parallel. Locking copy and schema
before the pilot saves a re-run if wording has to change.
--- BODY END ---

---
===============================================================================
---

## Issue 3

**Title:** Pilot the user study with 2–3 internal testers

**Labels:** `user-study`, `qa`
**Estimate:** 0.5 day
**Depends on:** Issues 1, 2, 4

--- BODY START ---
## Context
Roadmap #11 already requires a pilot to confirm the 4-card layout is
readable. Bundle that with a wider pilot pass that also stress-tests the
new structured-evidence UI (#86), the consent / debrief flow from Issue 2,
and the live Vercel deploy from Issue 4. The goal at the end of this issue
is "ready to recruit."

## Approach
- Send the live Vercel URL to 2–3 internal testers (lab / supervisor /
  friends with no project context).
- Watch the first run over their shoulder where possible — capture
  completion time, fatigue points, any explanation that reads as obviously
  confabulated, any UI bug.
- Confirm the 4-card layout fits at 1440×900 without horizontal scroll
  (already in #11 AC, re-verify on the deployed build).
- Confirm evidence-tag hover highlights work on the metric strips, ELA
  tile, and crops.
- Verify `/study/results` POST captures the A/B/C/D → provider mapping per
  image so per-condition preference is recoverable in analysis.
- Optionally have one tester continue past Phase 2 into the post-study
  product trial (account creation + image upload) to verify the live
  backend round-trip.

## Acceptance criteria
- [ ] 2–3 testers complete the full Phase 1 + Phase 2 flow on the deployed
      Vercel URL (not localhost).
- [ ] 4-card layout fits at 1440×900 without horizontal scroll.
- [ ] Evidence-tag hover highlights verified on metric strips, ELA tile,
      and crops.
- [ ] `/study/results` JSONL records provider-slot mapping per image.
- [ ] At least one tester completes the post-study product trial end-to-end
      (upload → grounded explanation render).
- [ ] Decision recorded in `docs/user_study_plan.md`: keep all 4 arms or
      drop one based on pilot fatigue feedback.
- [ ] Any UI / pipeline bugs surfaced are fixed inside this issue, not
      spawned out — the goal is "ready to recruit."

## Notes
If the pilot surfaces bugs, fix them in this issue rather than spawning new
ones. The exception is anything that requires re-running the smoke test or
changing the schema — those go back to Issue 1 / Issue 2 respectively.
--- BODY END ---

---
===============================================================================
---

## Issue 4

**Title:** Web build of the desktop app deployed to Vercel with study assets bundled

**Labels:** `user-study`, `desktop`, `infra`, `deploy`
**Estimate:** 1 day
**Depends on:** Issues 1, 6

--- BODY START ---
## Context
The desktop app currently ships only as an Electron build. Strangers
recruited from a mailing list will not install an unsigned executable —
recruitment success requires a URL. Vite already supports both web and
Electron targets from the same React code; this issue produces the web
build, deploys it to Vercel, and bundles the precomputed study assets
(`study-analyses.json`, `quiz-heatmaps/`) so Phase 1 + Phase 2 work without
a live backend.

The Electron installer stays as a bonus track for users who want a "real
app" experience.

## Approach
- Add a Vite web-build script in [desktop/package.json](../desktop/package.json)
  alongside the existing Electron build. The web build excludes anything in
  `desktop/electron/` and any code that imports `electron` /
  `ipcRenderer` — that decoupling is Issue 6, which this issue depends on.
- Read the backend URL from `import.meta.env.VITE_API_BASE_URL` so the same
  build can target localhost (dev), Vercel preview (PR), and production
  (Vercel prod).
- Commit a minimal [`vercel.json`](../vercel.json) at the repo root with
  the Vite build command, output dir (`desktop/dist`), and SPA rewrites
  so client-side routing works.
- Confirm `desktop/public/study-analyses.json` and
  `desktop/public/quiz-heatmaps/` produced by Issue 1 are committed and
  end up in the deployed bundle.
- Connect Vercel to the GitHub repo so PRs get preview URLs automatically.
- Set the production environment variables in Vercel: `VITE_SUPABASE_URL`,
  `VITE_SUPABASE_ANON_KEY`, `VITE_API_BASE_URL` (pointing at the Issue 5
  backend).
- Document the deploy + redeploy flow (incl. how to rebuild
  `study-analyses.json` and trigger a redeploy) in
  `docs/user_study_plan.md`.

## Acceptance criteria
- [ ] `npm --prefix desktop run build:web` produces a static `dist/` that
      runs in any modern browser without Electron present.
- [ ] [`vercel.json`](../vercel.json) committed at the repo root.
- [ ] Production deploy succeeds on Vercel; URL recorded in
      `docs/user_study_plan.md`.
- [ ] Phase 1 (12-image quiz) and Phase 2 (4-arm explanation comparison)
      both work end-to-end on the deployed URL with no backend reachable.
- [ ] Backend URL is read from `VITE_API_BASE_URL`; no hardcoded
      `localhost:8000` in the bundle.
- [ ] `/study/results` POST succeeds against the Issue 5 backend; failed
      submissions show a clear error to the user.
- [ ] PRs against `main` get a Vercel preview URL automatically.

## Notes
This is the "deploy" for the user study. The Electron installer is no
longer the primary delivery — keep it building so the post-thesis
"installable demo" story still works, but recruitment uses the URL.
--- BODY END ---

---
===============================================================================
---

## Issue 5

**Title:** Cloud-host the FastAPI backend on Railway (or Hugging Face Spaces) for the post-study product trial

**Labels:** `user-study`, `infra`, `backend`, `deploy`
**Estimate:** 1 day
**Depends on:** —

--- BODY START ---
## Context
Participants who finish the user study can opt to create an account and try
the live product on their own image. That requires a reachable FastAPI
backend running the full grounded pipeline (EfficientNet-B4 → LayerCAM →
BiSeNet → forensics → VLM). The study itself uses bundled static JSON
(Issue 4), so the backend is **not** required during Phase 1 + Phase 2 —
but it **is** required for `/study/results` POST and for the post-study
trial. Goal is the cheapest CPU-only deploy that runs the full pipeline in
under ~6 s round-trip per image.

## Approach
Evaluate two hosts and pick one:

| Option | Pricing (approx) | Notes |
|--------|------------------|-------|
| **Railway** | ~$5–10/mo for shared-CPU 1× / 2 GB | Docker-native, no auto-sleep, good logs, GitHub auto-deploy |
| **Hugging Face Spaces** (fallback) | $0 (CPU upgrade $9/mo) | Free CPU, no sleep, model already on HF Hub. Cold start ~30 s after idle, no custom domain on free tier |

- Add a `backend/Dockerfile` that produces a CPU-only image (`torch+cpu`,
  no CUDA) under ~3 GB. Lazy-download the EfficientNet-B4 weights from
  Hugging Face Hub on first request (existing behaviour) and the BiSeNet
  weights via the existing parser bootstrap.
- Move secrets (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`,
  `GOOGLE_GEMINI_API_KEY`, Supabase service role key) to the host's secret
  manager. Never commit.
- Configure CORS in [app/main.py](../backend/app/main.py) to allow the
  Vercel production URL + preview URLs.
- Document deploy + redeploy steps in `docs/deploy.md`.

## Acceptance criteria
- [ ] [`backend/Dockerfile`](../backend/Dockerfile) builds a CPU-only image
      under ~3 GB and runs `uvicorn app.main:app --host 0.0.0.0 --port $PORT`.
- [ ] Production deploy succeeds; URL recorded in `docs/user_study_plan.md`.
- [ ] `/api/v1/health` returns 200 within 2 s on a warm container.
- [ ] One end-to-end `/api/v1/analyses/` call against a real image succeeds
      and returns a structured JSON explanation in < 6 s warm.
- [ ] CORS allows the Vercel production + preview URLs.
- [ ] Provider API keys + Supabase service role key configured via the
      host's secret manager, not committed.
- [ ] `docs/deploy.md` documents deploy, redeploy, and secret-rotation
      steps.

## Notes
If Railway's monthly cost is annoying after the study, swap to Hugging Face
Spaces and live with the ~30 s cold start — for a thesis defence demo it's
acceptable. If the cold start hurts during the study window, bake the
EfficientNet-B4 + BiSeNet weights into the Docker image at build time
instead of lazy-downloading on first request.
--- BODY END ---

---
===============================================================================
---

## Issue 6

**Title:** Decouple the Vite/React app from Electron so it builds for the web

**Labels:** `desktop`, `refactor`, `infra`
**Estimate:** 0.5–1 day
**Depends on:** —

--- BODY START ---
## Context
The current desktop app couples its Vite/React code to Electron-specific
APIs (e.g. `ipcRenderer`, `window.electronAPI`, native dialogs, anything
under [desktop/electron/](../desktop/electron/)). Issue 4 needs a clean web
build that runs in a normal browser — which means every Electron-only
import must be either removed, behind a runtime check, or stubbed for the
web target. This issue does that decoupling. It is mechanical refactoring;
no new product behaviour.

## Approach
- Grep [desktop/src/](../desktop/src/) for `ipcRenderer`, `electron`,
  `window.electronAPI`, `process.versions`, `node:` imports, and any other
  Electron-only symbols. Catalogue every call site.
- For each call site, choose one:
  - **Browser-equivalent**: replace with the standard web API (e.g.
    `<input type="file">` instead of native open dialog).
  - **Runtime-guarded**: keep both paths behind
    `if (window.electronAPI) { … } else { … }`.
  - **HTTP**: route through the FastAPI backend instead of IPC.
- Move the API base URL out of any hardcoded constant and into
  `import.meta.env.VITE_API_BASE_URL`, defaulting to `http://localhost:8000`
  for dev.
- Add a `npm --prefix desktop run build:web` script that runs `vite build`
  with the Electron entry excluded.
- Verify the web build with `npm --prefix desktop run preview` — the full
  app should load and the study flow should work against bundled JSON.

## Acceptance criteria
- [ ] No file under `desktop/src/` imports `electron`, `ipcRenderer`,
      `node:*`, or accesses `window.electronAPI` without a runtime guard.
- [ ] `import.meta.env.VITE_API_BASE_URL` is the single source of truth for
      the backend URL across the frontend.
- [ ] `npm --prefix desktop run build:web` produces a `dist/` that loads
      cleanly in `npm run preview` (Chromium / Firefox / Safari spot
      check).
- [ ] Existing Electron build (`npm --prefix desktop run build:electron`,
      or whatever the current script is named) still produces a working
      installer — this issue does not break the Electron path.
- [ ] Phase 1 + Phase 2 of the user study run end-to-end in the web
      preview against the bundled `study-analyses.json`.

## Notes
This is purely mechanical and unblocks Issue 4. If a call site genuinely
needs an Electron-only capability with no browser equivalent (e.g. true
native filesystem access for batch import), it's fine to leave it
electron-only — just runtime-guard it so the web build doesn't crash on
import.
--- BODY END ---

---
===============================================================================
---

## Dependency graph

```
Issue 1 (smoke test)         ──┐
                               ├──► Issue 4 (Vercel web deploy) ──┐
Issue 6 (decouple Electron)  ──┘                                  │
                                                                  ├──► Issue 3 (pilot)
Issue 2 (study design)        ────────────────────────────────────┤
                                                                  │
Issue 5 (backend host)        ────────────────────────────────────┘
```

Issues 1, 2, 5, and 6 are independent — start them in parallel.

## Suggested 1-week sprint order

| Day  | Issue                                              | Output                                   |
|------|----------------------------------------------------|---------------------------------------   |
| 1    | #1 Smoke-test                                      | Validated `study-analyses.json` for 12×4 |
| 1    | #2 Study design                                    | `docs/user_study_plan.md`                |
| 1    | #6 Decouple Electron                               | `npm run build:web` works                |
| 1    | #5 Backend cloud host                              | Live `/api/v1/health`                    |
| 2    | #4 Vercel web deploy                               | Public URL with bundled study assets     |
| 3    | #3 Pilot                                           | "Ready to recruit"                       |
| 4–7  | Recruit + run                                      | First batch of `/study/results` JSONL    |

## Optional: create all issues via `gh` CLI

```bash
gh issue create \
  --title "Smoke-test the full grounded pipeline on the 12 study images" \
  --label "enhancement,backend,qa,user-study" \
  --body-file - <<'EOF'
... paste body here ...
EOF
```

The issues above are written as plain Markdown so they render identically in
GitHub's issue view.
