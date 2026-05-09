# Deployment

The user study deploys to **Vercel** as a static SPA. Participant results
write **directly to Supabase** with the public anon key — no backend is
required to run the study.

```
┌──────────────────────────────────────────┐         ┌─────────────────────┐
│ Vercel (static SPA)                      │         │ Supabase            │
│  - Vite web build                        │  HTTPS  │  - study_results    │
│  - bundled study-analyses.json + assets  │  ────►  │    table (anon RLS) │
│  - VITE_STUDY_ONLY=true (default)        │         │                     │
└──────────────────────────────────────────┘         └─────────────────────┘
```

The full app (auth + image upload + post-study product trial) needs the
FastAPI backend on Railway / HF Spaces, but those paths are **not** part
of the deployed user study. Backend deployment is covered separately in
this doc as a follow-up for the thesis defence demo.

---

## Vercel — first-time setup

1. Go to [vercel.com](https://vercel.com) → **New Project** → import the
   GitHub repo.
2. Build settings auto-fill from [`vercel.json`](../vercel.json):
   - Build: `npm --prefix desktop run build:web`
   - Output: `desktop/dist`
   - Install: `npm --prefix desktop install`
   - SPA rewrites: every path → `/index.html`
3. **Environment Variables** (Vercel → Settings → Environment Variables —
   inlined at build time, so changes need a redeploy):

   | Key | Required | Source |
   |---|---|---|
   | `VITE_SUPABASE_URL` | yes | Supabase project settings → API |
   | `VITE_SUPABASE_ANON_KEY` | yes | Supabase project settings → API → anon key |
   | `VITE_STUDY_ONLY` | optional, defaults to `true` | set to `false` to ship the full app instead of just the study |

   See [`desktop/.env.example`](../desktop/.env.example) for local-dev
   reference.
4. Deploy. PRs against `main` get preview URLs automatically; production
   deploys from `main`.
5. Smoke-test the production URL: completing the study should land on the
   "You are now done with the user study" screen and the Supabase
   `study_results` table should show one new row per participant.

---

## Supabase — minimum setup for results writes

The study writes participant data straight to Supabase using the anon
key. The `study_results` table needs an RLS policy that allows anonymous
inserts (and **only** inserts — no reads from public).

```sql
create table study_results (
  id uuid primary key default gen_random_uuid(),
  participant_id text not null,
  self_confidence_rating int,
  baseline_accuracy float,
  total_images int,
  correct_count int,
  incorrect_count int,
  classification_records jsonb,
  explanation_answers jsonb,
  retest_answers jsonb,
  trust_rating int,
  willingness_to_use text,
  explanations_helped_in_retest int,
  comments text,
  phase4_time_ms float8,
  total_time_ms float8,
  total_idle_discarded_ms float8,
  completed_at timestamptz,
  saved_at timestamptz default now()
);

alter table study_results enable row level security;

create policy "anon can insert" on study_results
  for insert to anon with check (true);
```

Read access stays restricted — pull data via the service role key from
the Supabase dashboard or a private analysis script.

---

## Redeploy

| Trigger | What happens |
|---|---|
| Push to `main` | Vercel auto-deploys production. |
| Open a PR against `main` | Vercel builds a preview URL. |
| Updated study assets (regenerated `study-analyses.json` or `quiz-heatmaps/`) | Commit them and push. Vercel rebuilds. Vercel can't run `/study/precompute` itself — run it locally against your dev backend, commit the output, push. |

### Re-running `/study/precompute`

```bash
# Local backend with OPENAI_API_KEY in backend/.env
cd backend
uvicorn app.main:app --reload &
curl -X POST http://localhost:8000/api/v1/study/precompute --max-time 1800
```

The endpoint writes
[`desktop/public/study-analyses.json`](../desktop/public/study-analyses.json)
and per-image assets under
[`desktop/public/quiz-heatmaps/`](../desktop/public/quiz-heatmaps/).

---

## Cost

- **Vercel Hobby:** $0. ~40 MB of bundled study assets is well inside
  the limit.
- **Supabase Free tier:** $0. The study_results table will hold a few
  dozen JSON rows — orders of magnitude under the free tier ceiling.

Total deployment cost for the study window: **$0**.

---

## Pulling participant results

Two options:

1. **Supabase dashboard** → Table editor → `study_results` → Export CSV.
2. **Service-role script** (for a private analysis script, never the
   frontend):

   ```python
   import os
   from supabase import create_client
   client = create_client(
       os.environ["SUPABASE_URL"],
       os.environ["SUPABASE_SERVICE_ROLE_KEY"],
   )
   rows = client.table("study_results").select("*").execute()
   ```

---

## Backend deployment (optional, post-study)

The full app (login + image upload + live grounded explanation) needs the
FastAPI backend hosted somewhere. This is **not required for the study
itself** but is needed for the thesis defence demo.

Hosting options:

| Option | Cost | Notes |
|---|---|---|
| **Railway** | ~$15–25/mo always-on | Best UX, no cold start. Stop the service when not actively demoing to save money. |
| **Hugging Face Spaces** (free) | $0 | ~30 s cold start after idle. Workable for a scheduled defence demo if you warm it 5 min beforehand. |
| **HF Spaces persistent CPU** | $9/mo | No cold start. |

When you want the backend live:

1. Set `VITE_STUDY_ONLY=false` in Vercel.
2. Set `VITE_API_BASE_URL` in Vercel to your backend URL.
3. On Railway / HF Spaces, set the same secrets the local
   [`backend/.env`](../backend/.env) has — at minimum `OPENAI_API_KEY`,
   `SUPABASE_URL`, `SUPABASE_KEY` (service role), `CORS_ORIGINS`.
4. The backend Dockerfile and Railway config live in
   [`backend/`](../backend/). Detailed step-by-step is on the
   `feat/deploy-vercel-railway` branch (or wherever it ends up merged) —
   look for `backend/railway.json` and `backend/Dockerfile`.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| Phase 2 shows "Unavailable" tiles | `study-analyses.json` is from before the ranker / ELA wiring. Re-run `/precompute` locally, commit, push. |
| `study_results` table stays empty after a participant finishes | Anon RLS insert policy missing or wrong. Verify the policy from the SQL block above is active. |
| Site loads but "Loading…" hangs forever | Supabase env vars wrong or missing in Vercel. Check `VITE_SUPABASE_URL` and `VITE_SUPABASE_ANON_KEY` are set on the Production environment. |
| Showing the auth page instead of "Thank you" after the study | `VITE_STUDY_ONLY` is set to `false` in Vercel. Either remove it or set to `true`. |
