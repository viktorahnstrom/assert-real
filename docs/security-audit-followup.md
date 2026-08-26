# Audit Follow-up: Auth and RLS

## Question 1: Is `require_auth` applied to the write endpoints?

**No. The images and analyses routers have zero auth protection.**

| Endpoint | File | Auth | Uses service role key |
|---|---|---|---|
| `POST /api/v1/images/upload` | images.py:75 | **None** | Yes (line 49: `access_token or SUPABASE_SERVICE_ROLE_KEY`) |
| `DELETE /api/v1/images/{id}` | images.py:232 | **None** | Yes (same helper) |
| `GET /api/v1/images/` | images.py:174 | **None** | Yes |
| `GET /api/v1/images/{id}` | images.py:205 | **None** | Yes |
| `POST /api/v1/analyses/` | analyses.py:402 | **None** | Yes (line 103: `Bearer {SUPABASE_SERVICE_ROLE_KEY}`) |
| `DELETE /api/v1/analyses/{id}` | analyses.py:782 | **None** | Yes |
| `GET /api/v1/analyses/` | analyses.py:745 | **None** | Yes |
| `GET /api/v1/analyses/{id}` | analyses.py:763 | **None** | Yes |
| `POST /api/detect` | detect.py:124 | **Yes** (`Depends(require_auth)`) | No (runs model locally) |
| `GET /api/model-info` | detect.py:172 | **Yes** (`Depends(require_auth)`) | No |
| `POST /api/v1/study/analyze` | study.py | **None** | No (runs model locally) |

This is the open door. Both `images.py` and `analyses.py` use `SUPABASE_SERVICE_ROLE_KEY` in every Supabase call (via `get_db_headers()` / `get_storage_headers()`), which **bypasses RLS entirely**. Anyone who discovers the URL can:

- Upload arbitrary files to Supabase Storage
- Insert/read/delete any user's images and analyses
- Trigger unbounded model inference + VLM calls via `POST /api/v1/analyses/`

The `images.py` helper has a fallback pattern (`access_token or SUPABASE_SERVICE_ROLE_KEY`) suggesting the intent was to forward the user's token, but no `access_token` is ever passed because there's no `Depends(require_auth)` injecting it.

## Question 2: Do RLS policies exist?

**Yes for the product tables. No for `study_results`** (in the main schema file).

From `docs/xade-schema.sql`:

| Table | RLS enabled | Policies |
|---|---|---|
| `profiles` | Yes (line 98) | SELECT own, UPDATE own |
| `images` | Yes (line 99) | SELECT own, INSERT own, DELETE own |
| `analyses` | Yes (line 100) | SELECT own, INSERT own (no UPDATE, no DELETE) |
| `user_preferences` | Yes (line 101) | SELECT own, INSERT own, UPDATE own |
| `api_logs` | Yes (line 102) | SELECT own (no INSERT -- backend presumably uses service role) |

`study_results` is defined separately in `docs/deploy.md` (lines 58-85), not in `xade-schema.sql`. The documented policy is:

```sql
alter table study_results enable row level security;
create policy "anon can insert" on study_results
  for insert with check (true);
```

This allows any holder of the anon key to INSERT (which is the intended design -- anonymous study participants write results). But there is **no SELECT policy for anon**, so the anon key cannot read other participants' results. However:

- There's no documented policy restricting what data the insert contains (no column-level checks).
- If someone adds a SELECT policy later (or if one was added in the Supabase dashboard but not documented), all study data becomes readable.
- The anon key is baked into the JS bundle, so any visitor has it.

**The critical gap is not RLS -- it's that the backend endpoints bypass RLS by using the service role key without auth.** Even with perfect RLS policies, the images/analyses routers go around them.
