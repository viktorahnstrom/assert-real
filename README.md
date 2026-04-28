# XADE — eXplainable Automated Deepfake Evaluation

> Bachelor's Thesis · Jönköping University  
> Viktor Ahnström & Viktor Carlsson · Supervised by Neziha Akalin

XADE is a desktop deepfake detection framework that pairs an EfficientNet-B4 classifier with LayerCAM heatmaps, BiSeNet face parsing, per-region forensic features, and Vision-Language Model explanations to make AI decisions auditable for non-technical users. The thesis focuses on explainability (XAI), forensic grounding of VLM output, and user preference across explanation sources.

---

## Research Questions

Two versions are kept in this README so the supervisor and writing chapters can converge before the user study runs. The thesis report uses **Current**; the **Proposed** revision realigns RQ3 with the actual within-subjects study design (RQ2 is left intact since the cross-dataset analysis is already written up).

### Current (in thesis writing)

| # | Question |
|---|---------|
| **RQ1** | How effectively can modern VLMs explain deepfake detection results to non-expert users? |
| **RQ2** | How well does the detection model generalize across datasets with different manipulation types? |
| **RQ3** | Which explanation modality (visual, textual, or combined) do users prefer? |

### Proposed (post-grounding-roadmap)

| # | Question |
|---|---------|
| **RQ1** | How effectively can modern VLMs explain deepfake detection results to non-expert users? |
| **RQ2** | How well does the detection model generalize across datasets with different manipulation types? *(unchanged — already covered in the thesis report)* |
| **RQ3** | Among commercial VLM providers (OpenAI, Google, Anthropic) and a deterministic rule-based template, which produces explanations participants prefer for clarity, trustworthiness, and helpfulness? |

> **Why the change?** The current RQ3 implies a modality contrast (visual / textual / combined), but the user study keeps the modality fixed — every participant sees image + heatmap + ELA overlay + region crops + text — and varies the *source* of the text across four arms. The proposed RQ3 matches the within-subjects design implemented in [`backend/app/routers/study.py`](backend/app/routers/study.py) and [`desktop/src/components/auth/DeepfakeTest.tsx`](desktop/src/components/auth/DeepfakeTest.tsx).

---

## Architecture

```
                    ┌──────────────┐
                    │   Desktop    │
                    │  Electron +  │
                    │  React/Vite  │
                    └──────┬───────┘
                           │ image upload
                           ▼
   ┌──────────────────────────────────────────────────────────┐
   │                    Backend (FastAPI)                      │
   │                                                           │
   │     EfficientNet-B4  ──▶  fake / real + confidence        │
   │            │                                              │
   │            ▼                                              │
   │     LayerCAM (multi-layer fusion, face-bbox masked)       │
   │            │                                              │
   │            ▼                                              │
   │     BiSeNet face parsing  (19 classes → 6 UI regions)     │
   │            │                                              │
   │            ▼                                              │
   │     Forensic features per region                          │
   │     Laplacian variance · FFT high-freq · ELA              │
   │     z-scored against a real-face reference distribution   │
   │            │                                              │
   │            ▼                                              │
   │     Region ranker — fuses CAM attention with z-scores     │
   │     → top 1–3 suspicion regions                           │
   │            │                                              │
   │            ▼                                              │
   │     VLM layer (strategy pattern, 4 arms)                  │
   │       OpenAI · Google · Anthropic · rule_based            │
   │     → structured JSON with per-claim evidence tags        │
   │       (evidence_type ∈ visual / metric / heatmap)         │
   │            │                                              │
   │            ▼                                              │
   │     Supabase: Auth · Storage · PostgreSQL (RLS)           │
   └──────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
xade/
├── backend/                 Python FastAPI server (ML + API)
│   ├── app/
│   │   ├── api/             Detection & VLM endpoints
│   │   ├── routers/         Auth, images, analyses, study routes
│   │   ├── services/
│   │   │   ├── forensics/   Laplacian / FFT / ELA + z-score helper
│   │   │   ├── vlm/         Provider strategy + structured-output schema
│   │   │   ├── face_parser  BiSeNet face parsing (pixel masks)
│   │   │   └── …            LayerCAM, region ranker, storage
│   │   ├── dependencies/    Auth middleware (require_auth)
│   │   ├── schemas/         Pydantic request/response models
│   │   └── utils/           Model loader (Hugging Face Hub)
│   ├── models/              Training & evaluation scripts
│   ├── scripts/             Dataset prep, reference-distribution builder
│   ├── tests/               Pytest unit tests (forensics, VLM, ranker)
│   └── checkpoints/         Model weights (auto-downloaded)
├── desktop/                 Electron + React + Vite + TypeScript
│   ├── electron/            Main process & preload
│   └── src/                 React app (components, contexts, lib)
├── shared/                  Shared TypeScript types
└── docs/                    Documentation, study materials, roadmap
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **Detection model** | EfficientNet-B4 (fine-tuned, hosted on [Hugging Face Hub](https://huggingface.co/viktorahnstrom/xade-deepfake-detector)) |
| **XAI — saliency** | LayerCAM with multi-layer fusion on EfficientNet-B4's last two conv stages, face-bbox-masked before normalization (Grad-CAM kept as a config-flagged baseline) |
| **XAI — face parsing** | BiSeNet (CelebAMask-HQ, 19-class) → 6 UI region masks |
| **XAI — forensic features** | Per-region Laplacian variance, FFT high-frequency energy, Error Level Analysis (ELA), all z-scored against a 200-image FFHQ reference distribution |
| **XAI — region ranking** | Fused score: `α · CAM + (1 − α) · max-abs forensic z-score`, configurable threshold |
| **VLM Providers** | OpenAI (`gpt-4o-mini`), Google Gemini (`gemini-2.5-flash`), Anthropic Claude (`claude-haiku-4-5`), plus a deterministic `rule_based` template provider — all behind a strategy interface, all emit the same structured JSON schema with per-claim `evidence_type` / `evidence_ref` tags |
| **Backend** | Python 3.11, FastAPI, PyTorch, Supabase (Auth + Storage + PostgreSQL with RLS) |
| **Desktop** | Electron 40, React 19, Vite 7, TypeScript, Tailwind CSS 4, shadcn/ui |
| **CI/CD** | GitHub Actions (ESLint, Prettier, Ruff, TypeScript type-check, pytest) |
| **Formatting** | Prettier (JS/TS), Ruff (Python) |

---

## Detection Performance

Trained on 100k samples across four progressively added datasets:

| Dataset | Manipulation Type | Accuracy |
|---------|------------------|----------|
| 140k Real-Fake Faces | StyleGAN generation | 98.51% |
| CIPLAB | Photoshop splicing/copy-move | Evaluated |
| FaceForensics++ | Neural face swap | Evaluated |
| Celeb-DF v2 | Advanced face swap | Evaluated |
| <!-- TODO: failed run --> | <!-- e.g. StyleGAN3 / FFHQ-512 --> | <!-- e.g. < baseline; cross-type transfer failed --> |

**Key finding:** Each manipulation type requires explicit training examples. Cross-type transfer is limited even within the same manipulation family — a result the most recent attempt (last row) reinforces.

---

## Setup

### Prerequisites

- **Node.js** ≥ 22 and npm
- **Python** 3.11+
- A [Supabase](https://supabase.com) project (for auth and storage)
- GPU recommended for training; **CPU is sufficient for inference** — every component (LayerCAM, BiSeNet parsing, forensic features) is benchmarked under 2 s per image on CPU

### Backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Copy and fill in environment variables
cp .env.example .env

# Start the server (model auto-downloads from Hugging Face on first run)
uvicorn app.main:app --reload --port 8000
```

### Desktop

```bash
# From the repo root — installs all workspaces
npm install

# Start the Vite dev server
cd desktop
npm run dev

# Or run with Electron
npm run dev:electron
```

### Environment Variables

See [`backend/.env.example`](backend/.env.example) for the full list. At minimum you need:

- `SUPABASE_URL` / `SUPABASE_ANON_KEY` / `SUPABASE_SERVICE_ROLE_KEY`
- At least one VLM key (`OPENAI_API_KEY`, `GOOGLE_GEMINI_API_KEY`, or `ANTHROPIC_API_KEY`) — the `rule_based` provider works with no key and is always available as a fallback or 4th study arm
- Desktop needs `VITE_SUPABASE_URL` and `VITE_SUPABASE_ANON_KEY` in a `.env` file

---

## Development

### Git Workflow

- Branch naming: `feature/`, `fix/`, `docs/`, `refactor/`
- Conventional commits: `feat(desktop): add heatmap panel`
- PRs required with one approval before merging to main
- Rebase preferred over merge for linear history

### Linting & Formatting

```bash
# JavaScript / TypeScript (from root)
npm run lint              # ESLint
npm run format:check      # Prettier

# Python (from backend/)
ruff check .              # Linter
ruff format --check .     # Formatter
pytest tests/             # Unit tests
```

CI runs all of the above automatically on push and PR.

---

## How It Works

1. **Upload** an image through the desktop app.
2. **EfficientNet-B4** classifies the image as real or fake and returns a confidence score.
3. **LayerCAM** generates a multi-layer-fused heatmap of where the detector looked. The face bounding box is masked before normalization so background activations cannot dominate the scale.
4. **BiSeNet face parsing** turns the image into pixel-accurate region masks, mapped down to six UI categories (eyes & pupils, eyebrows, mouth & teeth, skin texture, hairline & ears, facial boundaries).
5. **Per-region forensic features** are computed inside each mask:
   - **Laplacian variance** — sharpness; fake skin is often unnaturally low.
   - **FFT high-frequency energy** — GAN-generated and upsampled images show characteristic deficits or excesses.
   - **Error Level Analysis (ELA)** — JPEG re-compression residuals spike near blending boundaries.
   - Every metric is **z-scored** against a 200-image real-face reference distribution.
6. **Region ranker** fuses the LayerCAM activation per region with the maximum absolute forensic z-score and surfaces the top 1–3 most suspicious regions.
7. **Evidence package** for the VLM: original image, LayerCAM overlay, ELA overlay, per-region zoom crops, and a `[FORENSIC EVIDENCE]` block listing each region's z-scores in plain text.
8. **VLM layer** (one of four interchangeable providers — OpenAI, Google, Anthropic, or the deterministic rule-based template) returns a **strict JSON schema** containing summary, detailed analysis, technical notes, and per-region claims. Each claim carries:
   - `evidence_type` ∈ `visual` / `metric` / `heatmap`
   - `evidence_ref` — the specific cue cited (e.g. `sharpness_z=-3.8`, `GradCAM peak around the mouth`, `left jaw crop`)
   - `confidence` ∈ [0, 1]
9. **Frontend** renders the original / heatmap / ELA tiles, region cards with z-score strips, and per-claim evidence highlights so users can audit each statement against the cited metric, region, or crop.
10. **Supabase** stores every analysis (image, classification, structured explanation, evidence regions) for history.

---

## License

[MIT](LICENSE) © 2026 Viktor Ahnström
