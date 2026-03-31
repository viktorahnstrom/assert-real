# XADE — eXplainable Automated Deepfake Evaluation

> Bachelor's Thesis · Jönköping University  
> Viktor Ahnström & Viktor Carlsson · Supervised by Neziha Akalin

XADE is a cross-platform deepfake detection framework that pairs an EfficientNet-B4 classifier with Grad-CAM heatmaps and Vision-Language Model explanations to make AI decisions understandable for non-technical users. The thesis focuses on explainability (XAI), cross-dataset generalization, and user preference across explanation modalities.

---

## Research Questions

| # | Question |
|---|---------|
| **RQ1** | How effectively can modern VLMs explain deepfake detection results to non-expert users? |
| **RQ2** | How well does the detection model generalize across datasets with different manipulation types? |
| **RQ3** | Which explanation modality (visual, textual, or combined) do users prefer? |

---

## Architecture

```
┌──────────────┐     ┌──────────────────────────────────────────────────┐
│   Desktop    │     │                  Backend (FastAPI)                │
│  Electron +  │────▶│                                                  │
│  React/Vite  │     │  ┌────────────┐  ┌──────────┐  ┌────────────┐  │
└──────────────┘     │  │ EfficientNet│  │ Grad-CAM │  │ VLM Layer  │  │
                     │  │    B4       │──│ Heatmaps │──│ (Strategy) │  │
┌──────────────┐     │  └────────────┘  └──────────┘  └────────────┘  │
│    Mobile    │     │        │                              │         │
│ React Native │────▶│        ▼                              ▼         │
│    (Expo)    │     │   Detection          Explanations (3-tier)      │
└──────────────┘     │   Result             Summary / Detailed /       │
                     │                      Technical Notes            │
                     │                                                  │
                     │           ┌──────────────────┐                  │
                     │           │    Supabase       │                  │
                     │           │  Auth · Storage   │                  │
                     │           │  PostgreSQL (RLS) │                  │
                     │           └──────────────────┘                  │
                     └──────────────────────────────────────────────────┘
```

---

## Project Structure

```
xade/
├── backend/               Python FastAPI server (ML + API)
│   ├── app/
│   │   ├── api/           Detection & VLM endpoints
│   │   ├── routers/       Auth, images, analyses routes
│   │   ├── services/      Grad-CAM, VLM providers, usage tracking
│   │   ├── dependencies/  Auth middleware (require_auth)
│   │   ├── schemas/       Pydantic request/response models
│   │   └── utils/         Model loader (Hugging Face Hub)
│   ├── models/            Training & evaluation scripts
│   ├── scripts/           Dataset preparation utilities
│   └── checkpoints/       Model weights (auto-downloaded)
├── desktop/               Electron + React + Vite + TypeScript
│   ├── electron/          Main process & preload
│   └── src/               React app (components, contexts, lib)
├── mobile/                React Native (Expo)
├── shared/                Shared TypeScript types
└── docs/                  Documentation & user study materials
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **Detection Model** | EfficientNet-B4 (fine-tuned, hosted on [Hugging Face Hub](https://huggingface.co/viktorahnstrom/xade-deepfake-detector)) |
| **XAI** | Grad-CAM heatmaps + VLM natural-language explanations |
| **VLM Providers** | Google Gemini 2.0 Flash, OpenAI GPT-4o-mini (strategy pattern, swappable) |
| **Backend** | Python 3.11, FastAPI, PyTorch, Supabase (Auth + Storage + PostgreSQL) |
| **Desktop** | Electron 40, React 19, Vite 7, TypeScript, Tailwind CSS 4, shadcn/ui |
| **Mobile** | React Native, Expo SDK 54 |
| **CI/CD** | GitHub Actions (ESLint, Prettier, Ruff, TypeScript type-check) |
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

**Key finding:** Each manipulation type requires explicit training examples. Cross-type transfer is limited even within the same manipulation family.

---

## Setup

### Prerequisites

- **Node.js** ≥ 22 and npm
- **Python** 3.11+
- A [Supabase](https://supabase.com) project (for auth and storage)
- GPU recommended for inference (CUDA 12.1 supported); CPU works too

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

### Mobile

```bash
cd mobile
npm install
npx expo start
```

### Environment Variables

See [`backend/.env.example`](backend/.env.example) for the full list. At minimum you need:

- `SUPABASE_URL` / `SUPABASE_ANON_KEY` / `SUPABASE_SERVICE_ROLE_KEY`
- At least one VLM key (`GOOGLE_GEMINI_API_KEY` or `OPENAI_API_KEY`) for explanations
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
```

CI runs all of the above automatically on push and PR.

---

## How It Works

1. **Upload** an image through the desktop or mobile app
2. **EfficientNet-B4** classifies the image as real or fake with a confidence score
3. **Grad-CAM** generates a heatmap highlighting the facial regions the model focused on
4. **Evidence regions** are cropped from the highest-activation areas and labeled by facial location
5. A **VLM** (Gemini or GPT-4o-mini) receives the original image, heatmap, and detection results, then produces a three-tier explanation:
   - **Summary** — one-sentence verdict for quick scanning
   - **Detailed Analysis** — plain-language walkthrough of visual evidence
   - **Technical Notes** — deeper observations for advanced users
6. Everything is returned to the frontend and stored in Supabase for history

---

## License

[MIT](LICENSE) © 2026 Viktor Ahnström
