# AI Stethoscope - Heart Sound Classifier

This repository contains a lightweight pipeline to train and evaluate a heart
sound classifier using YAMNet embeddings. It includes scripts for dataset
splits, embedding extraction, model training, and evaluation.

## Repository layout

- `scripts/` - Pipeline scripts
- `data/raw/` - Raw WAV files organized by class
- `data/splits/` - Train/val/test CSVs
- `data/embeddings/` - Extracted embeddings (`*.npz`)
- `models/` - Trained model artifacts

## Requirements

- Python 3.11+

Install dependencies:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install tensorflow tensorflow-hub numpy pandas scikit-learn soundfile
```

## Dataset layout

This repository now expects the following class folders under `data/raw/`:

- `AS`
- `MR`
- `MS`
- `MVP`
- `N`

The downloaded dataset used in this workspace contains 200 `.wav` files per class.

## Training pipeline

Generate deterministic train/validation/test splits:

```bash
python scripts/make_splits.py
```

Extract YAMNet embeddings:

```bash
python scripts/extract_yamnet_embeddings.py
```

Train the classifier and export it into `models/heart_yamnet/`:

```bash
python scripts/train_classifier.py
```

Evaluate the exported classifier:

```bash
python scripts/evaluate.py
```

## Web app

The repository also includes a local-only web interface:

- Backend API: `app/backend/main.py`
- Frontend app: `app/frontend/`

### Backend setup

Create or activate a Python environment with the backend dependencies:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements-ui.txt
```

Start the API:

```bash
uvicorn app.backend.main:app --reload
```

The backend serves:

- `GET /health`
- `POST /predict`

### Frontend setup

Install frontend dependencies:

```bash
cd app/frontend
npm install
```

Start the local Vite dev server:

```bash
npm run dev
```

Open the frontend at `http://127.0.0.1:5173`.

## Local-only features

The current UI includes:

- drag-and-drop WAV upload
- in-browser audio playback
- waveform preview drawn in the browser
- ranked class probabilities
- browser-local analysis history stored in `localStorage`

No remote persistence or cloud services are configured.

## Verification status

Within this Codex session, the following were verified locally:

- dataset extraction into `data/raw/`
- class counts for all five classes
- split generation via `scripts/make_splits.py`
- model artifact presence at `models/heart_yamnet/model.keras`
- backend/frontend source syntax checks

The full backend + frontend runtime was not launched inside this session because the current sandboxed environment does not have installable network access for missing packages such as `fastapi`, `uvicorn`, and frontend `node_modules`.
