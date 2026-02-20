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

Install dependencies (no `tensorflow-io` required):

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install tensorflow tensorflow-hub numpy pandas scikit-learn soundfile
