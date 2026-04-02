from __future__ import annotations

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.backend.inference import (
    HeartSoundInferenceService,
    InvalidAudioError,
    ModelNotReadyError,
)

app = FastAPI(
    title="AI Stethoscope API",
    version="0.1.0",
    description="Inference API for uploading heart sound recordings and receiving classifier predictions.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

service = HeartSoundInferenceService()


@app.get("/health")
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "model_ready": service.model_dir.exists(),
        "expected_model_dir": str(service.model_dir),
        "classes": [
            {"label": "AS", "display_name": "Aortic Stenosis"},
            {"label": "MR", "display_name": "Mitral Regurgitation"},
            {"label": "MS", "display_name": "Mitral Stenosis"},
            {"label": "MVP", "display_name": "Mitral Valve Prolapse"},
            {"label": "N", "display_name": "Normal"},
        ],
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict[str, object]:
    if not file.filename:
        raise HTTPException(status_code=400, detail="A WAV file is required.")

    if not file.filename.lower().endswith(".wav"):
        raise HTTPException(status_code=400, detail="Only .wav files are supported.")

    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")

    try:
        result = service.predict_bytes(audio_bytes)
    except ModelNotReadyError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except InvalidAudioError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - defensive API boundary
        raise HTTPException(status_code=500, detail="Prediction failed unexpectedly.") from exc

    return {
        "filename": file.filename,
        "predicted_class": result.predicted_class,
        "predicted_display_name": result.predicted_display_name,
        "predicted_index": result.predicted_index,
        "confidence": result.confidence,
        "probabilities": result.probabilities,
        "ranked_predictions": result.ranked_predictions,
        "top_classes": result.top_classes,
        "audio_seconds": result.audio_seconds,
        "sample_rate_hz": result.sample_rate_hz,
        "message": (
            f"The current classifier assigns the highest posterior probability to "
            f"{result.predicted_display_name} at {result.confidence * 100:.1f}%."
        ),
        "interpretation": (
            "This output should be interpreted as a model estimate derived from an uploaded phonocardiogram segment."
        ),
        "disclaimer": (
            "This interface is intended for academic and research demonstration only and does not constitute a clinical diagnosis."
        ),
    }
