from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import tensorflow as tf
import tensorflow_hub as hub

CLASSES = ["AS", "MR", "MS", "MVP", "N"]
DISPLAY_NAMES = {
    "AS": "Aortic Stenosis",
    "MR": "Mitral Regurgitation",
    "MS": "Mitral Stenosis",
    "MVP": "Mitral Valve Prolapse",
    "N": "Normal",
}
DEFAULT_MODEL_DIR = Path("models/heart_yamnet")
YAMNET_HANDLE = "https://tfhub.dev/google/yamnet/1"


class InferenceError(RuntimeError):
    """Base exception for inference failures."""


class InvalidAudioError(InferenceError):
    """Raised when uploaded audio cannot be decoded safely."""


class ModelNotReadyError(InferenceError):
    """Raised when the trained classifier has not been exported yet."""


@dataclass
class PredictionResult:
    predicted_class: str
    predicted_display_name: str
    predicted_index: int
    confidence: float
    probabilities: dict[str, float]
    ranked_predictions: list[dict[str, float | str | int]]
    top_classes: list[dict[str, float | str]]
    audio_seconds: float
    sample_rate_hz: int


class HeartSoundInferenceService:
    def __init__(self, model_dir: Path | str = DEFAULT_MODEL_DIR) -> None:
        self.model_dir = Path(model_dir)
        self._yamnet = None
        self._classifier = None

    def _load_yamnet(self):
        if self._yamnet is None:
            self._yamnet = hub.load(YAMNET_HANDLE)
        return self._yamnet

    def _load_classifier(self):
        if self._classifier is None:
            model_file = self.model_dir / "model.keras"
            if model_file.exists():
                self._classifier = tf.keras.models.load_model(model_file)
                return self._classifier

            if not self.model_dir.exists():
                raise ModelNotReadyError(
                    f"Trained model not found at '{self.model_dir}'. Train and export the classifier first."
                )
            self._classifier = tf.keras.models.load_model(self.model_dir)
        return self._classifier

    def _decode_audio(self, audio_bytes: bytes) -> tuple[np.ndarray, int]:
        try:
            audio, sample_rate = sf.read(io.BytesIO(audio_bytes), dtype="float32")
        except Exception as exc:  # pragma: no cover - soundfile raises different subclasses
            raise InvalidAudioError("The uploaded file could not be decoded as audio.") from exc

        if audio.size == 0:
            raise InvalidAudioError("The uploaded audio file is empty.")

        if audio.ndim == 2:
            audio = np.mean(audio, axis=1)

        if sample_rate <= 0:
            raise InvalidAudioError("The uploaded audio file has an invalid sample rate.")

        audio = np.clip(audio, -1.0, 1.0).astype(np.float32)
        return audio, int(sample_rate)

    def _resample_to_16k(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        if sample_rate == 16000:
            return audio

        target_length = max(1, round(len(audio) * 16000 / sample_rate))
        source_positions = np.linspace(0.0, 1.0, num=len(audio), endpoint=False)
        target_positions = np.linspace(0.0, 1.0, num=target_length, endpoint=False)
        resampled = np.interp(target_positions, source_positions, audio)
        return resampled.astype(np.float32)

    def _embed_audio(self, audio: np.ndarray) -> np.ndarray:
        yamnet = self._load_yamnet()
        _, embeddings, _ = yamnet(audio)
        pooled = tf.reduce_mean(embeddings, axis=0)
        return pooled.numpy().astype(np.float32)

    def predict_bytes(self, audio_bytes: bytes) -> PredictionResult:
        classifier = self._load_classifier()
        audio, sample_rate = self._decode_audio(audio_bytes)
        waveform = self._resample_to_16k(audio, sample_rate)
        embedding = self._embed_audio(waveform)

        probs = classifier.predict(embedding[None, :], verbose=0)[0]
        predicted_index = int(np.argmax(probs))
        probabilities = {label: float(prob) for label, prob in zip(CLASSES, probs)}
        ranked_predictions = [
            {
                "label": CLASSES[idx],
                "display_name": DISPLAY_NAMES[CLASSES[idx]],
                "probability": float(probs[idx]),
                "rank": rank + 1,
            }
            for rank, idx in enumerate(np.argsort(probs)[::-1])
        ]
        top_classes = [
            {
                "label": item["label"],
                "display_name": item["display_name"],
                "probability": item["probability"],
            }
            for item in ranked_predictions[:3]
        ]

        return PredictionResult(
            predicted_class=CLASSES[predicted_index],
            predicted_display_name=DISPLAY_NAMES[CLASSES[predicted_index]],
            predicted_index=predicted_index,
            confidence=float(probs[predicted_index]),
            probabilities=probabilities,
            ranked_predictions=ranked_predictions,
            top_classes=top_classes,
            audio_seconds=round(len(audio) / sample_rate, 2),
            sample_rate_hz=sample_rate,
        )
