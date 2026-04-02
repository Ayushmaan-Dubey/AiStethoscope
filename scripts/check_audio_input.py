import argparse
import json
import struct
import wave
from pathlib import Path


def inspect_wav(path: Path) -> dict[str, object]:
    with wave.open(str(path), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_rate = wav_file.getframerate()
        sample_width = wav_file.getsampwidth()
        frame_count = wav_file.getnframes()
        preview_frames = wav_file.readframes(min(2000, frame_count))

    if sample_width != 2:
        raise ValueError(
            f"Expected 16-bit PCM WAV input for this smoke check, got sample width={sample_width}."
        )

    sample_count = len(preview_frames) // sample_width
    pcm = struct.unpack("<" + "h" * sample_count, preview_frames)
    if channels == 2:
        mono = [((pcm[i] + pcm[i + 1]) / 2.0) for i in range(0, len(pcm), 2)]
    else:
        mono = list(pcm)

    duration_seconds = frame_count / float(sample_rate)
    target_16k_samples = max(1, round(frame_count * 16000 / sample_rate))
    peak_pcm_amplitude = max(abs(value) for value in mono) if mono else 0

    return {
        "path": str(path),
        "channels": channels,
        "sample_rate_hz": sample_rate,
        "sample_width_bytes": sample_width,
        "duration_seconds": round(duration_seconds, 3),
        "target_16k_samples": target_16k_samples,
        "peak_pcm_amplitude_preview": int(peak_pcm_amplitude),
        "backend_contract": {
            "wav_extension_supported": path.suffix.lower() == ".wav",
            "mono_conversion_needed": channels > 1,
            "resample_to_16k_needed": sample_rate != 16000,
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Inspect a WAV file against the backend's expected upload contract."
    )
    parser.add_argument("path", help="Path to the WAV file to inspect.")
    args = parser.parse_args()

    result = inspect_wav(Path(args.path))
    print(json.dumps(result, indent=2))
