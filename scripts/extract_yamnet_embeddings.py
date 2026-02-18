import argparse
import csv
import json
import os

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_io as tfio

CLASSES = ["AS", "MR", "MS", "MVP", "Normal"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}


def load_wav_16k(path):
    wav_bytes = tf.io.read_file(path)
    audio, sr = tf.audio.decode_wav(wav_bytes, desired_channels=1)
    audio = tf.squeeze(audio, axis=-1)

    sr_int = int(sr.numpy())
    if sr_int != 16000:
        audio = tfio.audio.resample(audio, rate_in=sr_int, rate_out=16000)

    audio = tf.clip_by_value(audio, -1.0, 1.0)
    return audio


def embed_file(yamnet, path):
    waveform = load_wav_16k(path)
    scores, embeddings, spectrogram = yamnet(waveform)
    emb = tf.reduce_mean(embeddings, axis=0)
    return emb.numpy()


def load_csv(csv_path):
    rows = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append((row["path"], row["label"]))
    return rows


def main(splits_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    yamnet = hub.load("https://tfhub.dev/google/yamnet/1")

    for split in ["train", "val", "test"]:
        rows = load_csv(os.path.join(splits_dir, f"{split}.csv"))
        X, y, paths = [], [], []

        for path, label in rows:
            X.append(embed_file(yamnet, path))
            y.append(CLASS_TO_IDX[label])
            paths.append(path)

        np.savez(
            os.path.join(out_dir, f"{split}.npz"),
            X=np.stack(X),
            y=np.array(y, dtype=np.int64),
            paths=np.array(paths),
        )

    with open(os.path.join(out_dir, "labels.json"), "w") as f:
        json.dump(CLASSES, f)

    print("Embeddings saved to", out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract YAMNet embeddings.")
    parser.add_argument("--splits_dir", default="data/splits")
    parser.add_argument("--out_dir", default="data/embeddings")
    args = parser.parse_args()

    main(args.splits_dir, args.out_dir)
