import argparse
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix


def main(model_dir, emb_dir):
    model = tf.keras.models.load_model(model_dir)
    data = np.load(f"{emb_dir}/test.npz", allow_pickle=True)
    X_test, y_test = data["X"], data["y"]

    y_pred = np.argmax(model.predict(X_test), axis=1)

    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate classifier.")
    parser.add_argument("--model_dir", default="models/heart_yamnet")
    parser.add_argument("--emb_dir", default="data/embeddings")
    args = parser.parse_args()

    main(args.model_dir, args.emb_dir)
