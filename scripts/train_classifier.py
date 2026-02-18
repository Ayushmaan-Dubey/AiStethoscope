import argparse
import os

import numpy as np
import tensorflow as tf


def load_npz(path):
    data = np.load(path, allow_pickle=True)
    return data["X"], data["y"]


def main(emb_dir, out_dir, epochs, batch_size):
    os.makedirs(out_dir, exist_ok=True)

    X_train, y_train = load_npz(os.path.join(emb_dir, "train.npz"))
    X_val, y_val = load_npz(os.path.join(emb_dir, "val.npz"))
    X_test, y_test = load_npz(os.path.join(emb_dir, "test.npz"))

    num_classes = len(np.unique(y_train))

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(1024,)),
            tf.keras.layers.Dense(512, activation="relu"),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    callbacks = [tf.keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True)]

    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=2,
    )

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print("Test accuracy:", test_acc)

    model.save(out_dir, include_optimizer=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train classifier on YAMNet embeddings.")
    parser.add_argument("--emb_dir", default="data/embeddings")
    parser.add_argument("--out_dir", default="models/heart_yamnet")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    main(args.emb_dir, args.out_dir, args.epochs, args.batch_size)
