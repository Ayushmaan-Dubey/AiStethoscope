import argparse
import csv
import os
import random
from glob import glob

CLASSES = ["AS", "MR", "MS", "MVP", "Normal"]
TRAIN_COUNT = 128
VAL_COUNT = 32
TEST_COUNT = 40


def write_csv(rows, out_path):
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "label"])
        writer.writerows(rows)


def main(data_dir, out_dir, seed):
    os.makedirs(out_dir, exist_ok=True)
    random.seed(seed)

    train_rows = []
    val_rows = []
    test_rows = []

    expected_per_class = TRAIN_COUNT + VAL_COUNT + TEST_COUNT

    for label in CLASSES:
        files = glob(os.path.join(data_dir, label, "*.wav"))
        files.sort()
        random.shuffle(files)

        if len(files) != expected_per_class:
            raise ValueError(
                f"{label} has {len(files)} files, expected {expected_per_class}."
            )

        train = files[:TRAIN_COUNT]
        val = files[TRAIN_COUNT : TRAIN_COUNT + VAL_COUNT]
        test = files[TRAIN_COUNT + VAL_COUNT : expected_per_class]

        train_rows.extend([(f, label) for f in train])
        val_rows.extend([(f, label) for f in val])
        test_rows.extend([(f, label) for f in test])

    write_csv(train_rows, os.path.join(out_dir, "train.csv"))
    write_csv(val_rows, os.path.join(out_dir, "val.csv"))
    write_csv(test_rows, os.path.join(out_dir, "test.csv"))

    print(
        "Split sizes:",
        len(train_rows),
        len(val_rows),
        len(test_rows),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create train/val/test splits for heart sound dataset."
    )
    parser.add_argument("--data_dir", default="data/raw")
    parser.add_argument("--out_dir", default="data/splits")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    main(args.data_dir, args.out_dir, args.seed)
