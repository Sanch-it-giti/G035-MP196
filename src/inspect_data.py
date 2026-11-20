"""Inspect original data without modifying it."""

import hashlib

import pandas as pd


def main():
    train_path = "data/raw/train.csv"
    train = pd.read_csv(train_path)
    holdout = pd.read_csv("data/raw/holdout_features.csv")

    with open(train_path, "rb") as file:
        print("Original train SHA-256:")
        print(hashlib.sha256(file.read()).hexdigest())

    print("\nRows and columns:", train.shape)
    print("\nColumn types:")
    print(train.dtypes)

    print("\nMissing values:")
    print(train.isna().sum())

    print("\nExtra duplicate rows:", train.duplicated().sum())

    print("\nColumns absent from holdout:")
    print(sorted(set(train.columns) - set(holdout.columns)))

    print("\nNumeric summary:")
    print(train.describe().to_string())

    for column in ["booking", "zone"]:
        print(f"\nExact category counts for {column}:")
        print(train[column].map(repr).value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
