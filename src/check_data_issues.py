"""Reproduce Stage 1 evidence; never write to the original CSV files.

Run from the repository root: python src/check_data_issues.py
These are diagnostic summaries, not an automatic proof of data semantics.
"""

import pandas as pd


def main():
    train = pd.read_csv("data/raw/train.csv")
    later = pd.read_csv("data/raw/later.csv")
    holdout = pd.read_csv("data/raw/holdout_features.csv")

    print("1. LEAKAGE: receipt is created after payment")
    print("Absent from holdout:", sorted(set(train) - set(holdout)))
    print(train[["fare_paid_receipt_rs", "fare_rs"]].corr().to_string())

    print("\n2. UNIT MIXUP: compare zones before any conversion")
    features = ["distance_km", "wait_min", "passengers", "hour",
                "luggage_bags", "fuel_price_rs"]
    print(train.groupby("zone")[features].agg(["min", "median", "max"]).to_string())
    print("Station Area wait_min / 60 (seconds to minutes):")
    print(train.loc[train["zone"].eq("Station Area"), "wait_min"].div(60).describe())

    print("\n3. SENTINELS: inspect repeated extreme values")
    for column in features:
        counts = train[column].value_counts().sort_index()
        print(column, "lowest values:", counts.head(3).to_dict(),
              "highest values:", counts.tail(3).to_dict())
    print("Invalid luggage_bags = -1 count:", train["luggage_bags"].eq(-1).sum())

    print("\n4. DUPLICATES: extra complete-row copies")
    print(train.duplicated().sum())

    print("\n5. STUCK SENSOR: longest unchanged observations by date")
    ordered = train.copy()
    ordered["date"] = pd.to_datetime(ordered["date"])
    ordered = ordered.sort_values("date", kind="stable")
    for column in ordered.select_dtypes(include="number").columns:
        blocks = ordered[column].ne(ordered[column].shift()).cumsum()
        runs = ordered.groupby(blocks).agg(
            value=(column, "first"), rows=(column, "size"),
            start=("date", "min"), end=("date", "max"),
        )
        runs["span_days"] = (runs["end"] - runs["start"]).dt.days
        print(column)
        print(runs.sort_values(["span_days", "rows"], ascending=False).head(3).to_string(index=False))

    print("\n6. TARGET TYPOS: inspect upper tail for isolated tenfold values")
    print(train["fare_rs"].describe().to_string())
    print(train["fare_rs"].nlargest(20).to_string(index=False))
    values = train["fare_rs"].sort_values().drop_duplicates()
    print("Largest adjacent target ratios:")
    print(values.div(values.shift()).nlargest(5).to_string(index=False))

    print("\n7. SPELLING VARIANTS: raw values versus clean reference")
    for column in ["booking", "zone"]:
        print(column, "train:")
        print(train[column].map(repr).value_counts().to_string())
        print(column, "later:")
        print(later[column].map(repr).value_counts().to_string())
    print("Clean booking levels:", later["booking"].nunique())


if __name__ == "__main__":
    main()
