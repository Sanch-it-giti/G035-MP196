"""Pure cleaning functions: no fitting and no writes to source data."""

import pandas as pd

FEATURES = [
    "distance_km", "wait_min", "passengers", "hour",
    "luggage_bags", "fuel_price_rs", "booking", "zone",
]
NUMERIC_FEATURES = FEATURES[:6]
TARGET = "fare_rs"
BOOKING_ALIASES = {"str.": "street", "sta.": "stand", "app.": "app"}


def clean_frame(frame, *, source, settings):
    """Clean one original dataset; source must be train, later, or holdout.

    Only original train has the seconds/minutes defect. Do not pass processed
    training data here a second time. Missing luggage stays missing so the
    model pipeline can fit its imputer on the training split only.
    """
    if source not in {"train", "later", "holdout"}:
        raise ValueError("source must be train, later, or holdout")
    required = ["record_id", "date", *FEATURES]
    if source != "holdout":
        required.append(TARGET)
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Selecting an explicit allowlist excludes the receipt/leaking column.
    result = frame.loc[:, required].copy()
    if source == "train":
        # Compare complete original rows, not just the selected features.
        result = result.loc[~frame.duplicated()].copy()
    result["date"] = pd.to_datetime(result["date"], errors="raise").dt.strftime("%Y-%m-%d")
    numeric = NUMERIC_FEATURES + ([TARGET] if source != "holdout" else [])
    for column in numeric:
        result[column] = pd.to_numeric(result[column], errors="raise")
    booking = result["booking"].astype("string").str.strip().str.lower()
    result["booking"] = booking.replace(BOOKING_ALIASES)
    if result["booking"].isna().any() or not result["booking"].isin(["street", "app", "stand"]).all():
        raise ValueError("Unknown or missing booking category")
    if not result["zone"].isin(["Station Area", "Market", "Outskirts"]).all():
        raise ValueError("Unknown or missing zone")
    result["wait_min"] = result["wait_min"].astype(float)
    if source == "train":
        affected = result["zone"].eq(settings["unit_group"])
        result.loc[affected, "wait_min"] /= settings["wait_divisor"]
    result["luggage_bags"] = result["luggage_bags"].replace(settings["luggage_sentinel"], float("nan"))
    return result.reset_index(drop=True)
