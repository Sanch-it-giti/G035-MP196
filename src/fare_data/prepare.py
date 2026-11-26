"""Run from repository root: python -m src.fare_data.prepare."""

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
import yaml

from .cleaning import clean_frame


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(config_path):
    root = config_path.resolve().parent
    with config_path.open() as handle:
        config = yaml.safe_load(handle)
    paths = config["data"]
    raw = root / paths["raw_dir"]
    output = root / paths["processed_dir"]
    sources = {"train": "train.csv", "later": "later.csv", "holdout": "holdout_features.csv"}
    before = {name: sha256(raw / filename) for name, filename in sources.items()}
    if before["train"] != paths["train_sha256"]:
        raise ValueError("Original train SHA-256 mismatch. Restore train.csv from the group ZIP.")
    # Validate all datasets before writing any output.
    cleaned = {}
    summary = {"raw_sha256": before, "datasets": {}}
    for source, filename in sources.items():
        original = pd.read_csv(raw / filename)
        cleaned[source] = clean_frame(original, source=source, settings=config["cleaning"])
        summary["datasets"][source] = {
            "input_rows": len(original), "output_rows": len(cleaned[source]),
            "missing_luggage": int(cleaned[source]["luggage_bags"].isna().sum()),
            "booking_levels": sorted(cleaned[source]["booking"].unique().tolist()),
            "receipt_removed": "fare_paid_receipt_rs" not in cleaned[source],
            "wait_conversion_rows": int(original["zone"].eq(config["cleaning"]["unit_group"]).sum()) if source == "train" else 0,
        }
    output.mkdir(parents=True, exist_ok=True)
    for source, filename in sources.items():
        destination = output / filename
        if destination.resolve() == (raw / filename).resolve():
            raise ValueError("Output must not overwrite raw data")
        cleaned[source].to_csv(destination, index=False)
    after = {name: sha256(raw / filename) for name, filename in sources.items()}
    if before != after:
        raise RuntimeError("Raw dataset changed during preparation")
    summary["raw_files_unchanged"] = True
    report = root / paths["report_path"]
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("params.yaml"))
    prepare(parser.parse_args().config)
