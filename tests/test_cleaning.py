"""Synthetic-data checks; no original CSV files required."""

import unittest

import pandas as pd

from src.fare_data.cleaning import clean_frame

SETTINGS = {"unit_group": "Station Area", "wait_divisor": 60, "luggage_sentinel": -1}


def sample():
    return pd.DataFrame([{
        "record_id": "test-1", "date": "2026-01-02", "distance_km": 5.0,
        "wait_min": 600.0, "passengers": 2, "hour": 12, "luggage_bags": -1,
        "fuel_price_rs": 100.0, "booking": " STR. ", "zone": "Station Area",
        "fare_paid_receipt_rs": 100, "fare_rs": 99,
    }])


class CleaningTests(unittest.TestCase):
    def test_training_units_and_original_unchanged(self):
        frame = sample()
        original = frame.copy(deep=True)
        cleaned = clean_frame(frame, source="train", settings=SETTINGS)
        self.assertEqual(cleaned.loc[0, "wait_min"], 10)
        pd.testing.assert_frame_equal(frame, original)

    def test_other_zone_and_later_not_converted(self):
        frame = sample()
        frame["wait_min"] = 12.0
        self.assertEqual(clean_frame(frame, source="later", settings=SETTINGS).loc[0, "wait_min"], 12)
        frame["zone"] = "Market"
        self.assertEqual(clean_frame(frame, source="train", settings=SETTINGS).loc[0, "wait_min"], 12)

    def test_sentinel_missing_and_spelling_normalized(self):
        cleaned = clean_frame(sample(), source="train", settings=SETTINGS)
        self.assertTrue(pd.isna(cleaned.loc[0, "luggage_bags"]))
        self.assertEqual(cleaned.loc[0, "booking"], "street")
        self.assertNotIn("fare_paid_receipt_rs", cleaned.columns)

    def test_extra_duplicate_removed(self):
        frame = pd.concat([sample(), sample()], ignore_index=True)
        self.assertEqual(len(clean_frame(frame, source="train", settings=SETTINGS)), 1)

    def test_holdout_requires_no_target(self):
        frame = sample().drop(columns=["fare_rs", "fare_paid_receipt_rs"])
        cleaned = clean_frame(frame, source="holdout", settings=SETTINGS)
        self.assertNotIn("fare_rs", cleaned.columns)
        self.assertEqual(cleaned.loc[0, "record_id"], "test-1")

    def test_invalid_input_fails_explicitly(self):
        with self.assertRaises(ValueError):
            clean_frame(sample(), source="unknown", settings=SETTINGS)
        frame = sample()
        frame["booking"] = "unknown"
        with self.assertRaises(ValueError):
            clean_frame(frame, source="train", settings=SETTINGS)
        with self.assertRaises(ValueError):
            clean_frame(sample().drop(columns="distance_km"), source="train", settings=SETTINGS)


if __name__ == "__main__":
    unittest.main()
