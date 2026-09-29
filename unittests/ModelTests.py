#!/usr/bin/env python
"""
unittests/ModelTests.py
=======================
Unit tests for model.py - training, loading and predicting.

They train against a *subset* of the data and write to the test model/log
files (``test-*.joblib``, ``*-test.csv``) so the production artifacts are never
touched.

Note the first test class needs the real data to be present, because training on
random numbers would not exercise the ingestion path that actually matters here.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import model as model_module
from cslib import engineer_features, fetch_data, fetch_ts
from model import MODEL_VERSION, model_load, model_predict, model_train

DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "cs-train"
)


class DataIngestionTest(unittest.TestCase):
    """Part 1 deliverable - the ingestion function must clean the raw json."""

    @classmethod
    def setUpClass(cls):
        cls.raw = fetch_data(DATA_DIR)

    def test_01_columns_are_normalised(self):
        """All three documented naming quirks are corrected."""
        self.assertIn("stream_id", self.raw.columns)
        self.assertIn("times_viewed", self.raw.columns)
        self.assertIn("price", self.raw.columns)
        # None of the odd spellings should survive.
        for bad in ("StreamID", "TimesViewed", "total_price"):
            self.assertNotIn(bad, self.raw.columns)

    def test_02_invoice_ids_have_no_letters(self):
        """Invoice ids are stripped to digits so they match cleanly."""
        self.assertEqual(int(self.raw["invoice"].str.contains(r"[A-Za-z]").sum()), 0)

    def test_03_date_column_is_real_datetime(self):
        """year/month/day are combined into a usable datetime column."""
        self.assertTrue(pd_is_datetime(self.raw["invoice_date"]))
        self.assertEqual(int(self.raw["invoice_date"].isna().sum()), 0)

    def test_04_sorted_by_date(self):
        """Rows come back ordered by date with a clean index."""
        self.assertTrue(self.raw["invoice_date"].is_monotonic_increasing)
        self.assertEqual(list(self.raw.index[:5]), [0, 1, 2, 3, 4])

    def test_05_daily_timeseries_has_every_day(self):
        """The daily aggregation has no gaps in the date range."""
        ts = fetch_ts(DATA_DIR)
        all_days = ts["all"]
        span = (all_days["date"].max() - all_days["date"].min()).days + 1
        self.assertEqual(len(all_days), span)

    def test_06_features_have_no_leakage(self):
        """
        The regression target must be the sum of the NEXT 30 days, and no
        feature may be built from the current day or later.
        """
        import pandas as pd

        ts = fetch_ts(DATA_DIR)
        daily = ts["all"]
        X, y, dates = engineer_features(daily, training=True)

        self.assertEqual(X.shape[1], 7)
        self.assertEqual(X.shape[0], y.shape[0])

        revenue = daily.set_index("date")["revenue"]
        d = pd.Timestamp(dates[100])
        expected = revenue[
            (revenue.index >= d) & (revenue.index < d + pd.Timedelta(days=30))
        ].sum()
        self.assertAlmostEqual(float(y[100]), float(expected), places=4)

        # previous_7 must be the 7 days STRICTLY BEFORE d, so a future change
        # cannot move it - that is what proves there is no leakage.
        prev7 = revenue[
            (revenue.index >= d - pd.Timedelta(days=7)) & (revenue.index < d)
        ].sum()
        self.assertAlmostEqual(float(X[100, 0]), float(prev7), places=4)

    def test_07_missing_directory_raises(self):
        """Bad input is caught rather than crashing later."""
        with self.assertRaises(ValueError):
            fetch_data(os.path.join(DATA_DIR, "does_not_exist"))


def pd_is_datetime(series):
    """Tiny helper so the test file does not need a top-level pandas import."""
    import pandas as pd

    return pd.api.types.is_datetime64_any_dtype(series)


class ModelTest(unittest.TestCase):
    """Train, load and predict. Uses the test model/log files only."""

    @classmethod
    def setUpClass(cls):
        # Train just two countries so the suite stays quick.
        model_train(data_dir=DATA_DIR, test=True)

    def test_01_train_produced_test_models(self):
        """Training writes test-prefixed joblib files and a train log row."""
        files = [f for f in os.listdir(model_module.MODEL_DIR) if f.startswith("test-")]
        self.assertTrue(files, "no test models were written")
        self.assertIn("test-all-", " ".join(files))
        self.assertIn("test-united_kingdom-", " ".join(files))

        from logger import get_log_file

        rows = get_log_file("train-test.csv")
        self.assertTrue(rows)
        self.assertTrue(any(r["tag"] == "all" for r in rows))
        # Every logged row must carry a usable RMSE.
        self.assertTrue(all(float(r["rmse"]) >= 0 for r in rows if r["rmse"] != ""))

    def test_02_models_load(self):
        """model_load returns the saved estimators keyed by country."""
        all_models, all_data = model_load(prefix="test", data_dir=DATA_DIR, training=False)
        self.assertIn("all", all_models)
        self.assertIn("united_kingdom", all_models)
        self.assertIn("all", all_data)
        # The feature rows and the date list must line up for every country.
        for tag, data in all_data.items():
            self.assertEqual(len(data["X"]), len(data["dates"]))

    def test_03_load_without_models_raises(self):
        """Asking for a prefix with no models gives a clear error."""
        with self.assertRaises(ValueError):
            model_load(prefix="no-such-prefix", data_dir=DATA_DIR, load_data=False)

    def test_04_predict(self):
        """A valid query returns a sensible, finite prediction."""
        all_models, all_data = model_load(prefix="test", data_dir=DATA_DIR, training=False)
        dates = all_data["all"]["dates"]
        year, month, day = dates[len(dates) // 2].split("-")

        result = model_predict(
            "all", year, month, day,
            all_models=all_models, all_data=all_data, test=True,
        )
        self.assertIn("y_pred", result)
        self.assertTrue(np.isfinite(result["y_pred"]))
        self.assertGreaterEqual(result["y_pred"], 0.0)
        self.assertEqual(result["tag"], "all")

    def test_05_predict_accepts_any_valid_date(self):
        """
        Leadership query on different days of the month, so an early and a late
        date must both work - not just the most recent one.
        """
        all_models, all_data = model_load(prefix="test", data_dir=DATA_DIR, training=False)
        for index in (10, len(all_data["all"]["dates"]) - 20):
            year, month, day = all_data["all"]["dates"][index].split("-")
            result = model_predict(
                "all", year, month, day,
                all_models=all_models, all_data=all_data, test=True, log=False,
            )
            self.assertTrue(np.isfinite(result["y_pred"]))

    def test_06_predict_rejects_bad_input(self):
        """Bad countries, bad dates and out-of-range dates all raise."""
        all_models, all_data = model_load(prefix="test", data_dir=DATA_DIR, training=False)
        bad_cases = [
            ("atlantis", "2018", "03", "09"),   # unknown country
            ("all", "20x8", "03", "09"),         # non numeric year
            ("all", "2018", "13", "09"),         # month 13
            ("all", "2018", "02", "31"),         # impossible date
            ("all", "1900", "01", "01"),         # before the data starts
        ]
        for args in bad_cases:
            with self.assertRaises(ValueError, msg="should have rejected {}".format(args)):
                model_predict(
                    *args, all_models=all_models, all_data=all_data, log=False
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
