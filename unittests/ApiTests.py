#!/usr/bin/env python
"""
unittests/ApiTests.py
=====================
Unit tests for the Flask API in app.py.

The Flask test client is used, so no server has to be running. Every request
goes out in ``mode='test'``, which keeps the production models and logs
untouched.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app as app_module
import model as model_module
from model import model_train

DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "cs-train"
)


class ApiTest(unittest.TestCase):
    """Exercise every endpoint, including the error paths."""

    @classmethod
    def setUpClass(cls):
        # Train the test models once so /predict has something to serve.
        model_train(data_dir=DATA_DIR, test=True)
        # Configure the app for testing.
        app_module.app.config["TESTING"] = True
        cls.client = app_module.app.test_client()

    def test_01_index_lists_endpoints(self):
        """GET / returns 200 and documents the endpoints."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.assertEqual(body["model_version"], model_module.MODEL_VERSION)
        self.assertIn("POST /predict", body["endpoints"])
        self.assertIn("POST /train", body["endpoints"])
        self.assertIn("GET  /logs", body["endpoints"])

    def test_02_train_endpoint(self):
        """POST /train trains and reports which models were built."""
        response = self.client.post(
            "/train", json={"data_dir": DATA_DIR, "mode": "test"}
        )
        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.assertEqual(body["status"], "ok")
        self.assertIn("all", body["models_trained"])
        # rmse is reported per country tag, so 'all' must carry a number.
        self.assertIn("all", body["rmse"])
        self.assertGreaterEqual(body["rmse"]["all"], 0.0)

    def test_03_train_rejects_bad_mode(self):
        """An unknown mode is a client error, not a 500."""
        response = self.client.post("/train", json={"mode": "banana"})
        self.assertEqual(response.status_code, 400)
        self.assertIn("error", response.get_json())

    def test_04_train_rejects_bad_data_dir(self):
        """A missing data directory is reported as a 400."""
        response = self.client.post(
            "/train", json={"data_dir": os.path.join(DATA_DIR, "nope"), "mode": "test"}
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("error", response.get_json())

    def test_05_predict_endpoint(self):
        """POST /predict returns a finite prediction for a valid query."""
        # Pick a date that is definitely inside the test data range.
        all_models, all_data = model_module.model_load(
            prefix="test", data_dir=DATA_DIR, training=False
        )
        dates = all_data["all"]["dates"]
        year, month, day = dates[len(dates) // 2].split("-")

        response = self.client.post(
            "/predict",
            json={
                "query": {
                    "country": "all",
                    "year": year,
                    "month": month,
                    "day": day,
                },
                "mode": "test",
            },
        )
        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.assertEqual(body["status"], "ok")
        self.assertGreaterEqual(body["result"]["y_pred"], 0.0)
        self.assertEqual(body["result"]["tag"], "all")

    def test_06_predict_for_every_modelled_country(self):
        """A valid query works for every country that has a test model."""
        all_models, all_data = model_module.model_load(
            prefix="test", data_dir=DATA_DIR, training=False
        )
        for tag in sorted(all_models):
            if tag not in all_data:
                continue
            dates = all_data[tag]["dates"]
            if len(dates) == 0:
                continue
            year, month, day = dates[len(dates) // 2].split("-")
            response = self.client.post(
                "/predict",
                json={
                    "query": {
                        "country": tag, "year": year,
                        "month": month, "day": day,
                    },
                    "mode": "test",
                },
            )
            self.assertEqual(response.status_code, 200, "failed for {}".format(tag))
            self.assertGreaterEqual(
                response.get_json()["result"]["y_pred"], 0.0, tag
            )

    def test_07_predict_rejects_bad_requests(self):
        """Malformed bodies and bad queries all return 400."""
        cases = [
            {},                                                                   # no body
            {"mode": "test"},                                                     # no query
            {"query": "all", "mode": "test"},                                     # query not a dict
            {"query": {"year": "2018", "month": "03", "day": "09"}, "mode": "test"},  # no country
            {"query": {"country": "all", "month": "03", "day": "09"}, "mode": "test"},  # no year
            {"query": {"country": "all", "year": "2018", "month": "03", "day": "09"},
             "mode": "banana"},                                                   # bad mode
            {"query": {"country": "narnia", "year": "2018", "month": "03", "day": "09"},
             "mode": "test"},                                                     # unknown country
            {"query": {"country": "all", "year": "2018", "month": "13", "day": "09"},
             "mode": "test"},                                                     # bad month
        ]
        for body in cases:
            response = self.client.post("/predict", json=body)
            self.assertEqual(
                response.status_code, 400, "should have rejected {}".format(body)
            )
            self.assertIn("error", response.get_json())

    def test_08_logs_endpoint(self):
        """GET /logs returns the rows of an existing log file."""
        # Make sure at least one log row exists.
        self.client.post("/predict", json={
            "query": {"country": "all", "year": "2018", "month": "03", "day": "09"},
            "mode": "test",
        })
        response = self.client.get("/logs?filename=predict-test.csv")
        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.assertEqual(body["filename"], "predict-test.csv")
        self.assertTrue(isinstance(body["rows"], list))
        self.assertTrue(len(body["rows"]) > 0)
        self.assertIn("y_pred", body["rows"][0])

    def test_09_logs_rejects_bad_filename(self):
        """A missing filename, a non-csv and a traversal attempt are all refused."""
        for query in ("", "?filename=not_a_csv.txt", "?filename=../app.py",
                      "?filename=missing.csv"):
            response = self.client.get("/logs" + query)
            self.assertEqual(
                response.status_code, 400, "should have rejected /logs" + query
            )
            self.assertIn("error", response.get_json())


if __name__ == "__main__":
    unittest.main(verbosity=2)
