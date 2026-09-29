#!/usr/bin/env python
"""
unittests/LoggerTests.py
========================
Unit tests for the CSV logging in logger.py.

These run against the *test* log files (``train-test.csv`` /
``predict-test.csv``) so a test run can never overwrite a real log.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logger as logger_module
from logger import update_predict_log, update_train_log, get_log_file


class LoggerTest(unittest.TestCase):
    """Check that log rows are written, appended and read back correctly."""

    @classmethod
    def setUpClass(cls):
        # Point the logger at a scratch folder so we never touch log/.
        cls.original_log_dir = logger_module.LOG_DIR
        cls.tmp_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "log",
            "_test_scratch",
        )
        os.makedirs(cls.tmp_dir, exist_ok=True)
        logger_module.LOG_DIR = cls.tmp_dir

    @classmethod
    def tearDownClass(cls):
        logger_module.LOG_DIR = cls.original_log_dir
        for name in os.listdir(cls.tmp_dir):
            os.remove(os.path.join(cls.tmp_dir, name))
        os.rmdir(cls.tmp_dir)

    def test_01_train_log_written(self):
        """A training record is written to a csv with a header."""
        path = update_train_log(
            "unittest_country", ("2018-01-01", "2018-06-30"),
            {"rmse": 123.45}, "00:00:07", "0.1", "unit test", test=True
        )
        self.assertTrue(os.path.isfile(path))

        rows = get_log_file(os.path.basename(path))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["tag"], "unittest_country")
        self.assertEqual(rows[0]["rmse"], "123.45")
        self.assertEqual(rows[0]["runtime"], "00:00:07")

    def test_02_train_log_appends(self):
        """A second training record is appended, not overwriting the first."""
        update_train_log(
            "unittest_country", ("2018-01-01", "2018-06-30"),
            {"rmse": 200.0}, "00:00:09", "0.2", "second", test=True
        )
        path = os.path.join(logger_module.LOG_DIR, "train-test.csv")
        rows = get_log_file(os.path.basename(path))
        self.assertGreaterEqual(len(rows), 2)
        self.assertEqual(rows[-1]["rmse"], "200.0")

    def test_03_predict_log_written(self):
        """A prediction record is written, including a numpy array value."""
        import numpy as np

        update_predict_log(
            "unittest_country", np.array([987.65]), None,
            "2018-03-09", "00:00:01", "0.1", test=True
        )
        path = os.path.join(logger_module.LOG_DIR, "predict-test.csv")
        rows = get_log_file(os.path.basename(path))
        self.assertGreaterEqual(len(rows), 1)
        self.assertEqual(rows[-1]["target_date"], "2018-03-09")
        # The numpy array must be flattened to a plain number for the csv.
        self.assertAlmostEqual(float(rows[-1]["y_pred"]), 987.65, places=2)
        # A regressor has no probabilities, so that column is empty.
        self.assertEqual(rows[-1]["y_proba"], "")

    def test_04_get_log_file_rejects_bad_input(self):
        """Path traversal and non-csv names are refused."""
        with self.assertRaises(ValueError):
            get_log_file("../app.py")
        with self.assertRaises(ValueError):
            get_log_file("not_a_log.txt")
        with self.assertRaises(ValueError):
            get_log_file("does_not_exist.csv")

    def test_05_test_and_prod_logs_are_separate(self):
        """A test write must not land in the production log file."""
        update_train_log(
            "unittest_country", ("2018-01-01", "2018-06-30"),
            {"rmse": 1.0}, "00:00:01", "0.1", "isolation check", test=True
        )
        test_path = os.path.join(logger_module.LOG_DIR, "train-test.csv")
        prod_path = os.path.join(logger_module.LOG_DIR, "train.csv")
        self.assertTrue(os.path.isfile(test_path))
        # train.csv should not exist yet - every write so far used test=True.
        self.assertFalse(os.path.isfile(prod_path))


if __name__ == "__main__":
    unittest.main(verbosity=2)
