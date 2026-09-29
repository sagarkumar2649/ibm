#!/usr/bin/env python
"""
logger.py
=========
Append-only CSV logging for model training and prediction runs.

Why this exists
---------------
The assignment asks the API to expose a *logfile* endpoint and asks us to be
able to monitor a model after it goes into production. That is only possible if
every training run and every prediction is recorded. This module is the single
place that writes those records.

Two things matter for correctness:

1. **Test runs must never pollute production logs.** Unit tests write to
   ``*-test.csv`` files, real runs to the plain ``*.csv`` files.
2. **Appending must be safe.** Each call reads the existing file, adds a row
   and writes the whole thing back, creating the header on first use.
"""

import csv
import os
import uuid
from datetime import datetime

# The project root, so the log folder resolves no matter where this is called
# from (project root, notebooks/, or inside the Docker container).
# __file__ is <project_root>/logger.py, so a single dirname() gives the root.
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "log")

# Column order for each log file.
TRAIN_FIELDS = [
    "uuid",
    "timestamp",
    "tag",
    "model_version",
    "model_version_note",
    "train_start_date",
    "train_end_date",
    "rmse",
    "runtime",
    "test",
]

PREDICT_FIELDS = [
    "uuid",
    "timestamp",
    "tag",
    "model_version",
    "target_date",
    "y_pred",
    "y_proba",
    "runtime",
    "test",
]


def _ensure_log_dir():
    """Create the log directory if it is not there yet."""
    if not os.path.isdir(LOG_DIR):
        os.makedirs(LOG_DIR, exist_ok=True)


def _log_path(kind, test):
    """
    Work out the file to write to.

    ``kind`` is 'train' or 'predict'. When ``test`` is True a '-test' suffix is
    added so the unit tests can never overwrite a real log.
    """
    _ensure_log_dir()
    suffix = "-test" if test else ""
    return os.path.join(LOG_DIR, "{}{}.csv".format(kind, suffix))


def _read_existing(path, fields):
    """Read a log file into a list of dicts. Returns [] when the file is absent."""
    if not os.path.exists(path):
        return []
    with open(path, "r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows = []
        for row in reader:
            # Ignore anything that does not line up with the expected schema.
            if all(f in row for f in fields):
                rows.append(row)
        return rows


def _append(path, fields, row):
    """Write a header if needed, then rewrite the file with the new row added."""
    rows = _read_existing(path, fields)
    rows.append(row)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _now():
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S")


def update_train_log(
    tag,
    date_range,
    metrics,
    runtime,
    model_version,
    model_version_note="",
    test=False,
):
    """
    Record that a model was trained.

    Parameters
    ----------
    tag : str
        Which model was trained, e.g. ``'all'`` or ``'united_kingdom'``.
    date_range : tuple
        ``(first_date, last_date)`` covered by the training data.
    metrics : dict
        Evaluation metrics, e.g. ``{'rmse': 1234.5}``.
    runtime : str
        How long training took, formatted ``HH:MM:SS``.
    model_version : str
        Version string of the model, e.g. ``'0.1'``.
    model_version_note : str
        Free text describing what changed in this version.
    test : bool
        True when called from the unit tests.

    Returns
    -------
    str
        Path of the log file that was written.
    """
    start_date, end_date = (list(date_range) + ["", ""])[:2]
    row = {
        "uuid": str(uuid.uuid4()),
        "timestamp": _now(),
        "tag": tag,
        "model_version": str(model_version),
        "model_version_note": model_version_note,
        "train_start_date": str(start_date),
        "train_end_date": str(end_date),
        "rmse": metrics.get("rmse", ""),
        "runtime": runtime,
        "test": test,
    }
    path = _append(_log_path("train", test), TRAIN_FIELDS, row)
    print("... train log updated ->", path)
    return path


def update_predict_log(
    tag,
    y_pred,
    y_proba,
    target_date,
    runtime,
    model_version,
    test=False,
):
    """
    Record that a prediction was served.

    Parameters
    ----------
    tag : str
        Which model produced the prediction, e.g. ``'all'``.
    y_pred : float or numpy array
        The predicted value(s).
    y_proba : float or numpy array or None
        Class probabilities when the model provides them. Our regressor does
        not, so this is normally None.
    target_date : str
        The date the prediction was made for.
    runtime : str
        How long the prediction took, formatted ``HH:MM:SS``.
    model_version : str
        Version string of the model that served the prediction.
    test : bool
        True when called from the unit tests.

    Returns
    -------
    str
        Path of the log file that was written.
    """

    def _scalar(value):
        """Turn a numpy array / list into a single number for the csv."""
        if value is None:
            return ""
        try:
            import numpy as np

            if isinstance(value, np.ndarray):
                return float(value.reshape(-1)[0]) if value.size else ""
        except ImportError:  # pragma: no cover
            pass
        if isinstance(value, (list, tuple)):
            return float(value[0]) if len(value) else ""
        try:
            return float(value)
        except (TypeError, ValueError):
            return str(value)

    row = {
        "uuid": str(uuid.uuid4()),
        "timestamp": _now(),
        "tag": tag,
        "model_version": str(model_version),
        "target_date": str(target_date),
        "y_pred": _scalar(y_pred),
        "y_proba": _scalar(y_proba),
        "runtime": runtime,
        "test": test,
    }
    path = _append(_log_path("predict", test), PREDICT_FIELDS, row)
    print("... predict log updated ->", path)
    return path


def get_log_file(filename, test=False):
    """
    Read a log file back and return its rows.

    Raises
    ------
    ValueError
        If the file does not exist, or the name tries to escape the log
        directory. The API's /logs endpoint calls this.
    """
    _ensure_log_dir()

    # Only ever allow a bare file name from the log directory - never a path.
    safe_name = os.path.basename(filename)
    if not safe_name.lower().endswith(".csv"):
        raise ValueError(
            "ERROR (get_log_file) - '{}' is not a .csv log file".format(filename)
        )

    path = os.path.join(LOG_DIR, safe_name)
    if not os.path.isfile(path):
        raise ValueError(
            "ERROR (get_log_file) - log file '{}' does not exist".format(safe_name)
        )

    fields = PREDICT_FIELDS if safe_name.startswith("predict") else TRAIN_FIELDS
    return _read_existing(path, fields)
