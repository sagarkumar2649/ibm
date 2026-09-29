#!/usr/bin/env python
"""
model.py
========
Training, comparison, loading and prediction for the AAVAIL 30-day revenue
forecast. This is the module the Flask API calls.

Design notes
------------
* One model is trained per country key returned by :func:`cslib.fetch_ts`, i.e.
  ``'all'`` plus the top ten countries. Small countries simply do not get a
  model, and that is reported rather than silently ignored.
* The default estimator is a ``RandomForestRegressor`` inside a scikit-learn
  ``Pipeline`` with a ``StandardScaler``, selected by grid search on RMSE. That
  is the configuration the course reference solution uses, and Part 2 of the
  case study confirms it is the best of the candidates on this data.
* ``MODEL_VERSION`` must be bumped whenever the training logic changes, so the
  predict log records which model produced a number.

Run it directly for a quick end-to-end check::

    python model.py
"""

import argparse
import os
import re
import sys
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from cslib import FEATURE_NAMES, engineer_features, fetch_ts
from logger import update_predict_log, update_train_log

# Bump the version whenever the training logic changes.
MODEL_DIR = "models"
MODEL_VERSION = "1.0"
MODEL_VERSION_NOTE = (
    "ExtraTreesRegressor + StandardScaler pipeline, RMSE grid search, all 7 features. "
    "Selected in Part 2: lowest mean RMSE and best in every country."
)

# Which columns actually go into the model. Part 2 showed the two engagement
# features are the strongest predictors on this data, so all seven are used.
FEATURE_COLUMNS = list(FEATURE_NAMES)

# Rows smaller than this are not worth fitting an ensemble on.
MIN_TRAIN_ROWS = 30

# Single-process on purpose. Training takes about a second per country on this
# data, and joblib's parallel backend is unreliable inside a Jupyter kernel on
# Windows (it fails with BrokenProcessPool). Set to -1 to use every core.
N_JOBS = 1

# Only these are trained when test=True so the unit tests stay fast.
TEST_TAGS = ("all", "united_kingdom")


def _project_root():
    return os.path.dirname(os.path.abspath(__file__))


def _format_runtime(seconds):
    """Seconds -> 'HH:MM:SS'."""
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return "{:02d}:{:02d}:{:02d}".format(h, m, s)


def _time_cv(X):
    """
    The cross-validation splitter used for every hyper-parameter search.

    TimeSeriesSplit is used rather than a shuffled/random split on purpose.
    The rows are a daily time-series, so a random split leaks the future into
    the training folds: the trees then only have to interpolate inside the
    range they already saw, which flatters the score and hides how badly they
    extrapolate. TimeSeriesSplit always trains on the past and validates on the
    future, which is exactly the job the deployed model has to do.
    """
    n_splits = min(4, max(2, X.shape[0] // 25))
    return TimeSeriesSplit(n_splits=n_splits)


def build_model():
    """
    Build the estimator and the grid of hyper-parameters to search.

    Kept in its own function so the notebook and the tests can reuse exactly
    the same model definition. Part 2 found ExtraTreesRegressor beat every other
    candidate in all eleven country models, so that is what ships.
    """
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("et", ExtraTreesRegressor(random_state=42, n_jobs=N_JOBS)),
        ]
    )
    # The search is deliberately small and biased towards *heavily regularised*
    # trees. An earlier grid offered only 10-30 trees and left
    # min_samples_leaf at its default of 1, which overfitted badly: the search
    # happily picked a model whose CV RMSE was ~10x worse than the untuned
    # default, and the production numbers fell apart. min_samples_leaf is the
    # knob that matters here, so it is the one we actually search over.
    param_grid = {
        "et__n_estimators": [100, 200],
        "et__max_features": [0.5, 1.0],
        "et__min_samples_leaf": [5, 20, 40],
    }
    return pipeline, param_grid


def evaluate_models(X, y, random_state=42):
    """
    Part 2 deliverable: compare a handful of candidate regressors on the same
    feature matrix and report RMSE for each.

    The candidates deliberately span the families the course covers -
    a linear baseline, a regularised linear model, a bagged tree ensemble, a
    boosted tree ensemble and an extremely randomised forest.

    Returns
    -------
    (results, fitted)
        ``results``  - DataFrame of RMSE per model, best first.
        ``fitted``   - dict of model name -> fitted estimator (for inspection).
    """
    from sklearn.ensemble import (
        ExtraTreesRegressor,
        GradientBoostingRegressor,
        RandomForestRegressor,
    )
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    candidates = {
        "linear_regression": Pipeline(
            [("scaler", StandardScaler()), ("m", LinearRegression())]
        ),
        "ridge": Pipeline([("scaler", StandardScaler()), ("m", Ridge(alpha=1.0))]),
        "random_forest": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("m", RandomForestRegressor(random_state=random_state, n_jobs=N_JOBS)),
            ]
        ),
        "extra_trees": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("m", ExtraTreesRegressor(random_state=random_state, n_jobs=N_JOBS)),
            ]
        ),
        "gradient_boosting": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "m",
                    GradientBoostingRegressor(random_state=random_state),
                ),
            ]
        ),
    }

    # Time-ordered holdout, for the same reason as _time_cv(): a shuffled split
    # would leak the future into training and overstate every candidate.
    split = int(X.shape[0] * 0.75)
    X_train, X_test, y_train, y_test = X[:split], X[split:], y[:split], y[split:]

    rows, fitted = [], {}
    for name, est in candidates.items():
        started = time.time()
        est.fit(X_train, y_train)
        pred = est.predict(X_test)
        rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
        rows.append(
            {
                "model": name,
                "rmse": rmse,
                "runtime": _format_runtime(time.time() - started),
            }
        )
        fitted[name] = est
        print("    {:<22} RMSE = {:,.2f}".format(name, rmse))

    results = pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True)
    return results, fitted


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def _train_one(df, tag, test=False):
    """
    Train and persist a single model for one country key.

    Returns the evaluation metrics, or None if the data was too thin to train on.
    """
    X, y, dates = engineer_features(df, training=True)

    if X.shape[0] < MIN_TRAIN_ROWS:
        print(
            "    ... skipping '{}': only {} usable training rows "
            "(need {})".format(tag, X.shape[0], MIN_TRAIN_ROWS)
        )
        return None

    started = time.time()

    # Hold out the MOST RECENT slice as a time-ordered holdout, then re-train
    # on everything. A shuffled split would be much more flattering, because a
    # tree can interpolate inside the range it was trained on - but in
    # production we always have to forecast forward, so the holdout has to
    # respect time to be an honest estimate.
    split = int(X.shape[0] * 0.75)
    X_train, X_test, y_train, y_test = X[:split], X[split:], y[:split], y[split:]

    pipeline, param_grid = build_model()
    grid = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        cv=_time_cv(X_train),
        n_jobs=N_JOBS,
        scoring="neg_mean_squared_error",
    )
    grid.fit(X_train, y_train)

    y_pred = grid.predict(X_test)
    eval_rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    runtime = _format_runtime(time.time() - started)

    print(
        "    ... '{}': {} rows, time-ordered holdout RMSE = {:,.2f}, best params = {}".format(
            tag,
            X.shape[0],
            eval_rmse,
            grid.best_params_,
        )
    )

    # Re-train on ALL the data with the winning hyper-parameters, so the model
    # that ships has seen everything available.
    print("    ... re-training '{}' on all {} rows".format(tag, X.shape[0]))
    pipeline, param_grid = build_model()
    final = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        cv=_time_cv(X),
        n_jobs=N_JOBS,
        scoring="neg_mean_squared_error",
    )
    final.fit(X, y)

    if not os.path.isdir(MODEL_DIR):
        os.makedirs(MODEL_DIR, exist_ok=True)

    version = str(MODEL_VERSION).replace(".", "_")
    prefix = "test" if test else "sl"
    saved_path = os.path.join(MODEL_DIR, "{}-{}-{}.joblib".format(prefix, tag, version))
    joblib.dump(final, saved_path)
    print("    ... saved ->", saved_path)

    update_train_log(
        tag,
        (str(pd.Timestamp(dates[0]).date()), str(pd.Timestamp(dates[-1]).date())),
        {"rmse": round(eval_rmse, 2)},
        runtime,
        MODEL_VERSION,
        MODEL_VERSION_NOTE,
        test=test,
    )

    return {"rmse": eval_rmse, "rows": int(X.shape[0]), "path": saved_path}


def model_train(data_dir=None, test=False, tags=None):
    """
    Train a model for every country in ``data_dir``.

    ``data_dir`` defaults to ``data/cs-train``. When ``test`` is True only
    ``all`` and ``united_kingdom`` are trained, which keeps the unit tests quick.
    """
    if data_dir is None:
        data_dir = os.path.join(_project_root(), "data", "cs-train")

    if not os.path.isdir(MODEL_DIR):
        os.makedirs(MODEL_DIR, exist_ok=True)

    print("... loading time-series data from '{}'".format(data_dir))
    ts_data = fetch_ts(data_dir)
    print("... found {} country keys: {}".format(len(ts_data), ", ".join(sorted(ts_data))))

    if tags is None:
        tags = TEST_TAGS if test else None

    results = {}
    for tag, frame in sorted(ts_data.items()):
        if tags is not None and tag not in tags:
            continue
        print("\n  training '{}' ...".format(tag))
        outcome = _train_one(frame, tag, test=test)
        if outcome is not None:
            results[tag] = outcome

    print("\n... training complete for {} models".format(len(results)))
    return results


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def _model_files(prefix="sl"):
    """List the saved model files, newest last."""
    if not os.path.isdir(MODEL_DIR):
        return []
    return sorted(
        f
        for f in os.listdir(MODEL_DIR)
        if f.endswith(".joblib") and prefix in f
    )


def model_load(prefix="sl", data_dir=None, training=True, load_data=True, extra_dirs=None):
    """
    Load the saved models and, optionally, the feature rows needed to serve them.

    ``extra_dirs`` is forwarded to :func:`cslib.fetch_ts` so the production
    months can be resolved against the full history.

    Returns
    -------
    (all_models, all_data)
        ``all_models`` maps country key -> fitted estimator.
        ``all_data``   maps country key -> {'X': DataFrame, 'dates': array}.
        ``all_data`` is empty when ``load_data`` is False.
    """
    if data_dir is None:
        data_dir = os.path.join(_project_root(), "data", "cs-train")

    files = _model_files(prefix)
    if not files:
        raise ValueError(
            "ERROR (model_load) - no models with prefix '{}' found in '{}'. "
            "Train the model first.".format(prefix, MODEL_DIR)
        )

    all_models = {}
    for f in files:
        # sl-<tag>-<version>.joblib  ->  <tag>
        parts = os.path.splitext(f)[0].split("-")
        if len(parts) < 3:
            continue
        tag = parts[1]
        all_models[tag] = joblib.load(os.path.join(MODEL_DIR, f))

    print("... loaded {} models: {}".format(len(all_models), ", ".join(sorted(all_models))))

    all_data = {}
    if load_data:
        ts_data = fetch_ts(data_dir, extra_dirs=extra_dirs)
        for tag in all_models:
            if tag not in ts_data:
                # A model was trained for a country that the current data
                # directory does not contain - it can still be served, we just
                # cannot look the query date up in it.
                continue
            try:
                X, _, dates = engineer_features(ts_data[tag], training=training)
            except ValueError:
                # Too little history in THIS data directory to build features
                # (e.g. a country that only has a few weeks of production
                # data). Skip it instead of failing every other country.
                print("... skipping '{}' - not enough history in this data".format(tag))
                continue
            all_data[tag] = {
                "X": pd.DataFrame(X, columns=FEATURE_COLUMNS),
                "dates": np.array([str(pd.Timestamp(d).date()) for d in dates]),
            }

    return all_models, all_data


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
def model_predict(
    country,
    year,
    month,
    day,
    all_models=None,
    all_data=None,
    test=False,
    log=True,
):
    """
    Predict the 30-day revenue for one country starting on the given date.

    The date is looked up in the feature matrix built from the data, and that
    single row is fed to the model. This is what lets leadership query any past
    date they like, not just the most recent one.

    Returns
    -------
    dict
        ``{'y_pred': float, 'y_proba': None, 'tag': str, 'target_date': str}``
    """
    started = time.time()

    if all_models is None:
        all_models, all_data = model_load(training=False, load_data=True)

    # ---- input checks -----------------------------------------------------
    if not isinstance(country, str) or not country.strip():
        raise ValueError("ERROR (model_predict) - a country name is required")

    tag = country.strip()
    if tag not in all_models:
        raise ValueError(
            "ERROR (model_predict) - no model for country '{}'. Available: {}".format(
                tag, ", ".join(sorted(all_models))
            )
        )

    for name, value in (("year", year), ("month", month), ("day", day)):
        if value is None or str(value).strip() == "":
            raise ValueError("ERROR (model_predict) - {} is required".format(name))
        if not re.match(r"^\d+$", str(value).strip()):
            raise ValueError(
                "ERROR (model_predict) - {} must be a number, got '{}'".format(
                    name, value
                )
            )

    # Build the query date, zero padding month and day.
    try:
        target_date = "{}-{}-{}".format(
            int(year), str(int(month)).zfill(2), str(int(day)).zfill(2)
        )
        # Reject impossible dates such as 2019-02-31.
        pd.Timestamp(target_date)
    except Exception:
        raise ValueError(
            "ERROR (model_predict) - '{}' is not a valid date".format(
                "-".join([str(year), str(month), str(day)])
            )
        )

    # ---- find the matching feature row ------------------------------------
    if all_data is None or tag not in all_data:
        raise ValueError(
            "ERROR (model_predict) - no feature data available for '{}'. "
            "The model exists but the date cannot be resolved.".format(tag)
        )

    data = all_data[tag]
    dates = data["dates"]
    if len(dates) != len(data["X"]):
        raise ValueError(
            "ERROR (model_predict) - dimension mismatch for '{}': "
            "{} dates vs {} feature rows".format(tag, len(dates), len(data["X"]))
        )

    matches = np.where(dates == target_date)[0]
    if len(matches) == 0:
        raise ValueError(
            "ERROR (model_predict) - date {} is not in range {}-{} for '{}'".format(
                target_date, dates[0], dates[-1], tag
            )
        )

    query = data["X"].iloc[[matches[0]]]
    model = all_models[tag]

    y_pred = float(np.asarray(model.predict(query)).reshape(-1)[0])
    y_proba = None
    runtime = _format_runtime(time.time() - started)

    if log:
        update_predict_log(
            tag,
            y_pred,
            y_proba,
            target_date,
            runtime,
            MODEL_VERSION,
            test=test,
        )

    return {
        "y_pred": y_pred,
        "y_proba": y_proba,
        "tag": tag,
        "target_date": target_date,
        "model_version": MODEL_VERSION,
        "runtime": runtime,
    }


# ---------------------------------------------------------------------------
# Command line entry point
# ---------------------------------------------------------------------------
def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Train and serve the AAVAIL 30-day revenue forecast."
    )
    parser.add_argument(
        "-d", "--data-dir", default=None, help="directory holding the invoice json files"
    )
    parser.add_argument(
        "-t",
        "--test",
        action="store_true",
        help="train only 'all' and 'united_kingdom' and use the test log/model files",
    )
    parser.add_argument(
        "-c",
        "--countries",
        default=None,
        help="comma separated list of countries to predict, e.g. all,united_kingdom",
    )
    parser.add_argument(
        "-p", "--predict", action="store_true", help="run a prediction instead of training"
    )
    parser.add_argument("--year", default=None, help="target year for --predict")
    parser.add_argument("--month", default=None, help="target month for --predict")
    parser.add_argument("--day", default=None, help="target day for --predict")
    args = parser.parse_args(argv)

    os.chdir(_project_root())

    if not args.predict:
        model_train(data_dir=args.data_dir, test=args.test)
        return 0

    # Prediction mode.
    if not (args.countries and args.year and args.month and args.day):
        parser.error("--predict needs --countries and --year/--month/--day")

    prefix = "test" if args.test else "sl"
    all_models, all_data = model_load(prefix=prefix, training=False, load_data=True)
    for country in [c.strip() for c in args.countries.split(",") if c.strip()]:
        result = model_predict(
            country, args.year, args.month, args.day,
            all_models=all_models, all_data=all_data, test=args.test,
        )
        print(
            "{} on {}: predicted 30-day revenue = {:,.2f}  (model v{})".format(
                country,
                result["target_date"],
                result["y_pred"],
                result["model_version"],
            )
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
