#!/usr/bin/env python
"""
app.py
======
Flask API for the AAVAIL 30-day revenue forecast.

Endpoints
---------
GET  /            index page listing the endpoints
POST /train       train the models from a directory of invoice json files
POST /predict     get the 30-day revenue forecast for a country and date
GET  /logs        read back a train/predict log file

Run it locally::

    python app.py            # production models, port 8080
    python app.py -d         # debug mode
    python app.py -t         # use the test models and test logs
"""

import argparse
import os
import sys

from flask import Flask, jsonify, request

import model as model_module
from cslib import fetch_ts
from logger import get_log_file

app = Flask(__name__)

# Set at startup from the command line: 'test' uses the test models and the
# test log files so running the app can never disturb production artifacts.
MODEL_MODE = "prod"

# Memoises the loaded models + feature rows so repeated predictions are fast.
_LOAD_CACHE = {}


@app.route("/", methods=["GET"])
def index():
    """A tiny index so hitting the root in a browser is not a 404."""
    return jsonify(
        {
            "service": "AAVAIL 30-day revenue forecast",
            "model_version": model_module.MODEL_VERSION,
            "model_mode": MODEL_MODE,
            "endpoints": {
                "POST /train": "body: {'data_dir': <path>, 'mode': 'test'|'prod'}",
                "POST /predict": "body: {'query': {'country','year','month','day'}, 'mode': 'test'|'prod'}",
                "GET  /logs": "args: ?filename=predict.csv",
            },
        }
    )


@app.route("/train", methods=["POST"])
def train():
    """
    Train the models.

    Request body (all optional):
        data_dir : directory of invoice json files
                   (default: data/cs-train)
        mode     : 'test' or 'prod'. 'test' trains only two countries and
                   writes to the test model/log files.
    """
    body = request.get_json(silent=True) or {}
    data_dir = body.get("data_dir") or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "data", "cs-train"
    )
    mode = body.get("mode", MODEL_MODE)
    if mode not in ("test", "prod"):
        return jsonify({"error": "mode must be 'test' or 'prod'"}), 400

    try:
        results = model_module.model_train(
            data_dir=data_dir, test=(mode == "test")
        )
    except ValueError as err:
        return jsonify({"error": str(err)}), 400
    except Exception as err:  # pragma: no cover - unexpected failure
        app.logger.exception("training failed")
        return jsonify({"error": "training failed: {}".format(err)}), 500

    # The models on disk just changed, so the memoised copies are stale.
    _clear_load_cache()

    return jsonify(
        {
            "status": "ok",
            "mode": mode,
            "data_dir": data_dir,
            "models_trained": sorted(results),
            "rmse": {tag: round(r["rmse"], 2) for tag, r in results.items()},
        }
    )


def _load_cached(prefix, data_dir, extra_dirs):
    """
    Load the models and their feature rows once and reuse them.

    Rebuilding the feature matrix for every country on every request is slow
    enough to hurt, and the answer only changes when someone retrains or the
    underlying data changes - so the result is memoised on the key.
    """
    key = (prefix, os.path.abspath(data_dir), tuple(extra_dirs or ()))
    if key not in _LOAD_CACHE:
        _LOAD_CACHE[key] = model_module.model_load(
            prefix=prefix, data_dir=data_dir, training=False, load_data=True,
            extra_dirs=extra_dirs,
        )
    return _LOAD_CACHE[key]


def _clear_load_cache():
    """Forget the memoised models, e.g. after a retrain."""
    _LOAD_CACHE.clear()


@app.route("/predict", methods=["POST"])
def predict():
    """
    Predict the 30-day revenue for one country on one date.

    Request body:
        query : dict with 'country', 'year', 'month', 'day' (all strings)
        mode  : 'test' or 'prod'
        data_dir : optional directory the query date is resolved against
                   (default: data/cs-train). Pass data/cs-production when you
                   want to serve the newest months.
    """
    body = request.get_json(silent=True) or {}
    query = body.get("query")
    mode = body.get("mode", MODEL_MODE)

    if mode not in ("test", "prod"):
        return jsonify({"error": "mode must be 'test' or 'prod'"}), 400
    if not isinstance(query, dict):
        return jsonify({"error": "body must contain a 'query' object"}), 400

    for field in ("country", "year", "month", "day"):
        if field not in query:
            return jsonify({"error": "query is missing '{}'".format(field)}), 400

    data_dir = body.get("data_dir") or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "data", "cs-train"
    )
    # Serving the production months needs the training history too, otherwise the
    # lag features for a new day have nothing to look back at.
    extra_dirs = body.get("extra_dirs")
    if extra_dirs is None:
        production_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "data", "cs-production"
        )
        extra_dirs = [production_dir] if os.path.isdir(production_dir) else None

    prefix = "test" if mode == "test" else "sl"
    try:
        all_models, all_data = _load_cached(prefix, data_dir, extra_dirs)
        result = model_module.model_predict(
            query["country"],
            query["year"],
            query["month"],
            query["day"],
            all_models=all_models,
            all_data=all_data,
            test=(mode == "test"),
        )
    except ValueError as err:
        return jsonify({"error": str(err)}), 400
    except Exception as err:  # pragma: no cover - unexpected failure
        app.logger.exception("prediction failed")
        return jsonify({"error": "prediction failed: {}".format(err)}), 500

    return jsonify(
        {"status": "ok", "mode": mode, "data_dir": data_dir,
         "extra_dirs": extra_dirs, "result": result}
    )


@app.route("/logs", methods=["GET"])
def logs():
    """
    Read a log file back.

    Query args:
        filename : the csv to read, e.g. 'train.csv' or 'predict.csv'
    """
    filename = request.args.get("filename")
    if not filename:
        return jsonify({"error": "a 'filename' query argument is required"}), 400
    try:
        rows = get_log_file(filename)
    except ValueError as err:
        return jsonify({"error": str(err)}), 400
    return jsonify({"status": "ok", "filename": filename, "rows": rows})


def main(argv=None):
    parser = argparse.ArgumentParser(description="Serve the revenue forecast API.")
    parser.add_argument("-d", "--debug", action="store_true", help="run in debug mode")
    parser.add_argument(
        "-t", "--test", action="store_true", help="use the test models and test logs"
    )
    parser.add_argument("-p", "--port", type=int, default=8080, help="port to listen on")
    args = parser.parse_args(argv)

    global MODEL_MODE
    MODEL_MODE = "test" if args.test else "prod"

    # Always run from the project root so the relative data/log/models paths work.
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    print("... serving in '{}' mode on port {}".format(MODEL_MODE, args.port))
    app.run(host="0.0.0.0", port=args.port, debug=args.debug)
    return 0


if __name__ == "__main__":
    sys.exit(main())
