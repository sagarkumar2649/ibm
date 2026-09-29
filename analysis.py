#!/usr/bin/env python
"""
analysis.py
===========
Post-production analysis: query the running API, then compare the served
predictions against the known actual revenue.

This is the Part 3 deliverable "create a post-production analysis script that
investigates the relationship between model performance and the business
metric".

How it works
------------
1. Build the feature matrix from the *production* data (``data/cs-production``).
2. Pick a run of dates and ask the API for a prediction for each one, one day at
   a time, as if we had never seen the future. Those predictions are written to
   the predict log by the API itself.
3. Compare each prediction to the revenue that was actually collected over the
   following 30 days.
4. Write the comparison to csv and draw the monitoring plots.

Because we already know the true revenue for the production window, we can score
the model after the fact - that is the feedback loop the assignment asks for.

Usage
-----
Start the API first, then::

    python analysis.py
    python analysis.py --mode test       # if the API is running in test mode
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import requests

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from cslib import engineer_features, fetch_ts

import matplotlib
matplotlib.use("Agg")  # so the script works on a headless server
import matplotlib.pyplot as plt


def query_api(url, country, date, mode, data_dir, extra_dirs):
    """
    Ask the API for one prediction.

    ``data_dir`` + ``extra_dirs`` together give the API the full history, so the
    production months are resolved against real lag values instead of zeros.

    Returns (prediction, error_message). Never raises - a failed call is
    recorded as an error so the analysis can carry on.
    """
    payload = {
        "query": {
            "country": country,
            "year": date[0],
            "month": date[1],
            "day": date[2],
        },
        "mode": mode,
        "data_dir": data_dir,
        "extra_dirs": extra_dirs,
    }
    try:
        response = requests.post(url + "/predict", json=payload, timeout=60)
        response.raise_for_status()
        return float(response.json()["result"]["y_pred"]), None
    except Exception as err:
        return None, str(err)


def run_against_api(base_url, country, mode, limit=40):
    """
    Walk forward through the production dates, predicting one day at a time.

    Returns a DataFrame with the date, the served prediction and any error.
    """
    data_dir = os.path.join(PROJECT_ROOT, "data", "cs-train")
    production_dir = os.path.join(PROJECT_ROOT, "data", "cs-production")
    extra_dirs = [production_dir]

    # The API is asked for the *combined* history: the training months provide
    # the look-back that the production months need.
    print("... loading data from '{}' + '{}'".format(data_dir, production_dir))
    ts = fetch_ts(data_dir, extra_dirs=extra_dirs)

    if country not in ts:
        raise ValueError(
            "ERROR - country '{}' is not in the data. Available: {}".format(
                country, ", ".join(sorted(ts))
            )
        )

    # training=False keeps the last 30 days, so every row has a complete
    # 30-day revenue window behind it to score the prediction against.
    _, y_true, dates = engineer_features(ts[country], training=False)

    # Only score the days that the production data covers.
    production_start = fetch_ts(production_dir)[country]["date"].min()
    keep = [i for i, d in enumerate(dates) if pd.Timestamp(d) >= production_start]
    dates = [dates[i] for i in keep]
    y_true = y_true[keep]

    if limit and len(dates) > limit:
        # Use the most recent `limit` dates.
        dates, y_true = dates[-limit:], y_true[-limit:]

    rows = []
    print("... querying {} dates from the API at {}".format(len(dates), base_url))
    for i, stamp in enumerate(dates, 1):
        ymd = (str(stamp)[0:4], str(stamp)[5:7], str(stamp)[8:10])
        y_pred, err = query_api(base_url, country, ymd, mode, data_dir, extra_dirs)
        if err:
            print("  [{}/{}] {} -> API ERROR: {}".format(i, len(dates), "-".join(ymd), err))
        rows.append(
            {
                "date": str(stamp)[:10],
                "country": country,
                "y_true": float(y_true[i - 1]),
                "y_pred": y_pred,
            }
        )

    return pd.DataFrame(rows)


def score(comparison):
    """
    Score the predictions against the known values and print the result.

    The business metric *is* the 30-day revenue total, so RMSE is reported in the
    same units as the number leadership actually plans with. MAE and R^2 are
    included as supporting context.
    """
    valid = comparison.dropna(subset=["y_pred"]).copy()
    if valid.empty:
        print("\nNo successful predictions to score.")
        return None

    y_true = valid["y_true"].to_numpy(dtype=float)
    y_pred = valid["y_pred"].to_numpy(dtype=float)
    error = y_pred - y_true

    rmse = float(np.sqrt(np.mean(error ** 2)))
    mae = float(np.mean(np.abs(error)))
    # R^2 can be undefined when the actuals are constant, so guard it.
    denom = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = float(1 - np.sum(error ** 2) / denom) if denom > 0 else float("nan")

    # The baseline we have to beat: always predicting the historical mean.
    baseline_rmse = float(np.sqrt(np.mean((y_true.mean() - y_true) ** 2)))

    print("\n" + "=" * 66)
    print("post-production score for '{}' over {} days".format(
        valid["country"].iloc[0], len(valid)))
    print("=" * 66)
    print("  RMSE                  {:>14,.2f}".format(rmse))
    print("  MAE                   {:>14,.2f}".format(mae))
    print("  R^2                   {:>14.3f}".format(r2))
    print("  mean actual revenue   {:>14,.2f}".format(y_true.mean()))
    print("  baseline RMSE (mean)  {:>14,.2f}".format(baseline_rmse))
    print("  model vs baseline     {:>14,.1f}%".format(
        100.0 * (baseline_rmse - rmse) / baseline_rmse
        if baseline_rmse else float("nan")
    ))
    print("=" * 66)

    if baseline_rmse and rmse >= baseline_rmse:
        print("WARNING: the model is not beating 'always predict the historical mean'.")

    valid["error"] = error
    valid["abs_error"] = np.abs(error)
    return valid


def plot_results(scored, out_dir):
    """Draw the two monitoring plots the assignment asks for."""
    os.makedirs(out_dir, exist_ok=True)
    country = scored["country"].iloc[0]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(scored["date"], scored["y_true"], marker="o", ms=3,
            color="darkorange", label="actual 30-day revenue")
    ax.plot(scored["date"], scored["y_pred"], marker="s", ms=3,
            color="royalblue", linestyle="--", label="predicted (served by the API)")
    ax.set_title("Predicted vs actual 30-day revenue - '{}'".format(country))
    ax.set_xlabel("Date the 30-day window starts")
    ax.set_ylabel("Revenue")
    ax.legend()
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    path1 = os.path.join(out_dir, "predicted_vs_actual.png")
    plt.savefig(path1, dpi=120)
    plt.close()
    print("... wrote", path1)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(scored["date"], scored["error"], color="seagreen", width=0.7)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Prediction error over time - '{}'".format(country))
    ax.set_xlabel("Date the 30-day window starts")
    ax.set_ylabel("predicted - actual")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    path2 = os.path.join(out_dir, "error_over_time.png")
    plt.savefig(path2, dpi=120)
    plt.close()
    print("... wrote", path2)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Query the API and compare its predictions to the known values."
    )
    parser.add_argument("--url", default="http://localhost:8080",
                        help="base URL of the running API")
    parser.add_argument("--country", default="all", help="country key to analyse")
    parser.add_argument("--mode", default="prod", choices=("test", "prod"),
                        help="mode the API is running in")
    parser.add_argument("--limit", type=int, default=40,
                        help="how many dates to check (0 = all)")
    args = parser.parse_args(argv)

    comparison = run_against_api(
        args.url.rstrip("/"), args.country, args.mode, limit=args.limit
    )

    out_dir = os.path.join(PROJECT_ROOT, "data")
    comparison.to_csv(os.path.join(out_dir, "production_comparison.csv"), index=False)
    print("\n... saved -> data/production_comparison.csv")

    scored = score(comparison)
    if scored is not None:
        scored.to_csv(
            os.path.join(out_dir, "production_comparison.csv"), index=False
        )
        plot_results(scored, out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
