#!/usr/bin/env python
"""
cslib.py
========
Data ingestion library for the IBM AI Workflow capstone (AAVAIL revenue forecast).

The assignment requires that data ingestion live in a *python module* (not in a
notebook) and that it return a feature matrix ready for EDA / modelling.

The public functions are:

    fetch_data(data_dir)          -> DataFrame of raw invoice line items
    convert_to_ts(df, country)    -> daily time-series DataFrame
    fetch_ts(data_dir, clean)     -> dict of daily time-series (all + top 10 countries)
    engineer_features(df, train)  -> (X, y, dates) supervised learning matrix

The three problems this module has to solve are called out in the assignment:

  1. The JSON files do NOT use uniform column names across months.
  2. Some invoice ids contain letters that must be stripped for matching.
  3. Transactions must be aggregated by day to form a time-series.
"""

import os
import re
import shutil
import warnings

import numpy as np
import pandas as pd

# The nine columns every monthly file must expose once it has been normalised.
CORRECT_COLUMNS = sorted(
    [
        "country",
        "customer_id",
        "day",
        "invoice",
        "month",
        "price",
        "stream_id",
        "times_viewed",
        "year",
    ]
)

# The order of the columns produced by :func:`engineer_features`. Anything
# that needs to know which column is which (notebooks, the API, tests) should
# import this instead of hard-coding an index.
FEATURE_NAMES = [
    "previous_7",
    "previous_14",
    "previous_28",
    "previous_70",
    "recent_invoices",
    "recent_views",
    "previous_year",
]

# Files do not always agree on a column name, so map every known alias onto
# the canonical name. A handful of months use the "CamelCase" spelling and
# some use "total_price" instead of "price".
COLUMN_ALIASES = {
    "StreamID": "stream_id",
    "TimesViewed": "times_viewed",
    "total_price": "price",
    "TotalPrice": "price",
    "streamId": "stream_id",
    "timesViewed": "times_viewed",
}


# ---------------------------------------------------------------------------
# 1. Read the raw invoice files
# ---------------------------------------------------------------------------
def fetch_data(data_dir, extra_dirs=None):
    """
    Read every monthly json file in ``data_dir`` and return a single DataFrame.

    Parameters
    ----------
    data_dir : str
        Folder holding the monthly json files.
    extra_dirs : list of str, optional
        Further folders whose json files are appended to the same DataFrame.
        This is how the production months get the *full* history: the lag
        features for a production date have to look back past the training
        window, otherwise ``previous_year`` would be 0 for every new day.

    Steps performed here:
        * validate the input directory (does it exist / does it have files)
        * read each monthly file
        * rename any non-standard column names to the canonical names
        * validate that the columns now match expectations
        * build a real ``invoice_date`` datetime column
        * strip non-numeric characters out of the invoice ids
        * sort by date and reset the index

    Returns
    -------
    pandas.DataFrame
        One row per invoice line item, sorted by ``invoice_date``.
    """
    # ---- input validation -------------------------------------------------
    if data_dir is None:
        raise ValueError("ERROR (fetch_data) - no data directory given")
    if not os.path.isdir(data_dir):
        raise ValueError(
            "ERROR (fetch_data) - specified data dir '{}' does not exist".format(data_dir)
        )
    if not os.listdir(data_dir):
        raise ValueError(
            "ERROR (fetch_data) - specified data dir '{}' contains no files".format(data_dir)
        )

    # ---- find the json files ---------------------------------------------
    folders = [data_dir] + list(extra_dirs or [])
    for folder in folders:
        if not os.path.isdir(folder):
            raise ValueError(
                "ERROR (fetch_data) - specified data dir '{}' does not exist".format(folder)
            )
    file_list = sorted(
        os.path.join(folder, f)
        for folder in folders
        for f in os.listdir(folder)
        if f.lower().endswith(".json")
    )
    if not file_list:
        raise ValueError(
            "ERROR (fetch_data) - no .json files found in '{}'".format(data_dir)
        )

    # ---- read + normalise each file --------------------------------------
    monthly = {}
    for file_name in file_list:
        label = os.path.basename(file_name)
        try:
            df = pd.read_json(file_name)
        except ValueError as err:
            raise ValueError(
                "ERROR (fetch_data) - could not parse '{}' as json: {}".format(label, err)
            )

        if df.empty:
            print("WARNING (fetch_data) - '{}' contained no rows, skipping".format(label))
            continue

        # PROBLEM 1: normalise the column names.
        rename = {c: COLUMN_ALIASES[c] for c in df.columns if c in COLUMN_ALIASES}
        if rename:
            df = df.rename(columns=rename)

        # Anything still non-standard means we do not understand this file.
        unknown = set(df.columns) - set(CORRECT_COLUMNS)
        if unknown:
            raise ValueError(
                "ERROR (fetch_data) - '{}' has unexpected columns {}".format(
                    label, sorted(unknown)
                )
            )
        missing = set(CORRECT_COLUMNS) - set(df.columns)
        if missing:
            raise ValueError(
                "ERROR (fetch_data) - '{}' is missing columns {}".format(
                    label, sorted(missing)
                )
            )

        df = df[CORRECT_COLUMNS]
        df["source_file"] = label
        monthly[label] = df

    if not monthly:
        raise ValueError("ERROR (fetch_data) - none of the json files had any rows")

    # ---- concatenate ------------------------------------------------------
    df = pd.concat(monthly.values(), ignore_index=True, sort=False)

    # ---- build a real date column ----------------------------------------
    # year / month / day are separate integers, so combine them explicitly.
    df["invoice_date"] = pd.to_datetime(
        dict(
            year=df["year"].astype(int),
            month=df["month"].astype(int),
            day=df["day"].astype(int),
        ),
        errors="coerce",
    )
    bad_dates = int(df["invoice_date"].isna().sum())
    if bad_dates:
        print(
            "WARNING (fetch_data) - dropped {} rows with an unparseable date".format(
                bad_dates
            )
        )
        df = df[df["invoice_date"].notna()].copy()

    df["country"] = df["country"].astype(str).str.strip()
    df["price"] = pd.to_numeric(df["price"], errors="coerce").fillna(0.0)
    df["times_viewed"] = pd.to_numeric(df["times_viewed"], errors="coerce").fillna(0)

    # PROBLEM 2: some invoice ids contain letters - keep only the digits.
    df["invoice"] = df["invoice"].astype(str).map(lambda i: re.sub(r"\D+", "", i))

    # ---- final tidy -------------------------------------------------------
    df = df.sort_values(by="invoice_date", kind="mergesort").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# 2. Aggregate the line items into a daily time-series
# ---------------------------------------------------------------------------
def convert_to_ts(df_orig, country=None):
    """
    Turn raw invoice line items into a *daily* time-series.

    The full calendar is built with ``date_range`` so that days with no
    purchases appear as rows of zeros. Without this, "the next 30 days"
    window would silently cover fewer than 30 days.

    Returns a DataFrame with one row per day and the columns:
        date, purchases, unique_invoices, unique_streams, total_views,
        year_month, revenue
    """
    if country is not None:
        if country not in set(df_orig["country"].unique()):
            raise ValueError(
                "ERROR (convert_to_ts) - country '{}' not found in the data".format(country)
            )
        df = df_orig[df_orig["country"] == country]
    else:
        df = df_orig

    if df.empty:
        raise ValueError("ERROR (convert_to_ts) - no rows to aggregate")

    # Aggregate the line items to one row per day. groupby is used (rather
    # than set_index + reindex) because many rows share the same date.
    grouped = df.groupby("invoice_date")
    daily = pd.DataFrame(
        {
            # number of invoice line items that day
            "purchases": grouped["price"].count(),
            # how many distinct invoices that day
            "unique_invoices": grouped["invoice"].nunique(),
            # how many distinct streams were bought that day
            "unique_streams": grouped["stream_id"].nunique(),
            # how many times content was viewed that day
            "total_views": grouped["times_viewed"].sum(),
            # the revenue itself
            "revenue": grouped["price"].sum(),
        }
    )

    # Build a continuous daily index covering the whole observed period so
    # that days without purchases are present as rows of zeros. Without this
    # the "next 30 days" window would silently cover less than 30 days.
    full_index = pd.date_range(
        start=df["invoice_date"].min(), end=df["invoice_date"].max(), freq="D"
    )
    daily = daily.reindex(full_index).fillna(0.0)

    ts = daily.reset_index().rename(columns={"index": "date"})
    ts = ts[["date", "purchases", "unique_invoices", "unique_streams",
             "total_views", "revenue"]]
    ts["date"] = pd.to_datetime(ts["date"])
    ts["year_month"] = ts["date"].dt.strftime("%Y-%m")

    return ts


# ---------------------------------------------------------------------------
# 3. Convenience loader that caches the aggregated data as small csv files
# ---------------------------------------------------------------------------
def fetch_ts(data_dir, clean=False, top_n=10, extra_dirs=None):
    """
    Load the daily time-series for the whole business and for the ``top_n``
    countries by total revenue.

    Parameters
    ----------
    data_dir : str
        Main data folder.
    clean : bool
        Delete any existing cache and rebuild it from the json.
    top_n : int
        How many countries to keep, ranked by total revenue.
    extra_dirs : list of str, optional
        Extra folders appended to ``data_dir`` - see ``fetch_data``.

    Results are cached as csv under ``<data_dir>/ts-data`` because converting
    160MB of json takes a while and we only need to do it once per run. When
    ``extra_dirs`` is used the cache is written to ``ts-data-combined`` so it
    can never be confused with the training-only cache.

    Returns
    -------
    dict
        ``{'all': DataFrame, 'united_kingdom': DataFrame, ...}``
        Country keys are the country name lower-cased with spaces -> ``_``.
    """
    if data_dir is None or not os.path.isdir(data_dir):
        raise ValueError(
            "ERROR (fetch_ts) - data dir '{}' does not exist".format(data_dir)
        )

    ts_data_dir = os.path.join(
        data_dir, "ts-data-combined" if extra_dirs else "ts-data"
    )
    if clean and os.path.isdir(ts_data_dir):
        shutil.rmtree(ts_data_dir)
    if not os.path.isdir(ts_data_dir):
        os.makedirs(ts_data_dir, exist_ok=True)

    # ---- use the cache when it is already populated -----------------------
    cached = sorted(f for f in os.listdir(ts_data_dir) if f.endswith(".csv"))
    if cached:
        print("... loading cached time-series data from '{}'".format(ts_data_dir))
        result = {}
        for f in cached:
            key = os.path.splitext(f)[0][3:]  # strip the "ts-" prefix
            result[key] = pd.read_csv(os.path.join(ts_data_dir, f))
            result[key]["date"] = pd.to_datetime(result[key]["date"])
        return result

    # ---- build from the raw json ------------------------------------------
    print("... processing the raw json files (this takes a minute)")
    df = fetch_data(data_dir, extra_dirs=extra_dirs)

    # Which countries matter most? Rank by total revenue.
    revenue_by_country = (
        df.groupby("country")["price"].sum().sort_values(ascending=False)
    )
    top_countries = list(revenue_by_country.index[:top_n])

    result = {"all": convert_to_ts(df)}
    for country in top_countries:
        key = re.sub(r"\s+", "_", country.strip().lower())
        result[key] = convert_to_ts(df, country=country)

    # ---- persist the cache -------------------------------------------------
    for key, item in result.items():
        item.to_csv(os.path.join(ts_data_dir, "ts-{}.csv".format(key)), index=False)

    return result


# ---------------------------------------------------------------------------
# 4. Turn the daily time-series into a supervised learning problem
# ---------------------------------------------------------------------------
def engineer_features(df, training=True):
    """
    Build the feature matrix and the regression target.

    Business question
    -----------------
    "Given everything we know up to and including day *D*, what will AAVAIL
     earn over the next 30 days?"

    So for every day *D*:

        y(D) = revenue summed over [D, D+30)

    and the features describe the days *before* D (leakage check: no feature
    may use revenue from day D onwards, and ``y`` is the only thing allowed
    to).

    Features
    --------
    previous_7    revenue over the 7 days before D
    previous_14   revenue over the 14 days before D
    previous_28   revenue over the 28 days before D
    previous_70   revenue over the 70 days before D
    previous_year revenue over the same 30-day window one year before D
    recent_invoices  mean daily unique invoices over the 30 days before D
    recent_views     mean daily total views over the 30 days before D

    Parameters
    ----------
    df : DataFrame
        Daily time-series as produced by :func:`convert_to_ts`.
    training : bool
        When True (default) the final 30 rows are dropped because their
        target would be built from a partially observed future. Set to False
        when you need the full range, e.g. when serving predictions.

    Returns
    -------
    (X, y, dates) : tuple of numpy arrays
    """
    # Work on a clean, date-sorted copy with no gaps.
    df = df.sort_values(by="date").reset_index(drop=True).copy()
    df["date"] = pd.to_datetime(df["date"])

    # Collapse duplicate days (in case a partially filtered frame is passed in)
    # and then make sure every calendar day is present.
    if df["date"].duplicated().any():
        df = df.groupby("date", as_index=False).sum(numeric_only=True)
        df["date"] = pd.to_datetime(df["date"])

    full_index = pd.date_range(start=df["date"].min(), end=df["date"].max(), freq="D")
    full_index.name = "date"
    df = df.set_index("date").reindex(full_index).fillna(0.0).reset_index()

    revenue = df["revenue"].astype(float)
    n = revenue.size

    # A 30 day rolling sum, reused for both the target and the yearly feature.
    roll_30 = revenue.rolling(30, min_periods=1).sum()

    features = pd.DataFrame(index=df.index)

    # Revenue over the N days STRICTLY BEFORE the current day.
    # shift(1) moves the series down by one so the window ends at D-1.
    for window in (7, 14, 28, 70):
        features["previous_{}".format(window)] = (
            revenue.shift(1).rolling(window, min_periods=1).sum()
        )

    # Mean daily unique invoices / views over the 30 days before D.
    features["recent_invoices"] = (
        df["unique_invoices"].shift(1).rolling(30, min_periods=1).mean()
    )
    features["recent_views"] = (
        df["total_views"].shift(1).rolling(30, min_periods=1).mean()
    )

    # TARGET: sum of revenue over [D, D+30). roll_30 at D+29 gives exactly that.
    y_series = roll_30.shift(-29)

    # The same 30-day window, but one year earlier: [D-365, D-335).
    # roll_30[i-336] = sum(revenue[i-365 .. i-336]), so a POSITIVE shift by
    # 336 lands it on row i. (A negative shift would look into the future,
    # which would leak the target into the features.)
    # Where the series is too short to look a year back - e.g. the 5 month
    # production set - this is NaN; treat that as 0, matching the reference
    # behaviour of "no data a year ago == no revenue".
    features["previous_year"] = roll_30.shift(336).fillna(0.0)

    # ---- drop rows we cannot trust ----------------------------------------
    # Re-order so the matrix columns always match FEATURE_NAMES.
    features = features[FEATURE_NAMES]
    features = features.replace([np.inf, -np.inf], np.nan)
    usable = features.notna().all(axis=1) & y_series.notna()
    # A row whose features are all zero carries no signal.
    usable &= features.fillna(0.0).abs().sum(axis=1) > 0

    X = features[usable].reset_index(drop=True)
    y = y_series[usable].to_numpy(dtype=float).reshape(-1)
    dates = df.loc[usable.to_numpy(), "date"].to_numpy()

    # The last 30 rows have an incomplete target window - only drop them when
    # we are building a training set.
    if training and X.shape[0] > 30:
        X = X.iloc[:-30].reset_index(drop=True)
        y = y[:-30]
        dates = dates[:-30]

    if X.shape[0] == 0:
        raise ValueError("ERROR (engineer_features) - no usable rows were produced")

    return X.to_numpy(dtype=float), y, dates


# ---------------------------------------------------------------------------
# Small self-check so the module can be run directly during development
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import time

    data_dir = os.path.join("data", "cs-train")
    print("... fetching data from '{}'".format(data_dir))

    started = time.time()
    ts_all = fetch_ts(data_dir, clean=False)
    print("... done in {:.1f}s".format(time.time() - started))

    for key, item in ts_all.items():
        X, y, _ = engineer_features(item)
        print(
            "{:<20} daily rows={:<6} X={:<6} y={:<6} target mean={:,.0f}".format(
                key, item.shape[0], X.shape[0], y.shape[0], y.mean()
            )
        )
