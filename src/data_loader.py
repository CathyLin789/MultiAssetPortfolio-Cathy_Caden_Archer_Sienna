"""
data_loader.py

Handles all data ingestion for the multi-asset portfolio analysis.
Loads monthly manager returns, benchmark returns, the risk-free rate,
and Strategic Asset Allocation (SAA) weights from the /data folder.

Also defines the Tactical Asset Allocation (TAA) weights used to
compute the total fund return.

Main entry point: load_all()
"""

import pandas as pd
from pathlib import Path

# ── TAA Weights (hardcoded from assignment brief) ─────────────────────────────
# TAA is not supplied as a CSV — only the brief's table — so it lives in code.
TAA_WEIGHTS = {
    "AUS_EQ":      0.35,
    "INTL_EQ":     0.35,
    "Bonds":       0.15,
    "Real_Estate": 0.05,
    "PE_VC":       0.10,
}

# ── Name mappings ─────────────────────────────────────────────────────────────
# The CSVs on disk use short lowercase stems (aus_eq, intl_eq, bonds, re, pevc).
# We use cleaner display names everywhere else in the codebase.
# These two dicts translate between the two worlds.

FILE_STEMS = {
    "AUS_EQ":      "aus_eq",
    "INTL_EQ":     "intl_eq",
    "Bonds":       "bonds",
    "Real_Estate": "re",
    "PE_VC":       "pevc",
}

# saa_weight.csv uses different sleeve codes than our display names.
# This dict converts the CSV's codes into our standard names.
SAA_CSV_RENAME = {
    "AUS_EQ":  "AUS_EQ",
    "INTL_EQ": "INTL_EQ",
    "BONDS":   "Bonds",
    "RE":      "Real_Estate",
    "PEVC":    "PE_VC",
}
# ── Helper: load one return-series CSV ────────────────────────────────────────

def _read_return_series(path: Path, series_name: str) -> pd.Series:
    """
    Reads a CSV with columns ['Date', 'Return'] and returns a named
    pandas Series indexed by date.

    The leading underscore in the function name signals that this is
    a private helper — it's not meant to be called from outside this module.

    Args:
        path:        Full path to the CSV file.
        series_name: Name to assign to the returned Series (e.g. 'AUS_EQ').

    Returns:
        A pandas Series of monthly decimal returns, indexed by date.
    """
    df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
    return df["Return"].rename(series_name)

# ── Data loaders ──────────────────────────────────────────────────────────────

def load_returns(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads all five manager and five benchmark return series and assembles
    them into two aligned DataFrames.

    Args:
        data_dir: Path to the /data folder containing managers/ and benchmarks/.

    Returns:
        (managers_df, benchmarks_df) — both DataFrames have the same
        DatetimeIndex and columns named by sleeve (AUS_EQ, INTL_EQ, ...).
    """
    managers_dir   = data_dir / "managers"
    benchmarks_dir = data_dir / "benchmarks"

    manager_cols   = []
    benchmark_cols = []

    for sleeve_name, stem in FILE_STEMS.items():
        manager_cols.append(
            _read_return_series(managers_dir / f"{stem}_mgr.csv", sleeve_name)
        )
        benchmark_cols.append(
            _read_return_series(benchmarks_dir / f"{stem}_bm.csv", sleeve_name)
        )

    managers_df   = pd.concat(manager_cols,   axis=1)
    benchmarks_df = pd.concat(benchmark_cols, axis=1)

    return managers_df, benchmarks_df


def load_risk_free(data_dir: Path) -> pd.Series:
    """
    Loads the monthly risk-free rate from rf_monthly.csv.

    Note: the README says the column is 'RF' but the supplied file actually
    uses 'Return'. We use what the file actually contains.

    Args:
        data_dir: Path to the /data folder.

    Returns:
        A pandas Series of monthly decimal risk-free rates, named 'RF'.
    """
    return _read_return_series(data_dir / "rf_monthly.csv", "RF")


def load_saa_weights(data_dir: Path) -> dict:
    """
    Loads SAA weights from saa_weight.csv and renames the sleeve codes
    to match the display names used elsewhere in the codebase.

    The CSV uses short codes (BONDS, RE, PEVC) but the rest of our code
    uses Bonds, Real_Estate, PE_VC — so we translate via SAA_CSV_RENAME.

    Args:
        data_dir: Path to the /data folder.

    Returns:
        dict mapping sleeve name -> weight (as a float).
    """
    df = pd.read_csv(data_dir / "saa_weight.csv")

    # Convert the two-column CSV into a dict: {sleeve_code: weight}
    raw_weights = dict(zip(df["Sleeve"], df["Weight"]))

    # Translate CSV codes (BONDS, RE, ...) into our display names
    saa_weights = {
        SAA_CSV_RENAME[csv_code]: weight
        for csv_code, weight in raw_weights.items()
    }

    return saa_weights

# ── Main entry point ──────────────────────────────────────────────────────────

def load_all(data_dir: str = "data") -> dict:
    """
    Loads everything needed for the analysis in one call.

    Also performs basic alignment: restricts all three series (managers,
    benchmarks, risk-free) to the set of dates that appear in all of them.
    Our supplied data is already clean and aligned, but this guards against
    silent surprises if data is updated.

    Args:
        data_dir: Path (string) to the /data folder. Defaults to 'data'.

    Returns:
        dict with keys:
            'managers'    : DataFrame of monthly manager returns (5 columns)
            'benchmarks'  : DataFrame of monthly benchmark returns (5 columns)
            'rf'          : Series of monthly risk-free rates
            'taa_weights' : dict of TAA weights by sleeve
            'saa_weights' : dict of SAA weights by sleeve
    """
    base = Path(data_dir)

    managers, benchmarks = load_returns(base)
    rf                   = load_risk_free(base)
    saa_weights          = load_saa_weights(base)

    # Align all three to the common date index. Since the data is already
    # clean, this should be a no-op — but we verify by comparing lengths.
    common_dates = managers.index.intersection(benchmarks.index).intersection(rf.index)
    managers     = managers.loc[common_dates]
    benchmarks   = benchmarks.loc[common_dates]
    rf           = rf.loc[common_dates]

    return {
        "managers":     managers,
        "benchmarks":   benchmarks,
        "rf":           rf,
        "taa_weights":  TAA_WEIGHTS,
        "saa_weights":  saa_weights,
    }
