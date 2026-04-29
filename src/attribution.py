"""
attribution.py
Brinson attribution: allocation and selection effects by sleeve.
"""

import pandas as pd
import numpy as np


def brinson_monthly(
    mgr_returns: pd.DataFrame,
    bm_returns: pd.DataFrame,
    taa_weights: dict,
    saa_weights: dict,
) -> pd.DataFrame:
    """
    Compute monthly Brinson allocation and selection effects for each sleeve.

    Parameters
    ----------
    mgr_returns  : DataFrame of monthly manager returns (columns = sleeves)
    bm_returns   : DataFrame of monthly benchmark returns (columns = sleeves)
    taa_weights  : dict of TAA (portfolio) weights  e.g. {"AUS_EQ": 0.35, ...}
    saa_weights  : dict of SAA (benchmark) weights  e.g. {"AUS_EQ": 0.40, ...}

    Returns
    -------
    DataFrame with columns: date, sleeve, allocation, selection
    """
    sleeves = list(taa_weights.keys())
    records = []

    for date, row in mgr_returns.iterrows():
        for sleeve in sleeves:
            wP = taa_weights[sleeve]
            wB = saa_weights[sleeve]
            rP = row[sleeve]
            rB = bm_returns.loc[date, sleeve]

            allocation = (wP - wB) * rB
            selection  = wB * (rP - rB)

            records.append({
                "date":       date,
                "sleeve":     sleeve,
                "allocation": allocation,
                "selection":  selection,
            })

    return pd.DataFrame(records)


def brinson_summary(monthly_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate monthly Brinson effects into an annualised summary table.

    Sums monthly effects then scales by 12 to annualise, consistent with
    the linear approximation used in most attribution systems.

    Returns
    -------
    DataFrame indexed by sleeve with columns:
        Allocation Effect, Selection Effect, Total Active Contribution
    """
    summary = (
        monthly_df
        .groupby("sleeve")[["allocation", "selection"]]
        .sum()
        .rename(columns={
            "allocation": "Allocation Effect",
            "selection":  "Selection Effect",
        })
    )
    summary["Total Active Contribution"] = (
        summary["Allocation Effect"] + summary["Selection Effect"]
    )

    # Annualise
    summary *= 12

    # Totals row
    totals = summary.sum().rename("Total")
    summary = pd.concat([summary, totals.to_frame().T])

    return summary