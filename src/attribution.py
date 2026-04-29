"""
attribution.py
Brinson performance attribution for the multi-asset portfolio.

The module decomposes each sleeve's active return into 2 components:    
    - Allocation effect = the impact of over/underweighing the asset classes relative to the benchmark 
    - Selection effect = the contribution from manager performance versus benchmarks

Effects are calculated on a monthly basis and then aggregated into a full-sample summary across all five sleeves.
"""

import pandas as pd
import numpy as np


def brinson_monthly(
    mgr_returns: pd.DataFrame,
    bm_returns: pd.DataFrame,
    taa_weights: dict,
    saa_weights: dict,
    include_interaction: bool = False,
) -> pd.DataFrame:
    """
    Compute monthly Brinson allocation and selection effects for each sleeve.

    Parameters
    ----------
    mgr_returns         : DataFrame of monthly manager returns (columns = sleeves)
    bm_returns          : DataFrame of monthly benchmark returns (columns = sleeves)
    taa_weights         : dict of TAA (portfolio) weights  e.g. {"AUS_EQ": 0.35, ...}
    saa_weights         : dict of SAA (benchmark) weights  e.g. {"AUS_EQ": 0.40, ...}
    include_interaction : if True, adds an interaction effect column to the output

    Returns
    -------
    DataFrame with columns: date, sleeve, allocation, selection, [interaction]
    """
    sleeves = list(taa_weights.keys())
    records = []

    for date, row in mgr_returns.iterrows():
        for sleeve in sleeves:
            wP = taa_weights[sleeve]
            wB = saa_weights[sleeve]
            rP = row[sleeve]
            rB = bm_returns.loc[date, sleeve]

            allocation  = (wP - wB) * rB
            selection   = wB * (rP - rB)
            interaction = (wP - wB) * (rP - rB)

            records.append({
                "date":       date,
                "sleeve":     sleeve,
                "allocation": allocation,
                "selection":  selection,
                **( {"interaction": interaction} if include_interaction else {} )
            })

    return pd.DataFrame(records)


def brinson_summary(
    monthly_df: pd.DataFrame,
    include_interaction: bool = False,
) -> pd.DataFrame:
    """
    Aggregate monthly Brinson effects into a summary table.

    Parameters
    ----------
    monthly_df          : output of brinson_monthly()
    include_interaction : if True, includes the interaction effect column
                          and uses it in the Total Active Contribution

    Returns
    -------
    DataFrame indexed by sleeve with columns:
        Allocation Effect, Selection Effect, [Interaction Effect],
        Total Active Contribution
    """
    cols = ["allocation", "selection"]
    if include_interaction:
        cols.append("interaction")

    summary = (
        monthly_df
        .groupby("sleeve")[cols]
        .sum()
        .rename(columns={
            "allocation":  "Allocation Effect",
            "selection":   "Selection Effect",
            "interaction": "Interaction Effect",
        })
    )

    effect_cols = ["Allocation Effect", "Selection Effect"]
    if include_interaction:
        effect_cols.append("Interaction Effect")

    summary["Total Active Contribution"] = summary[effect_cols].sum(axis=1)

    # Totals row
    totals = summary.sum().rename("Total")
    summary = pd.concat([summary, totals.to_frame().T])

    return summary