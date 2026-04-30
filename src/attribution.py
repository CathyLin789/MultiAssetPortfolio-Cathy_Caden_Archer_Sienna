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
import matplotlib.pyplot as plt


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


# ---- Key Tables and Figures ----

# --- Table 6.1 ---

def table_6_1(managers, benchmarks, taa_weights, saa_weights):
    from attribution import brinson_monthly, brinson_summary

    monthly_attr = brinson_monthly(
        mgr_returns = managers,
        bm_returns  = benchmarks,
        taa_weights = taa_weights,
        saa_weights = saa_weights,
    )

    attr_summary = brinson_summary(monthly_attr) / 10  # 10 years -> annualised

    return attr_summary.style.format({
        "Allocation Effect":         "{:.2%}",
        "Selection Effect":          "{:.2%}",
        "Total Active Contribution": "{:.2%}",
    })

# --- Figure 6.1 ---

def plot_figure_6_1(managers, benchmarks, taa_weights, saa_weights):
    from attribution import brinson_monthly, brinson_summary

    monthly_attr = brinson_monthly(
        mgr_returns = managers,
        bm_returns  = benchmarks,
        taa_weights = taa_weights,
        saa_weights = saa_weights,
    )

    attr_summary = brinson_summary(monthly_attr) / 10  # 10 years -> annualised

    plot_data = attr_summary.drop("Total")[["Allocation Effect", "Selection Effect"]]

    fig, ax = plt.subplots(figsize=(10, 5))
    x     = np.arange(len(plot_data))
    width = 0.35

    ax.bar(x - width/2, plot_data["Allocation Effect"], width, label="Allocation Effect")
    ax.bar(x + width/2, plot_data["Selection Effect"],  width, label="Selection Effect")

    ax.set_xticks(x)
    ax.set_xticklabels(plot_data.index)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_ylabel("Cumulative Contribution to Active Return")
    ax.set_title("Brinson Attribution: Allocation vs Selection by Asset Class")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.show()

# --- Figure 6.2 ---

def plot_figure_6_2(managers, benchmarks, taa_weights, saa_weights):
    from attribution import brinson_monthly

    monthly_attr = brinson_monthly(
        mgr_returns = managers,
        bm_returns  = benchmarks,
        taa_weights = taa_weights,
        saa_weights = saa_weights,
    )

    # Pivot monthly attribution to get allocation and selection per sleeve over time
    alloc_pivot = monthly_attr.pivot(index="date", columns="sleeve", values="allocation")
    sel_pivot   = monthly_attr.pivot(index="date", columns="sleeve", values="selection")

    # Cumulative sums over time
    alloc_cumsum = alloc_pivot.cumsum()
    sel_cumsum   = sel_pivot.cumsum()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)

    # Allocation effects
    alloc_cumsum.plot(ax=axes[0])
    axes[0].set_title("Cumulative Allocation Effect by Sleeve")
    axes[0].set_ylabel("Cumulative Contribution")
    axes[0].set_xlabel("")
    axes[0].axhline(0, color="black", linewidth=0.5)
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=9)

    # Selection effects
    sel_cumsum.plot(ax=axes[1])
    axes[1].set_title("Cumulative Selection Effect by Sleeve")
    axes[1].set_ylabel("Cumulative Contribution")
    axes[1].set_xlabel("")
    axes[1].axhline(0, color="black", linewidth=0.5)
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=9)

    plt.tight_layout()
    plt.show()

# --- Table 6.2 ---
def table_6_2(managers, benchmarks, taa_weights, saa_weights):
    from attribution import brinson_monthly, brinson_summary

    monthly_attr_ext = brinson_monthly(
        mgr_returns         = managers,
        bm_returns          = benchmarks,
        taa_weights         = taa_weights,
        saa_weights         = saa_weights,
        include_interaction = True,
    )

    attr_summary_ext = brinson_summary(monthly_attr_ext, include_interaction=True) / 10  # 10 years -> annualised

    return attr_summary_ext.style.format({
        "Allocation Effect":         "{:.4%}",
        "Selection Effect":          "{:.4%}",
        "Interaction Effect":        "{:.4%}",
        "Total Active Contribution": "{:.4%}",
    })

# --- Figure 6.3 ---

def plot_figure_6_3(managers, benchmarks, taa_weights, saa_weights):
    from attribution import brinson_monthly, brinson_summary

    monthly_attr_ext = brinson_monthly(
        mgr_returns         = managers,
        bm_returns          = benchmarks,
        taa_weights         = taa_weights,
        saa_weights         = saa_weights,
        include_interaction = True,
    )

    attr_summary_ext = brinson_summary(monthly_attr_ext, include_interaction=True) / 10  # 10 years -> annualised

    plot_data = attr_summary_ext.drop("Total")[["Allocation Effect", "Selection Effect", "Interaction Effect"]]

    fig, ax = plt.subplots(figsize=(10, 5))

    x     = np.arange(len(plot_data))
    width = 0.25

    ax.bar(x - width, plot_data["Allocation Effect"],  width, label="Allocation Effect")
    ax.bar(x,         plot_data["Selection Effect"],   width, label="Selection Effect")
    ax.bar(x + width, plot_data["Interaction Effect"], width, label="Interaction Effect")

    ax.set_xticks(x)
    ax.set_xticklabels(plot_data.index)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_ylabel("Contribution to Active Return (annualised)")
    ax.set_title("Brinson Attribution: All Three Effects by Asset Class")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.show()