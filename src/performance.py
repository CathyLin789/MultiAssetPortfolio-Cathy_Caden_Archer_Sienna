"""
performance.py

Performance and risk metrics for the multi-asset portfolio.

Each metric is a small pure function that takes a return series (and
sometimes a benchmark or risk-free series) and returns a number.

Implements:
    - annualised_return        (geometric)
    - annualised_volatility    (sqrt-of-time scaled)
    - sharpe_ratio
    - active_return
    - tracking_error
    - information_ratio
    - max_drawdown
    - wealth_index             (helper for charts)

Two summary helpers (sleeve_summary, all_sleeves_summary) bundle the
metrics into a tidy DataFrame for the report.
"""

import numpy as np
import pandas as pd

# ── Single-series metrics ─────────────────────────────────────────────────────

def annualised_return(monthly_returns: pd.Series) -> float:
    """
    Geometric annualised return from a series of monthly returns.

    Formula: prod(1 + r_t) ^ (12/n) - 1
    where n is the number of monthly observations.

    Args:
        monthly_returns: monthly decimal returns (e.g. 0.012 = +1.2%).

    Returns:
        Annualised return as a decimal.
    """
    n = len(monthly_returns)
    growth_factor = (1 + monthly_returns).prod()
    return growth_factor ** (12 / n) - 1

def annualised_volatility(monthly_returns: pd.Series) -> float:
    """
    Annualised volatility from monthly returns. Scales monthly standard
    deviation by sqrt(12) (square-root-of-time rule).

    Formula: std(r) * sqrt(12)

    Args:
        monthly_returns: monthly decimal returns.

    Returns:
        Annualised volatility as a decimal.
    """
    return monthly_returns.std() * np.sqrt(12)