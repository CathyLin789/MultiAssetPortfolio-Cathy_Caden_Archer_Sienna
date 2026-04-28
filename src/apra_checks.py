"""

This code implements APRA performance and risk checks for the total
multi-asset portfolio.

These checks are designed to replicate regulatory
oversight used by institutions such as APRA, specifically focusing on:

- Long-run return objectives
- Volatility (risk) limits
- Downside risk via maximum drawdown
- Portfolio resilience under stress scenarios

All calculations are based on monthly portfolio returns.
"""

import pandas as pd
import numpy as np


# Core portfolio metrics 

def _annualised_return(returns: pd.Series) -> float:
    """
    This calculates the annualised return using geometric compounding.

    Args: 
        returns: Series of monthly returns.

    Returns:
        Annualised return (float).
    """
    returns = returns.dropna()
    return (1 + returns).prod() ** (12 / len(returns)) - 1

def _annualised_volatility(returns: pd.Series) -> float:
    """
    Computes annualised volatility from monthly returns.

    Args:
        returns: Series of monthly returns.

    Returns:
        Annualised standard deviation (float).
    """
    returns = returns.dropna()
    return returns.std()*np.sqrt(12)


def _max_drawdown(returns: pd.Series) -> float:
    """
    This calculates the maximum drawdown (peak-to-trough loss).

    Args:
        returns: Series of monthly returns.

    Returns:
        Maximum drawdown (negative float).
    """
    returns = returns.dropna()

    wealth_index = (1 + returns).cumprod()
    previous_peak = wealth_index.cummax()
    drawdown = (wealth_index - previous_peak) / previous_peak

    return drawdown.min()


# Portfolio construction help

def _compute_portfolio_returns(managers: pd.DataFrame, taa_weights: dict) -> pd.Series:
    """
    This calculates total portfolio returns using TAA weights.

    Args:
        managers: DataFrame of manager returns (columns = sleeves).
        taa_weights: mapping sleeve -> weight.

    Returns:
        Series of total portfolio returns.
    """
    weights = pd.Series(taa_weights)

    # Multiply each sleeve return by its weight, then sum across sleeves
    portfolio_returns = managers.mul(weights, axis=1).sum(axis=1)

    return portfolio_returns

# APRA-style checks 

def run_apra_checks(data: dict) -> pd.DataFrame:
    """
    Runs all APRA performance and risk checks.

    Args:
        data: dict returned by data_loader.load_all()

    Returns:
        DataFrame summarising all APRA checks, including:
        - Actual values
        - Thresholds
        - Pass/Fail status
    """
    managers     = data["managers"]
    taa_weights  = data["taa_weights"]

    # Step 1: Total portfolio returns 
    portfolio_returns = _compute_portfolio_returns(managers, taa_weights)

    #Step 2: Compute key metrics 
    ann_return = _annualised_return(portfolio_returns)
    ann_vol    = _annualised_volatility(portfolio_returns)
    drawdown   = _max_drawdown(portfolio_returns)

    # Step 3: Define the APRA thresholds 

    RETURN_TARGET      = 0.06   # 6% long-term return objective
    VOLATILITY_LIMIT   = 0.12   # 12% annual volatility cap
    DRAWDOWN_LIMIT     = -0.25  # max loss no worse than -25%
    SHOCK_LOSS_LIMIT   = -0.15  # stress loss threshold

  # Step 4: Stress scenarios
    # Following Reader §5.5: R_shock = w_eq*(r_eq + S_eq) + sum(w_i * r_i)
    # Shock is applied on top of actual returns in a "typical" baseline month.
    # We use the median monthly return for each sleeve as the baseline,
    # representing normal market conditions on which the shock is overlaid.

    weights = pd.Series(taa_weights)
    actual_returns_stress = managers.median()

    # --- Scenario A: Equity Crash (per Reader §5.5) ---
    # AUS EQ and INTL EQ each shocked by -20%; other sleeves keep actual returns
    scenario_a = actual_returns_stress.copy()
    scenario_a["AUS_EQ"]  += -0.20
    scenario_a["INTL_EQ"] += -0.20
    shock_loss_a = (scenario_a * weights).sum()

    # --- Scenario B: Bond Yield Spike +150 bps (per Reader §5.5) ---
    # Bonds shocked by -5% (duration x 1.5%); REITs -5%; equities -2%; PE/VC -2%
    scenario_b = actual_returns_stress.copy()
    scenario_b["AUS_EQ"]  += -0.02
    scenario_b["INTL_EQ"] += -0.02
    scenario_b["BONDS"]   += -0.05
    scenario_b["RE"]      += -0.05
    scenario_b["PEVC"]    += -0.02
    shock_loss_b = (scenario_b * weights).sum()

    # Step 5: Assemble results table
    results = pd.DataFrame({
        "Check": [
            "Long-run return objective",
            "Volatility limit",
            "Maximum drawdown",
            "Stress scenario A (Equity crash)",
            "Stress scenario B (Bond yield spike)",
        ],
        "Actual": [
            ann_return,
            ann_vol,
            drawdown,
            shock_loss_a,
            shock_loss_b,
        ],
        "Threshold": [
            RETURN_TARGET,
            VOLATILITY_LIMIT,
            DRAWDOWN_LIMIT,
            SHOCK_LOSS_LIMIT,
            SHOCK_LOSS_LIMIT,
        ],
    })

    # Step 6: Pass/Fail logic
    results["Pass"] = [
        ann_return >= RETURN_TARGET,
        ann_vol <= VOLATILITY_LIMIT,
        drawdown >= DRAWDOWN_LIMIT,
        shock_loss_a >= SHOCK_LOSS_LIMIT,
        shock_loss_b >= SHOCK_LOSS_LIMIT,
    ]

    return results

# Visual diagnostics 

def plot_diagnostics(data: dict) -> None:
    """
    Generates key APRA diagnostic charts:
    - Wealth index
    - Drawdown profile
    - Rolling volatility

    Args:
        data: dict returned by data_loader.load_all()
    """
    import matplotlib.pyplot as plt

    managers     = data["managers"]
    taa_weights  = data["taa_weights"]

    portfolio_returns = _compute_portfolio_returns(managers, taa_weights)

    # Wealth index 
    wealth = (1 + portfolio_returns).cumprod()

    wealth.plot(title="Portfolio Wealth Index", figsize=(9, 5))
    plt.ylabel("Wealth")
    plt.grid()
    plt.show()

    # Drawdown 
    peak = wealth.cummax()
    drawdown = (wealth - peak) / peak

    drawdown.plot(title="Portfolio Drawdown", figsize=(9, 5))
    plt.axhline(-0.25, linestyle="--", label="Drawdown Limit")
    plt.legend()
    plt.grid()
    plt.show()

    # Rolling volatility 
    rolling_vol = portfolio_returns.rolling(12).std() * np.sqrt(12)

    rolling_vol.plot(title="Rolling 12M Volatility", figsize=(9, 5))
    plt.axhline(0.12, linestyle="--", label="Volatility Limit")
    plt.legend()
    plt.grid()
    plt.show()
