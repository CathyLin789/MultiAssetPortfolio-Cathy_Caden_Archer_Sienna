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

    # Step 4: Stress scenario assumptions
    #Severe, but reasonable market shock across asset classes.

    shock_adjustments = pd.Series({
    "AUS_EQ":      -0.20,
    "INTL_EQ":     -0.20,
    "BONDS":       -0.02,
    "RE":          -0.10,
    "PEVC":        -0.25,
})

    # Add shocks to actual monthly returns
    shocked_returns = managers + shock_adjustments

    # Re compute portfolio returns using shocked data
    weights = pd.Series(taa_weights)
    shocked_portfolio_returns = shocked_returns.mul(weights, axis=1).sum(axis=1)

    # Define shock loss as worst observed monthly outcome
    shock_loss = shocked_portfolio_returns.min()

 # Step 5: Assemble results table
    results = pd.DataFrame({
        "Check": [
            "Long-run return objective",
            "Volatility limit",
            "Maximum drawdown",
            "Stress scenario loss"
        ],
        "Actual": [
            ann_return,
            ann_vol,
            drawdown,
            shock_loss
        ],
        "Threshold": [
            RETURN_TARGET,
            VOLATILITY_LIMIT,
            DRAWDOWN_LIMIT,
            SHOCK_LOSS_LIMIT
        ]
    })

    # Step 6: Pass/Fail logic
    results["Pass"] = [
        ann_return >= RETURN_TARGET,
        ann_vol <= VOLATILITY_LIMIT,
        drawdown >= DRAWDOWN_LIMIT,
        shock_loss >= SHOCK_LOSS_LIMIT
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
