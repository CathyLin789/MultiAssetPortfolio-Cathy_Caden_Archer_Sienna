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