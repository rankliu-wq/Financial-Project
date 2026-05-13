from __future__ import annotations

import numpy as np
import pandas as pd


def performance_summary(
    equity: pd.DataFrame,
    trades: pd.DataFrame,
    initial_cash: float,
    annualization: int,
) -> dict[str, float | int]:
    if equity.empty:
        return {}

    ending_equity = float(equity["Equity"].iloc[-1])
    total_return = ending_equity / initial_cash - 1
    days = max((equity.index[-1] - equity.index[0]).days, 1)
    years = days / 365.25
    cagr = (ending_equity / initial_cash) ** (1 / years) - 1 if years > 0 else 0
    returns = equity["Returns"].fillna(0)
    volatility = returns.std(ddof=0) * np.sqrt(annualization)
    sharpe = (returns.mean() * annualization) / volatility if volatility > 0 else 0
    max_drawdown = float(equity["Drawdown"].min())
    trade_count = int(len(trades))
    win_rate = float((trades["PnL"] > 0).mean()) if trade_count else 0

    return {
        "ending_equity": ending_equity,
        "total_return": total_return,
        "cagr": cagr,
        "volatility": float(volatility),
        "sharpe": float(sharpe),
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
        "trade_count": trade_count,
    }


def monthly_returns(equity: pd.DataFrame) -> pd.DataFrame:
    if equity.empty:
        return pd.DataFrame()

    month_end = equity["Equity"].resample("ME").last()
    monthly = month_end.pct_change().dropna()
    if monthly.empty:
        return pd.DataFrame()

    result = monthly.to_frame("Return")
    result["Year"] = result.index.year
    result["Month"] = result.index.month
    heatmap = result.pivot(index="Year", columns="Month", values="Return")
    return heatmap.reindex(columns=range(1, 13))
