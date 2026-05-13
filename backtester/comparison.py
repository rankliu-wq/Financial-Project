from __future__ import annotations

from dataclasses import dataclass
import re

import numpy as np
import pandas as pd

from backtester.data import DataValidationError, MarketData, fetch_yahoo_prices
from backtester.timeframes import DEFAULT_TIMEFRAME


DEFAULT_COMPARISON_SYMBOLS = ["0050.TW", "0056.TW", "2330.TW"]


@dataclass(frozen=True)
class ComparisonResult:
    metrics: pd.DataFrame
    normalized: pd.DataFrame
    returns: pd.DataFrame
    best_symbols: dict[str, str]
    warnings: list[str]
    start: pd.Timestamp
    end: pd.Timestamp


def parse_symbol_list(raw_symbols: str) -> list[str]:
    symbols = []
    seen = set()
    for item in re.split(r"[\s,;，；]+", raw_symbols.upper().strip()):
        symbol = item.strip()
        if symbol and symbol not in seen:
            symbols.append(symbol)
            seen.add(symbol)
    return symbols


def compare_symbols(
    symbols: list[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    allow_weekends: bool,
    annualization: int,
    timeframe: str = DEFAULT_TIMEFRAME,
) -> ComparisonResult:
    if len(symbols) < 2:
        raise DataValidationError("請至少輸入兩檔標的進行比較。")

    data_by_symbol: dict[str, MarketData] = {}
    warnings: list[str] = []
    for symbol in symbols:
        try:
            data_by_symbol[symbol] = fetch_yahoo_prices(symbol, start, end, allow_weekends, timeframe)
        except DataValidationError as exc:
            warnings.append(f"{symbol}: {exc}")

    if len(data_by_symbol) < 2:
        detail = "；".join(warnings) if warnings else "可用標的不足。"
        raise DataValidationError(f"比較需要至少兩檔可用標的。{detail}")

    adjusted = {
        symbol: market_data.prices["Adj Close"].rename(symbol)
        for symbol, market_data in data_by_symbol.items()
    }
    price_frame = pd.concat(adjusted.values(), axis=1).dropna(how="any")
    price_frame = price_frame[price_frame.gt(0).all(axis=1)]
    if len(price_frame) < 2:
        raise DataValidationError("多檔標的沒有足夠的共同日期資料可比較。")

    normalized = price_frame.divide(price_frame.iloc[0]).multiply(100)
    returns = price_frame.pct_change().dropna()
    metrics = _comparison_metrics(normalized, returns, annualization)
    best_symbols = _best_by_metric(metrics)

    return ComparisonResult(
        metrics=metrics,
        normalized=normalized,
        returns=returns,
        best_symbols=best_symbols,
        warnings=warnings,
        start=price_frame.index.min(),
        end=price_frame.index.max(),
    )


def _comparison_metrics(normalized: pd.DataFrame, returns: pd.DataFrame, annualization: int) -> pd.DataFrame:
    rows = []
    days = max((normalized.index[-1] - normalized.index[0]).days, 1)
    years = days / 365.25

    for symbol in normalized.columns:
        equity = normalized[symbol]
        symbol_returns = returns[symbol].fillna(0)
        total_return = float(equity.iloc[-1] / equity.iloc[0] - 1)
        cagr = float((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1) if years > 0 else 0.0
        volatility = float(symbol_returns.std(ddof=0) * np.sqrt(annualization))
        sharpe = float((symbol_returns.mean() * annualization) / volatility) if volatility > 0 else 0.0
        drawdown = equity / equity.cummax() - 1
        max_drawdown = float(drawdown.min())
        rows.append(
            {
                "標的": symbol,
                "總報酬": total_return,
                "CAGR": cagr,
                "年化波動": volatility,
                "Sharpe": sharpe,
                "最大回撤": max_drawdown,
                "期末指數": float(equity.iloc[-1]),
            }
        )

    return pd.DataFrame(rows).set_index("標的")


def _best_by_metric(metrics: pd.DataFrame) -> dict[str, str]:
    higher_is_better = ["總報酬", "CAGR", "Sharpe", "期末指數"]
    lower_is_better = ["年化波動"]
    closer_to_zero_is_better = ["最大回撤"]

    best = {metric: str(metrics[metric].idxmax()) for metric in higher_is_better}
    best.update({metric: str(metrics[metric].idxmin()) for metric in lower_is_better})
    best.update({metric: str(metrics[metric].idxmax()) for metric in closer_to_zero_is_better})
    return best
