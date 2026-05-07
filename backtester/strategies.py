from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


class StrategyError(ValueError):
    """Raised when strategy parameters do not fit the provided data."""


@dataclass(frozen=True)
class StrategyOutput:
    name: str
    signal: pd.Series
    indicators: pd.DataFrame


def generate_strategy(
    prices: pd.DataFrame,
    strategy_name: str,
    short_window: int = 20,
    long_window: int = 60,
    rsi_window: int = 14,
    rsi_buy: float = 30,
    rsi_sell: float = 70,
) -> StrategyOutput:
    if strategy_name == "買入持有":
        return buy_and_hold(prices)
    if strategy_name == "均線交叉":
        return sma_cross(prices, short_window, long_window)
    if strategy_name == "RSI 反轉":
        return rsi_reversal(prices, rsi_window, rsi_buy, rsi_sell)
    raise StrategyError(f"未知策略：{strategy_name}")


def buy_and_hold(prices: pd.DataFrame) -> StrategyOutput:
    signal = pd.Series(1, index=prices.index, name="Signal", dtype=int)
    indicators = pd.DataFrame(index=prices.index)
    return StrategyOutput("買入持有", signal, indicators)


def sma_cross(prices: pd.DataFrame, short_window: int, long_window: int) -> StrategyOutput:
    if short_window < 2 or long_window < 3:
        raise StrategyError("均線天數需要大於 1。")
    if short_window >= long_window:
        raise StrategyError("短期均線天數必須小於長期均線天數。")
    if len(prices) < long_window:
        raise StrategyError("價格資料不足以計算長期均線。")

    close = prices["Close"]
    short_ma = close.rolling(short_window, min_periods=short_window).mean()
    long_ma = close.rolling(long_window, min_periods=long_window).mean()
    signal = (short_ma > long_ma).astype(int).fillna(0)
    indicators = pd.DataFrame(
        {
            f"SMA {short_window}": short_ma,
            f"SMA {long_window}": long_ma,
        },
        index=prices.index,
    )
    return StrategyOutput("均線交叉", signal.rename("Signal"), indicators)


def rsi_reversal(
    prices: pd.DataFrame,
    rsi_window: int,
    rsi_buy: float,
    rsi_sell: float,
) -> StrategyOutput:
    if rsi_window < 2:
        raise StrategyError("RSI 週期需要大於 1。")
    if rsi_buy >= rsi_sell:
        raise StrategyError("RSI 買進門檻必須小於賣出門檻。")
    if len(prices) < rsi_window + 2:
        raise StrategyError("價格資料不足以計算 RSI。")

    rsi = calculate_rsi(prices["Close"], rsi_window)
    signal = pd.Series(0, index=prices.index, name="Signal", dtype=int)
    in_position = False
    for date, value in rsi.items():
        if np.isnan(value):
            signal.loc[date] = int(in_position)
            continue
        if not in_position and value < rsi_buy:
            in_position = True
        elif in_position and value > rsi_sell:
            in_position = False
        signal.loc[date] = int(in_position)

    indicators = pd.DataFrame({f"RSI {rsi_window}": rsi}, index=prices.index)
    return StrategyOutput("RSI 反轉", signal, indicators)


def calculate_rsi(close: pd.Series, window: int) -> pd.Series:
    delta = close.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    avg_gain = gains.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    avg_loss = losses.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50).rename("RSI")
