from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd


class StrategyError(ValueError):
    """Raised when strategy parameters do not fit the provided data."""


@dataclass(frozen=True)
class StrategyOutput:
    name: str
    signal: pd.Series
    indicators: pd.DataFrame


@dataclass(frozen=True)
class ParamSpec:
    key: str
    label: str
    kind: str
    default: Any
    min_value: float | int | None = None
    max_value: float | int | None = None
    step: float | int | None = None
    format: str | None = None
    options: tuple[Any, ...] = ()
    help: str | None = None


@dataclass(frozen=True)
class StrategySpec:
    name: str
    factory: Callable[..., StrategyOutput]
    params: tuple[ParamSpec, ...] = ()


def generate_strategy(
    prices: pd.DataFrame,
    strategy_name: str,
    **params: Any,
) -> StrategyOutput:
    spec = STRATEGY_SPECS.get(strategy_name)
    if spec is None:
        raise StrategyError(f"未知策略：{strategy_name}")

    resolved_params = {param.key: params.get(param.key, param.default) for param in spec.params}
    return spec.factory(prices, **resolved_params)


def strategy_names() -> list[str]:
    return list(STRATEGY_SPECS)


def buy_and_hold(prices: pd.DataFrame, **_: Any) -> StrategyOutput:
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


def bollinger_bands(
    prices: pd.DataFrame,
    bb_window: int,
    bb_std: float,
    bb_exit: str,
) -> StrategyOutput:
    if bb_window < 5:
        raise StrategyError("布林通道週期需要至少 5 天。")
    if bb_std <= 0:
        raise StrategyError("標準差倍數必須大於 0。")
    if len(prices) < bb_window:
        raise StrategyError("價格資料不足以計算布林通道。")

    close = prices["Close"]
    middle = close.rolling(bb_window, min_periods=bb_window).mean()
    rolling_std = close.rolling(bb_window, min_periods=bb_window).std(ddof=0)
    upper = middle + bb_std * rolling_std
    lower = middle - bb_std * rolling_std

    signal = pd.Series(0, index=prices.index, name="Signal", dtype=int)
    in_position = False
    for date, close_value in close.items():
        if np.isnan(middle.loc[date]) or np.isnan(upper.loc[date]) or np.isnan(lower.loc[date]):
            signal.loc[date] = int(in_position)
            continue

        if not in_position and close_value < lower.loc[date]:
            in_position = True
        elif in_position:
            if bb_exit == "上軌出場" and close_value > upper.loc[date]:
                in_position = False
            elif bb_exit == "中軌出場" and close_value > middle.loc[date]:
                in_position = False
        signal.loc[date] = int(in_position)

    multiplier = f"{bb_std:g}"
    indicators = pd.DataFrame(
        {
            f"BB 中軌 {bb_window}": middle,
            f"BB 上軌 {bb_window}x{multiplier}": upper,
            f"BB 下軌 {bb_window}x{multiplier}": lower,
        },
        index=prices.index,
    )
    return StrategyOutput("布林通道", signal, indicators)


def calculate_rsi(close: pd.Series, window: int) -> pd.Series:
    delta = close.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    avg_gain = gains.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    avg_loss = losses.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50).rename("RSI")


STRATEGY_SPECS = {
    "買入持有": StrategySpec("買入持有", buy_and_hold),
    "均線交叉": StrategySpec(
        "均線交叉",
        sma_cross,
        (
            ParamSpec("short_window", "短均線", "int", 20, 2, 250, 1),
            ParamSpec("long_window", "長均線", "int", 60, 3, 400, 1),
        ),
    ),
    "RSI 反轉": StrategySpec(
        "RSI 反轉",
        rsi_reversal,
        (
            ParamSpec("rsi_window", "RSI 週期", "int", 14, 2, 80, 1),
            ParamSpec("rsi_buy", "買進門檻", "int", 30, 1, 99, 1),
            ParamSpec("rsi_sell", "出場門檻", "int", 70, 1, 99, 1),
        ),
    ),
    "布林通道": StrategySpec(
        "布林通道",
        bollinger_bands,
        (
            ParamSpec("bb_window", "通道週期", "int", 20, 5, 250, 1),
            ParamSpec("bb_std", "標準差倍數", "float", 2.0, 0.5, 4.0, 0.1, "%.1f"),
            ParamSpec("bb_exit", "出場方式", "select", "中軌出場", options=("中軌出場", "上軌出場")),
        ),
    ),
}
