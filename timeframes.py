from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TimeframeSpec:
    label: str
    yahoo_interval: str
    annualization_trading_days: int
    annualization_calendar_days: int
    resample_rule: str | None = None
    default_lookback_days: int = 365 * 3
    note: str = ""


TIMEFRAME_SPECS = {
    "5 分鐘": TimeframeSpec("5 分鐘", "5m", 252 * 78, 365 * 288, default_lookback_days=30, note="分鐘資料通常僅適合近 60 天內分析。"),
    "15 分鐘": TimeframeSpec("15 分鐘", "15m", 252 * 26, 365 * 96, default_lookback_days=45, note="分鐘資料通常僅適合近 60 天內分析。"),
    "30 分鐘": TimeframeSpec("30 分鐘", "30m", 252 * 13, 365 * 48, default_lookback_days=60, note="分鐘資料通常僅適合近 60 天內分析。"),
    "1 小時": TimeframeSpec("1 小時", "1h", 252 * 7, 365 * 24, default_lookback_days=120, note="小時資料的可回溯區間依 Yahoo Finance 限制而定。"),
    "4 小時": TimeframeSpec("4 小時", "1h", 252 * 2, 365 * 6, resample_rule="4h", default_lookback_days=180, note="由 1 小時資料重採樣而成。"),
    "日線": TimeframeSpec("日線", "1d", 252, 365),
    "週線": TimeframeSpec("週線", "1wk", 52, 52),
    "月線": TimeframeSpec("月線", "1mo", 12, 12),
}


DEFAULT_TIMEFRAME = "日線"


def timeframe_labels() -> list[str]:
    return list(TIMEFRAME_SPECS)


def annualization_for_timeframe(timeframe: str, allow_weekends: bool) -> int:
    spec = TIMEFRAME_SPECS[timeframe]
    if allow_weekends:
        return spec.annualization_calendar_days
    return spec.annualization_trading_days


def is_intraday_timeframe(timeframe: str) -> bool:
    return TIMEFRAME_SPECS[timeframe].yahoo_interval.endswith(("m", "h"))
