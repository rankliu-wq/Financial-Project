from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import pandas as pd
import requests
import yfinance as yf
import yfinance.cache as yf_cache

from backtester.timeframes import DEFAULT_TIMEFRAME, TIMEFRAME_SPECS


REQUIRED_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]
OPTIONAL_COLUMNS = ["Adj Close"]
OUTPUT_COLUMNS = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache" / "yfinance"


class DataValidationError(ValueError):
    """Raised when input price data cannot be used for backtesting."""


@dataclass(frozen=True)
class MarketData:
    symbol: str
    prices: pd.DataFrame
    source: str


def fetch_yahoo_prices(
    symbol: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    allow_weekends: bool,
    timeframe: str = DEFAULT_TIMEFRAME,
) -> MarketData:
    symbol = symbol.strip().upper()
    if not symbol:
        raise DataValidationError("請輸入金融商品代號。")

    if timeframe not in TIMEFRAME_SPECS:
        raise DataValidationError(f"不支援的時間週期：{timeframe}。")

    _configure_yfinance_cache()
    errors: list[str] = []
    data = pd.DataFrame()

    try:
        data = _download_with_yahoo_chart_api(symbol, start, end, timeframe)
    except DataValidationError as exc:
        errors.append(str(exc))

    if data.empty:
        try:
            data = _download_with_yfinance(symbol, start, end, timeframe)
        except DataValidationError as exc:
            errors.append(str(exc))

    if data.empty:
        message = f"找不到 {symbol} 在指定日期範圍內的價格資料。請確認代號、日期範圍，或改用 CSV 上傳。"
        if errors:
            message += " 下載診斷：" + "；".join(errors)
        raise DataValidationError(message)

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    prices = normalize_price_frame(data, allow_weekends=allow_weekends)
    prices = _resample_prices(prices, timeframe)
    return MarketData(symbol=symbol, prices=prices, source="Yahoo Finance")


def _configure_yfinance_cache() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    yf_cache.set_cache_location(str(CACHE_DIR))


def _download_with_yfinance(symbol: str, start: pd.Timestamp, end: pd.Timestamp, timeframe: str) -> pd.DataFrame:
    spec = TIMEFRAME_SPECS[timeframe]
    try:
        return yf.download(
            symbol,
            start=start.date().isoformat(),
            end=(end + pd.Timedelta(days=1)).date().isoformat(),
            interval=spec.yahoo_interval,
            progress=False,
            auto_adjust=False,
            group_by="column",
            threads=False,
            timeout=20,
        )
    except Exception as exc:
        raise DataValidationError(f"Yahoo Finance 下載失敗：{exc}") from exc


def _download_with_yahoo_chart_api(symbol: str, start: pd.Timestamp, end: pd.Timestamp, timeframe: str) -> pd.DataFrame:
    spec = TIMEFRAME_SPECS[timeframe]
    start_ts = int(start.normalize().timestamp())
    end_ts = int((end.normalize() + pd.Timedelta(days=1)).timestamp())
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
    params = {
        "period1": start_ts,
        "period2": end_ts,
        "interval": spec.yahoo_interval,
        "events": "history",
        "includeAdjustedClose": "true",
    }
    headers = {"User-Agent": "Mozilla/5.0"}

    try:
        session = requests.Session()
        if _has_dead_local_proxy():
            session.trust_env = False
        response = session.get(url, params=params, headers=headers, timeout=20)
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException as exc:
        raise DataValidationError(f"Yahoo Finance 連線失敗：{exc}") from exc
    except ValueError as exc:
        raise DataValidationError("Yahoo Finance 回傳資料格式無法解析。") from exc

    chart = payload.get("chart", {})
    error = chart.get("error")
    if error:
        description = error.get("description") or error.get("code") or "未知錯誤"
        raise DataValidationError(f"Yahoo Finance 回傳錯誤：{description}")

    results = chart.get("result") or []
    if not results:
        return pd.DataFrame()

    result = results[0]
    timestamps = result.get("timestamp") or []
    quote = ((result.get("indicators") or {}).get("quote") or [{}])[0]
    adjclose = ((result.get("indicators") or {}).get("adjclose") or [{}])[0].get("adjclose")
    if not timestamps or not quote:
        return pd.DataFrame()

    timezone = (result.get("meta") or {}).get("exchangeTimezoneName")
    dates = pd.to_datetime(timestamps, unit="s", utc=True)
    if timezone:
        dates = dates.tz_convert(timezone)
    dates = dates.tz_localize(None)

    frame = pd.DataFrame(
        {
            "Date": dates,
            "Open": quote.get("open"),
            "High": quote.get("high"),
            "Low": quote.get("low"),
            "Close": quote.get("close"),
            "Volume": quote.get("volume"),
        }
    )
    if adjclose:
        frame["Adj Close"] = adjclose
    return frame


def _has_dead_local_proxy() -> bool:
    proxy_keys = ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"]
    return any("127.0.0.1:9" in os.environ.get(key, "") for key in proxy_keys)


def _resample_prices(prices: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    rule = TIMEFRAME_SPECS[timeframe].resample_rule
    if rule is None:
        return prices

    resampled = prices.resample(rule).agg(
        {
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Adj Close": "last",
            "Volume": "sum",
        }
    )
    resampled = resampled.dropna(subset=["Open", "High", "Low", "Close"])
    if resampled.empty:
        raise DataValidationError(f"{timeframe} 重採樣後沒有可用價格資料。")
    return resampled


def load_csv_prices(file: BinaryIO, allow_weekends: bool) -> MarketData:
    try:
        raw = pd.read_csv(file)
    except pd.errors.EmptyDataError as exc:
        raise DataValidationError("CSV 檔案是空的，請上傳包含 K 線 OHLCV 的檔案。") from exc
    except Exception as exc:  # pragma: no cover - pandas gives many parser subclasses
        raise DataValidationError(f"CSV 讀取失敗：{exc}") from exc

    if raw.empty:
        raise DataValidationError("CSV 檔案沒有資料列。")

    prices = normalize_price_frame(raw, allow_weekends=allow_weekends)
    return MarketData(symbol="CSV 商品", prices=prices, source="CSV")


def normalize_price_frame(data: pd.DataFrame, allow_weekends: bool) -> pd.DataFrame:
    frame = data.copy()
    frame.columns = [str(col).strip() for col in frame.columns]

    if "Date" in frame.columns:
        frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
        frame = frame.dropna(subset=["Date"]).set_index("Date")
    else:
        frame.index = pd.to_datetime(frame.index, errors="coerce")
        frame = frame[~frame.index.isna()]

    missing = [col for col in REQUIRED_COLUMNS if col not in frame.columns]
    if missing:
        raise DataValidationError(f"CSV/價格資料缺少必要欄位：{', '.join(missing)}。")

    if "Adj Close" not in frame.columns:
        frame["Adj Close"] = frame["Close"]

    for col in OUTPUT_COLUMNS:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")

    frame = frame[OUTPUT_COLUMNS].sort_index()
    frame = frame[~frame.index.duplicated(keep="last")]
    frame = frame.dropna(subset=["Open", "High", "Low", "Close"])
    frame = frame[frame["Close"] > 0]

    if not allow_weekends:
        frame = frame[frame.index.dayofweek < 5]

    if frame.empty:
        raise DataValidationError("清理後沒有可用價格資料，請確認日期範圍與欄位內容。")

    return frame
