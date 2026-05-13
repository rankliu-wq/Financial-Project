from __future__ import annotations

import io

import pandas as pd
import pytest

from backtester.comparison import compare_symbols, parse_symbol_list
from backtester.data import DataValidationError, fetch_yahoo_prices, load_csv_prices, normalize_price_frame
from backtester.engine import run_backtest
from backtester.metrics import performance_summary
from backtester.strategies import generate_strategy, strategy_names
from backtester.timeframes import annualization_for_timeframe


def sample_prices(days: int = 120) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=days, freq="D")
    close = pd.Series(range(100, 100 + days), index=index, dtype=float)
    return pd.DataFrame(
        {
            "Open": close - 0.5,
            "High": close + 1,
            "Low": close - 1,
            "Close": close,
            "Volume": 1000,
        },
        index=index,
    )


def test_csv_loader_accepts_ohlcv() -> None:
    csv = b"Date,Open,High,Low,Close,Volume\n2024-01-01,10,11,9,10.5,100\n"
    data = load_csv_prices(io.BytesIO(csv), allow_weekends=True)
    assert data.prices.iloc[0]["Close"] == 10.5
    assert "Adj Close" in data.prices.columns


def test_csv_loader_reports_missing_columns() -> None:
    csv = b"Date,Open,High,Low,Close\n2024-01-01,10,11,9,10.5\n"
    with pytest.raises(DataValidationError, match="Volume"):
        load_csv_prices(io.BytesIO(csv), allow_weekends=True)


def test_weekend_filter_can_be_disabled() -> None:
    prices = normalize_price_frame(sample_prices(7), allow_weekends=False)
    assert prices.index.dayofweek.max() < 5


def test_buy_and_hold_backtest_finishes_with_position() -> None:
    prices = sample_prices()
    strategy = generate_strategy(prices, "買入持有")
    result = run_backtest(prices, strategy.signal, 100_000, 0.001, 0.0005)
    summary = performance_summary(result.equity, result.trades, 100_000, 365)
    assert result.equity["Equity"].iloc[-1] > 100_000
    assert summary["trade_count"] == 0


def test_sma_strategy_generates_signal() -> None:
    prices = sample_prices()
    strategy = generate_strategy(prices, "均線交叉", short_window=5, long_window=20)
    assert strategy.signal.sum() > 0
    assert "SMA 5" in strategy.indicators.columns


def test_rsi_strategy_returns_aligned_signal() -> None:
    prices = sample_prices()
    strategy = generate_strategy(prices, "RSI 反轉", rsi_window=14, rsi_buy=30, rsi_sell=70)
    assert strategy.signal.index.equals(prices.index)


def test_bollinger_strategy_generates_bands_and_signal() -> None:
    index = pd.date_range("2024-01-01", periods=45, freq="D")
    close = pd.Series([100.0] * 24 + [70.0, 72.0, 78.0, 86.0, 94.0, 101.0] + [102.0] * 15, index=index)
    prices = pd.DataFrame(
        {
            "Open": close,
            "High": close + 1,
            "Low": close - 1,
            "Close": close,
            "Volume": 1000,
        },
        index=index,
    )
    strategy = generate_strategy(prices, "布林通道", bb_window=20, bb_std=2.0, bb_exit="中軌出場")
    assert strategy.signal.sum() > 0
    assert "BB 中軌 20" in strategy.indicators.columns
    assert "布林通道" in strategy_names()


def test_parse_symbol_list_deduplicates_and_accepts_multiple_separators() -> None:
    assert parse_symbol_list("0050.tw, 0056.TW\n2330.tw 0050.TW") == ["0050.TW", "0056.TW", "2330.TW"]


def test_compare_symbols_uses_common_dates(monkeypatch) -> None:
    prices_a = sample_prices(40)
    prices_b = sample_prices(45).iloc[5:]

    def fake_fetch(symbol, start, end, allow_weekends, timeframe):
        frame = prices_a if symbol == "AAA" else prices_b
        frame = frame.copy()
        frame["Adj Close"] = frame["Close"]
        return type("MarketData", (), {"prices": frame})()

    monkeypatch.setattr("backtester.comparison.fetch_yahoo_prices", fake_fetch)
    result = compare_symbols(["AAA", "BBB"], pd.Timestamp("2024-01-01"), pd.Timestamp("2024-02-29"), True, 365, "日線")
    assert list(result.metrics.index) == ["AAA", "BBB"]
    assert result.normalized.index.min() == prices_b.index.min()
    assert "總報酬" in result.best_symbols


def test_four_hour_timeframe_resamples_hourly_prices(monkeypatch) -> None:
    hourly = sample_prices(12)
    hourly.index = pd.date_range("2024-01-02 09:00", periods=12, freq="h")

    def fake_download(symbol, start, end, timeframe):
        assert timeframe == "4 小時"
        return hourly

    monkeypatch.setattr("backtester.data._download_with_yahoo_chart_api", fake_download)
    monkeypatch.setattr("backtester.data._download_with_yfinance", fake_download)
    data = fetch_yahoo_prices("TEST", pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03"), True, "4 小時")
    assert len(data.prices) == 4
    assert data.prices.iloc[0]["Open"] == hourly.iloc[0]["Open"]
    assert data.prices.iloc[0]["Close"] == hourly.iloc[2]["Close"]


def test_annualization_changes_by_timeframe_and_weekend_mode() -> None:
    assert annualization_for_timeframe("日線", allow_weekends=False) == 252
    assert annualization_for_timeframe("日線", allow_weekends=True) == 365
    assert annualization_for_timeframe("5 分鐘", allow_weekends=False) > annualization_for_timeframe("月線", allow_weekends=False)
