from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class BacktestResult:
    equity: pd.DataFrame
    trades: pd.DataFrame
    events: pd.DataFrame


def run_backtest(
    prices: pd.DataFrame,
    signal: pd.Series,
    initial_cash: float,
    commission_rate: float,
    slippage_rate: float,
) -> BacktestResult:
    if initial_cash <= 0:
        raise ValueError("初始資金必須大於 0。")
    if commission_rate < 0 or slippage_rate < 0:
        raise ValueError("交易成本不可為負數。")

    signal = signal.reindex(prices.index).fillna(0).astype(int).clip(0, 1)
    executable_signal = signal.shift(1).fillna(0).astype(int)

    cash = float(initial_cash)
    shares = 0.0
    entry_date = None
    entry_price = None
    entry_cost = None
    rows = []
    events = []
    trades = []

    for date, row in prices.iterrows():
        close = float(row["Close"])
        target = int(executable_signal.loc[date])

        if target == 1 and shares == 0:
            execution_price = close * (1 + slippage_rate)
            shares = cash / (execution_price * (1 + commission_rate))
            gross_cost = shares * execution_price
            commission = gross_cost * commission_rate
            cash -= gross_cost + commission
            entry_date = date
            entry_price = execution_price
            entry_cost = gross_cost + commission
            events.append(
                {
                    "Date": date,
                    "Side": "Buy",
                    "Price": execution_price,
                    "Shares": shares,
                    "Commission": commission,
                }
            )

        elif target == 0 and shares > 0:
            execution_price = close * (1 - slippage_rate)
            gross_proceeds = shares * execution_price
            commission = gross_proceeds * commission_rate
            net_proceeds = gross_proceeds - commission
            cash += net_proceeds
            pnl = net_proceeds - float(entry_cost)
            trades.append(
                {
                    "Entry Date": entry_date,
                    "Exit Date": date,
                    "Entry Price": entry_price,
                    "Exit Price": execution_price,
                    "Shares": shares,
                    "PnL": pnl,
                    "Return": pnl / float(entry_cost),
                    "Holding Days": (date - entry_date).days if entry_date is not None else 0,
                }
            )
            events.append(
                {
                    "Date": date,
                    "Side": "Sell",
                    "Price": execution_price,
                    "Shares": shares,
                    "Commission": commission,
                }
            )
            shares = 0.0
            entry_date = None
            entry_price = None
            entry_cost = None

        equity = cash + shares * close
        rows.append(
            {
                "Date": date,
                "Cash": cash,
                "Shares": shares,
                "Position": 1 if shares > 0 else 0,
                "Close": close,
                "Equity": equity,
                "Signal": int(signal.loc[date]),
                "Executable Signal": target,
            }
        )

    equity_frame = pd.DataFrame(rows).set_index("Date")
    equity_frame["Returns"] = equity_frame["Equity"].pct_change().fillna(0)
    equity_frame["Peak"] = equity_frame["Equity"].cummax()
    equity_frame["Drawdown"] = equity_frame["Equity"] / equity_frame["Peak"] - 1

    return BacktestResult(
        equity=equity_frame,
        trades=pd.DataFrame(trades),
        events=pd.DataFrame(events),
    )
