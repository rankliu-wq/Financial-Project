from __future__ import annotations

import io

import pandas as pd
import streamlit as st

from backtester.charts import (
    comparison_metric_chart,
    comparison_performance_chart,
    correlation_heatmap,
    equity_chart,
    monthly_heatmap,
    price_chart,
)
from backtester.comparison import DEFAULT_COMPARISON_SYMBOLS, compare_symbols, correlation_pairs, parse_symbol_list
from backtester.data import DataValidationError, fetch_yahoo_prices, load_csv_prices
from backtester.engine import run_backtest
from backtester.metrics import monthly_returns, performance_summary
from backtester.strategies import STRATEGY_SPECS, ParamSpec, StrategyError, generate_strategy, strategy_names
from backtester.timeframes import DEFAULT_TIMEFRAME, TIMEFRAME_SPECS, annualization_for_timeframe, timeframe_labels


st.set_page_config(page_title="金融商品回測分析工具", page_icon="📈", layout="wide")


def inject_style() -> None:
    st.markdown(
        """
        <style>
        .main .block-container {
            padding-top: 1.6rem;
            max-width: 1320px;
        }
        [data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            padding: 14px 16px;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.05);
        }
        [data-testid="stMetricLabel"] {
            color: #475569;
        }
        .notice {
            border-left: 4px solid #2563eb;
            background: #eff6ff;
            color: #1e3a8a;
            padding: 12px 14px;
            border-radius: 6px;
            margin: 0.35rem 0 1rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def pct(value: float) -> str:
    return f"{value:.2%}"


def money(value: float) -> str:
    return f"{value:,.0f}"


def render_strategy_param(param: ParamSpec):
    if param.kind == "int":
        return st.sidebar.number_input(
            param.label,
            min_value=int(param.min_value),
            max_value=int(param.max_value),
            value=int(param.default),
            step=int(param.step or 1),
            help=param.help,
        )
    if param.kind == "float":
        return st.sidebar.number_input(
            param.label,
            min_value=float(param.min_value),
            max_value=float(param.max_value),
            value=float(param.default),
            step=float(param.step or 0.1),
            format=param.format,
            help=param.help,
        )
    if param.kind == "select":
        return st.sidebar.selectbox(
            param.label,
            options=list(param.options),
            index=list(param.options).index(param.default),
            help=param.help,
        )
    raise StrategyError(f"不支援的策略參數型態：{param.kind}")


def sidebar_inputs() -> dict:
    st.sidebar.header("回測設定")
    data_source = st.sidebar.radio("資料來源", ["Yahoo Finance", "CSV 上傳"], horizontal=True)
    timeframe = st.sidebar.selectbox(
        "時間週期",
        timeframe_labels(),
        index=timeframe_labels().index(DEFAULT_TIMEFRAME),
        help="CSV 模式會將上傳資料視為已經是此週期；Yahoo 模式會依週期下載或重採樣。",
    )
    allow_weekends = st.sidebar.checkbox("允許週末資料", value=True, help="加密貨幣通常會有週末資料；股票與 ETF 可關閉。")
    timeframe_note = TIMEFRAME_SPECS[timeframe].note
    if timeframe_note:
        st.sidebar.caption(timeframe_note)

    today = pd.Timestamp.today().normalize()
    default_start = today - pd.Timedelta(days=TIMEFRAME_SPECS[timeframe].default_lookback_days)
    date_range = st.sidebar.date_input("分析日期", value=(default_start.date(), today.date()))
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = [pd.Timestamp(item) for item in date_range]
    else:
        start_date, end_date = default_start, today

    symbol = "SPY"
    uploaded_file = None
    if data_source == "Yahoo Finance":
        symbol = st.sidebar.text_input("商品代號", value="SPY", help="範例：AAPL、SPY、2330.TW、BTC-USD")
    else:
        uploaded_file = st.sidebar.file_uploader("上傳 CSV", type=["csv"])

    comparison_raw = st.sidebar.text_area(
        "比較標的",
        value=", ".join(DEFAULT_COMPARISON_SYMBOLS),
        help="以逗號、空白或換行分隔。比較模組使用 Yahoo Finance 資料。",
        height=78,
    )

    strategy_name = st.sidebar.selectbox("策略", strategy_names())
    strategy_spec = STRATEGY_SPECS[strategy_name]
    params = {param.key: render_strategy_param(param) for param in strategy_spec.params}

    st.sidebar.divider()
    initial_cash = st.sidebar.number_input("初始資金", min_value=1000.0, value=1_000_000.0, step=10_000.0)
    commission_rate = st.sidebar.number_input("手續費率", min_value=0.0, max_value=0.05, value=0.001, step=0.0005, format="%.4f")
    slippage_rate = st.sidebar.number_input("滑價率", min_value=0.0, max_value=0.05, value=0.0005, step=0.0005, format="%.4f")

    return {
        "data_source": data_source,
        "timeframe": timeframe,
        "allow_weekends": allow_weekends,
        "start_date": start_date,
        "end_date": end_date,
        "symbol": symbol,
        "uploaded_file": uploaded_file,
        "comparison_symbols": parse_symbol_list(comparison_raw),
        "strategy_name": strategy_name,
        "strategy_params": params,
        "initial_cash": float(initial_cash),
        "commission_rate": float(commission_rate),
        "slippage_rate": float(slippage_rate),
    }


@st.cache_data(show_spinner=False)
def get_yahoo_data(symbol: str, start_date: pd.Timestamp, end_date: pd.Timestamp, allow_weekends: bool, timeframe: str):
    return fetch_yahoo_prices(symbol, start_date, end_date, allow_weekends, timeframe)


@st.cache_data(show_spinner=False)
def get_comparison_data(
    symbols: tuple[str, ...],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    allow_weekends: bool,
    annualization: int,
    timeframe: str,
):
    return compare_symbols(list(symbols), start_date, end_date, allow_weekends, annualization, timeframe)


def get_csv_data(uploaded_file, allow_weekends: bool):
    if uploaded_file is None:
        raise DataValidationError("請先上傳 CSV 檔案。")
    return load_csv_prices(io.BytesIO(uploaded_file.getvalue()), allow_weekends)


def show_metrics(summary: dict[str, float | int]) -> None:
    cols = st.columns(4)
    cols[0].metric("期末資產", money(float(summary["ending_equity"])))
    cols[1].metric("總報酬", pct(float(summary["total_return"])))
    cols[2].metric("年化報酬 CAGR", pct(float(summary["cagr"])))
    cols[3].metric("最大回撤", pct(float(summary["max_drawdown"])))

    cols = st.columns(4)
    cols[0].metric("年化波動", pct(float(summary["volatility"])))
    cols[1].metric("Sharpe", f"{float(summary['sharpe']):.2f}")
    cols[2].metric("勝率", pct(float(summary["win_rate"])))
    cols[3].metric("交易次數", f"{int(summary['trade_count'])}")


def show_comparison(result) -> None:
    st.caption(f"共同比較期間：{result.start.date()} 至 {result.end.date()}，共 {len(result.normalized):,} 個共同資料點")

    best_rows = []
    judgement = {
        "總報酬": "越高越好",
        "CAGR": "越高越好",
        "年化波動": "越低越好",
        "Sharpe": "越高越好",
        "最大回撤": "越接近 0 越好",
        "期末指數": "越高越好",
    }
    for metric in ["總報酬", "CAGR", "年化波動", "Sharpe", "最大回撤"]:
        best_rows.append(
            {
                "比較項目": metric,
                "較佳標的": result.best_symbols[metric],
                "判斷方式": judgement[metric],
            }
        )

    left, right = st.columns([0.58, 0.42])
    with left:
        st.plotly_chart(comparison_performance_chart(result.normalized), use_container_width=True)
    with right:
        st.dataframe(pd.DataFrame(best_rows), use_container_width=True, hide_index=True)

    display = result.metrics.copy()
    for column in ["總報酬", "CAGR", "年化波動", "最大回撤"]:
        display[f"{column} %"] = display[column] * 100
    display = display[["總報酬 %", "CAGR %", "年化波動 %", "Sharpe", "最大回撤 %", "期末指數"]]
    st.dataframe(
        display,
        use_container_width=True,
        column_config={
            "總報酬 %": st.column_config.NumberColumn("總報酬 %", format="%.2f%%"),
            "CAGR %": st.column_config.NumberColumn("CAGR %", format="%.2f%%"),
            "年化波動 %": st.column_config.NumberColumn("年化波動 %", format="%.2f%%"),
            "Sharpe": st.column_config.NumberColumn("Sharpe", format="%.2f"),
            "最大回撤 %": st.column_config.NumberColumn("最大回撤 %", format="%.2f%%"),
            "期末指數": st.column_config.NumberColumn("期末指數", format="%.2f"),
        },
    )
    st.plotly_chart(comparison_metric_chart(result.metrics), use_container_width=True)

    st.subheader("標的相關係數")
    st.caption("以各標的在共同期間內的報酬率計算；越接近 1 代表走勢越同向，越接近 -1 代表越反向。")
    corr_left, corr_right = st.columns([0.58, 0.42])
    with corr_left:
        st.plotly_chart(correlation_heatmap(result.correlation), use_container_width=True)
    with corr_right:
        st.dataframe(
            result.correlation.round(3),
            use_container_width=True,
            column_config={
                column: st.column_config.NumberColumn(column, format="%.3f")
                for column in result.correlation.columns
            },
        )

    pairwise = correlation_pairs(result.correlation)
    if len(pairwise) > 3:
        with st.expander("查看各標的兩兩相關係數表"):
            st.dataframe(
                pairwise,
                use_container_width=True,
                hide_index=True,
                column_config={"相關係數": st.column_config.NumberColumn("相關係數", format="%.3f")},
            )

    if result.warnings:
        st.warning("部分標的無法納入比較：" + "；".join(result.warnings))


def main() -> None:
    inject_style()
    inputs = sidebar_inputs()

    st.title("金融商品回測分析工具")
    st.markdown(
        '<div class="notice">本工具用於研究與教育用途，回測結果不代表未來績效，也不構成投資建議。</div>',
        unsafe_allow_html=True,
    )

    try:
        if inputs["start_date"] >= inputs["end_date"]:
            raise DataValidationError("開始日期必須早於結束日期。")

        with st.spinner("讀取價格資料中..."):
            if inputs["data_source"] == "Yahoo Finance":
                market_data = get_yahoo_data(
                    inputs["symbol"],
                    inputs["start_date"],
                    inputs["end_date"],
                    inputs["allow_weekends"],
                    inputs["timeframe"],
                )
            else:
                market_data = get_csv_data(inputs["uploaded_file"], inputs["allow_weekends"])

        strategy = generate_strategy(
            market_data.prices,
            inputs["strategy_name"],
            **inputs["strategy_params"],
        )
        result = run_backtest(
            market_data.prices,
            strategy.signal,
            inputs["initial_cash"],
            inputs["commission_rate"],
            inputs["slippage_rate"],
        )
        annualization = annualization_for_timeframe(inputs["timeframe"], inputs["allow_weekends"])
        summary = performance_summary(result.equity, result.trades, inputs["initial_cash"], annualization)
        heatmap = monthly_returns(result.equity)

    except (DataValidationError, StrategyError, ValueError) as exc:
        st.error(str(exc))
        st.stop()
    except Exception as exc:
        st.exception(exc)
        st.stop()

    header_left, header_right = st.columns([0.72, 0.28])
    header_left.subheader(f"{market_data.symbol} · {strategy.name}")
    header_right.caption(
        f"{market_data.source}｜{inputs['timeframe']}｜{market_data.prices.index.min()} 至 {market_data.prices.index.max()}｜{len(market_data.prices):,} 根 K 線"
    )

    show_metrics(summary)

    tab_price, tab_equity, tab_monthly, tab_compare, tab_trades, tab_data = st.tabs(
        ["價格與訊號", "權益與回撤", "月報酬", "標的比較", "交易紀錄", "資料預覽"]
    )
    with tab_price:
        st.plotly_chart(
            price_chart(market_data.prices, strategy.indicators, result.events, f"{market_data.symbol} 價格與交易訊號"),
            use_container_width=True,
        )
    with tab_equity:
        st.plotly_chart(equity_chart(result.equity), use_container_width=True)
    with tab_monthly:
        if heatmap.empty:
            st.info("資料期間不足以計算月報酬。")
        else:
            st.plotly_chart(monthly_heatmap(heatmap), use_container_width=True)
    with tab_compare:
        try:
            with st.spinner("讀取比較標的資料中..."):
                comparison = get_comparison_data(
                    tuple(inputs["comparison_symbols"]),
                    inputs["start_date"],
                    inputs["end_date"],
                    inputs["allow_weekends"],
                    annualization,
                    inputs["timeframe"],
                )
            show_comparison(comparison)
        except DataValidationError as exc:
            st.error(str(exc))
    with tab_trades:
        if result.trades.empty:
            st.info("這段期間沒有已完成交易。")
        else:
            trades = result.trades.copy()
            trades["Entry Date"] = pd.to_datetime(trades["Entry Date"]).dt.date
            trades["Exit Date"] = pd.to_datetime(trades["Exit Date"]).dt.date
            trades["Return %"] = trades["Return"] * 100
            trades = trades.drop(columns=["Return"])
            st.dataframe(
                trades,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Entry Price": st.column_config.NumberColumn("Entry Price", format="%.2f"),
                    "Exit Price": st.column_config.NumberColumn("Exit Price", format="%.2f"),
                    "Shares": st.column_config.NumberColumn("Shares", format="%.4f"),
                    "PnL": st.column_config.NumberColumn("PnL", format="%.2f"),
                    "Return %": st.column_config.NumberColumn("Return %", format="%.2f%%"),
                },
            )
            st.download_button(
                "下載交易紀錄 CSV",
                data=result.trades.to_csv(index=False).encode("utf-8-sig"),
                file_name=f"{market_data.symbol}_trades.csv",
                mime="text/csv",
            )
    with tab_data:
        st.dataframe(market_data.prices.tail(300), use_container_width=True)
        st.download_button(
            "下載清理後價格資料 CSV",
            data=market_data.prices.to_csv().encode("utf-8-sig"),
            file_name=f"{market_data.symbol}_prices.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()
