from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


MONTH_LABELS = ["1月", "2月", "3月", "4月", "5月", "6月", "7月", "8月", "9月", "10月", "11月", "12月"]


def price_chart(
    prices: pd.DataFrame,
    indicators: pd.DataFrame,
    events: pd.DataFrame,
    title: str,
) -> go.Figure:
    rows = 2 if any("RSI" in col for col in indicators.columns) else 1
    row_heights = [0.74, 0.26] if rows == 2 else [1.0]
    fig = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=row_heights,
    )

    fig.add_trace(
        go.Candlestick(
            x=prices.index,
            open=prices["Open"],
            high=prices["High"],
            low=prices["Low"],
            close=prices["Close"],
            name="價格",
            increasing_line_color="#16a34a",
            decreasing_line_color="#dc2626",
        ),
        row=1,
        col=1,
    )

    for col in indicators.columns:
        if "RSI" in col:
            fig.add_trace(go.Scatter(x=indicators.index, y=indicators[col], name=col, line={"color": "#7c3aed"}), row=2, col=1)
            fig.add_hline(y=70, line_dash="dash", line_color="#ef4444", row=2, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="#22c55e", row=2, col=1)
        else:
            fig.add_trace(go.Scatter(x=indicators.index, y=indicators[col], name=col, line={"width": 1.6}), row=1, col=1)

    if not events.empty:
        buys = events[events["Side"] == "Buy"]
        sells = events[events["Side"] == "Sell"]
        fig.add_trace(
            go.Scatter(
                x=buys["Date"],
                y=buys["Price"],
                mode="markers",
                marker={"symbol": "triangle-up", "size": 12, "color": "#059669"},
                name="買進",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=sells["Date"],
                y=sells["Price"],
                mode="markers",
                marker={"symbol": "triangle-down", "size": 12, "color": "#e11d48"},
                name="賣出",
            ),
            row=1,
            col=1,
        )

    fig.update_layout(
        title=title,
        height=640,
        margin={"l": 10, "r": 10, "t": 54, "b": 10},
        legend={"orientation": "h", "y": 1.02, "x": 0},
        xaxis_rangeslider_visible=False,
        template="plotly_white",
    )
    fig.update_yaxes(title_text="價格", row=1, col=1)
    if rows == 2:
        fig.update_yaxes(title_text="RSI", range=[0, 100], row=2, col=1)
    return fig


def equity_chart(equity: pd.DataFrame) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.65, 0.35])
    fig.add_trace(
        go.Scatter(x=equity.index, y=equity["Equity"], name="權益曲線", line={"color": "#2563eb", "width": 2.4}),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=equity.index,
            y=equity["Drawdown"],
            name="回撤",
            fill="tozeroy",
            line={"color": "#dc2626", "width": 1.6},
        ),
        row=2,
        col=1,
    )
    fig.update_yaxes(title_text="資產", row=1, col=1)
    fig.update_yaxes(title_text="回撤", tickformat=".0%", row=2, col=1)
    fig.update_layout(
        height=520,
        margin={"l": 10, "r": 10, "t": 34, "b": 10},
        template="plotly_white",
        legend={"orientation": "h", "y": 1.05, "x": 0},
    )
    return fig


def monthly_heatmap(heatmap: pd.DataFrame) -> go.Figure:
    values = heatmap.values if not heatmap.empty else [[]]
    years = heatmap.index.astype(str).tolist() if not heatmap.empty else []
    fig = go.Figure(
        data=go.Heatmap(
            z=values,
            x=MONTH_LABELS,
            y=years,
            colorscale=[
                [0, "#b91c1c"],
                [0.5, "#f8fafc"],
                [1, "#15803d"],
            ],
            zmid=0,
            colorbar={"title": "報酬"},
            hovertemplate="%{y} %{x}<br>報酬：%{z:.2%}<extra></extra>",
        )
    )
    fig.update_layout(
        height=max(260, 72 + 34 * max(len(years), 1)),
        margin={"l": 10, "r": 10, "t": 20, "b": 10},
        template="plotly_white",
    )
    return fig


def comparison_performance_chart(normalized: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    palette = ["#2563eb", "#16a34a", "#dc2626", "#9333ea", "#ea580c", "#0891b2"]
    for idx, symbol in enumerate(normalized.columns):
        fig.add_trace(
            go.Scatter(
                x=normalized.index,
                y=normalized[symbol],
                name=symbol,
                mode="lines",
                line={"width": 2.4, "color": palette[idx % len(palette)]},
                hovertemplate=f"{symbol}<br>%{{x|%Y-%m-%d}}<br>指數：%{{y:.2f}}<extra></extra>",
            )
        )
    fig.add_hline(y=100, line_dash="dash", line_color="#94a3b8")
    fig.update_layout(
        height=460,
        margin={"l": 10, "r": 10, "t": 28, "b": 10},
        template="plotly_white",
        legend={"orientation": "h", "y": 1.08, "x": 0},
        yaxis_title="起始日 = 100",
    )
    return fig


def comparison_metric_chart(metrics: pd.DataFrame) -> go.Figure:
    display = metrics[["總報酬", "CAGR", "年化波動", "Sharpe", "最大回撤"]].copy()
    percent_metrics = ["總報酬", "CAGR", "年化波動", "最大回撤"]
    for metric in percent_metrics:
        display[metric] = display[metric] * 100

    fig = make_subplots(
        rows=2,
        cols=3,
        subplot_titles=["總報酬 %", "CAGR %", "年化波動 %", "Sharpe", "最大回撤 %"],
        vertical_spacing=0.18,
        horizontal_spacing=0.1,
    )
    placements = {
        "總報酬": (1, 1),
        "CAGR": (1, 2),
        "年化波動": (1, 3),
        "Sharpe": (2, 1),
        "最大回撤": (2, 2),
    }
    colors = ["#2563eb", "#16a34a", "#dc2626", "#9333ea", "#ea580c", "#0891b2"]
    for metric, (row, col) in placements.items():
        fig.add_trace(
            go.Bar(
                x=display.index,
                y=display[metric],
                name=metric,
                marker_color=[colors[index % len(colors)] for index in range(len(display.index))],
                showlegend=False,
                text=[f"{value:.2f}" for value in display[metric]],
                textposition="outside",
                cliponaxis=False,
            ),
            row=row,
            col=col,
        )
    fig.update_layout(
        height=560,
        margin={"l": 10, "r": 10, "t": 44, "b": 10},
        template="plotly_white",
    )
    return fig


def correlation_heatmap(correlation: pd.DataFrame) -> go.Figure:
    symbols = correlation.columns.tolist()
    fig = go.Figure(
        data=go.Heatmap(
            z=correlation.values,
            x=symbols,
            y=symbols,
            zmin=-1,
            zmax=1,
            colorscale=[
                [0, "#2563eb"],
                [0.5, "#f8fafc"],
                [1, "#dc2626"],
            ],
            colorbar={"title": "相關係數"},
            text=correlation.round(2).astype(str).values,
            texttemplate="%{text}",
            hovertemplate="%{y} vs %{x}<br>相關係數：%{z:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        height=max(360, 70 + 42 * max(len(symbols), 1)),
        margin={"l": 10, "r": 10, "t": 24, "b": 10},
        template="plotly_white",
        xaxis={"side": "top"},
    )
    return fig
