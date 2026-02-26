import math
from dataclasses import dataclass

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

DEFAULTS = {
    "ticker": "BTC-USD",
    "period": "2y",
    "fast": 20,
    "slow": 50,
}


@dataclass
class BacktestResult:
    data: pd.DataFrame
    trades: pd.DataFrame
    total_return_pct: float
    max_drawdown_pct: float
    win_rate_pct: float | None
    num_trades: int
    cagr_pct: float | None


def download_price_data(ticker: str, period: str) -> pd.DataFrame:
    df = yf.download(ticker, period=period, interval="1d", progress=False, auto_adjust=False)
    if df is None or df.empty:
        raise ValueError("No data returned for this ticker/period.")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    price_col = "Adj Close" if "Adj Close" in df.columns else "Close"
    if price_col not in df.columns:
        raise ValueError("Downloaded data is missing both Adj Close and Close columns.")

    out = df[[price_col]].rename(columns={price_col: "price"}).copy()
    out.index = pd.to_datetime(out.index)
    return out


def extract_trades(df: pd.DataFrame) -> pd.DataFrame:
    trades: list[dict] = []
    in_position = False
    entry_date = None
    entry_price = None

    for idx, row in df.iterrows():
        if (not in_position) and row["entry"]:
            in_position = True
            entry_date = idx
            entry_price = row["price"]
        elif in_position and row["exit"]:
            exit_date = idx
            exit_price = row["price"]
            trades.append(
                {
                    "Entry Date": entry_date.date(),
                    "Exit Date": exit_date.date(),
                    "Entry Price": round(float(entry_price), 2),
                    "Exit Price": round(float(exit_price), 2),
                    "Return %": round((float(exit_price) / float(entry_price) - 1.0) * 100.0, 2),
                }
            )
            in_position = False
            entry_date = None
            entry_price = None

    if in_position and entry_date is not None and entry_price is not None:
        last_idx = df.index[-1]
        last_price = float(df.iloc[-1]["price"])
        trades.append(
            {
                "Entry Date": entry_date.date(),
                "Exit Date": last_idx.date(),
                "Entry Price": round(float(entry_price), 2),
                "Exit Price": round(last_price, 2),
                "Return %": round((last_price / float(entry_price) - 1.0) * 100.0, 2),
            }
        )

    return pd.DataFrame(trades)


def run_backtest(ticker: str, period: str, fast: int, slow: int) -> BacktestResult:
    base = download_price_data(ticker=ticker, period=period)

    df = base.copy()
    df["fast_sma"] = df["price"].rolling(fast).mean()
    df["slow_sma"] = df["price"].rolling(slow).mean()
    df["signal"] = (df["fast_sma"] > df["slow_sma"]).astype(int)
    df["daily_returns"] = df["price"].pct_change()
    df["strategy_returns"] = df["signal"].shift(1).fillna(0) * df["daily_returns"]

    # Drop rows with NA after indicators are computed
    df = df.dropna().copy()
    if df.empty:
        raise ValueError("Not enough data after indicator calculation. Try a longer period or smaller SMA lengths.")

    df["equity"] = (1 + df["strategy_returns"]).cumprod()
    df["rolling_max_equity"] = df["equity"].cummax()
    df["drawdown"] = df["equity"] / df["rolling_max_equity"] - 1

    signal_change = df["signal"].diff().fillna(0)
    df["entry"] = signal_change == 1
    df["exit"] = signal_change == -1

    trades = extract_trades(df)

    total_return_pct = (float(df["equity"].iloc[-1]) - 1.0) * 100.0
    max_drawdown_pct = float(df["drawdown"].min()) * 100.0

    num_trades = len(trades)
    if num_trades > 0:
        win_rate_pct = float((trades["Return %"] > 0).mean() * 100.0)
    else:
        win_rate_pct = None

    elapsed_days = max((df.index[-1] - df.index[0]).days, 1)
    years = elapsed_days / 365.25
    cagr_pct = None
    if years >= 1.0 and df["equity"].iloc[-1] > 0:
        cagr_pct = (math.pow(float(df["equity"].iloc[-1]), 1 / years) - 1.0) * 100.0

    return BacktestResult(
        data=df,
        trades=trades,
        total_return_pct=total_return_pct,
        max_drawdown_pct=max_drawdown_pct,
        win_rate_pct=win_rate_pct,
        num_trades=num_trades,
        cagr_pct=cagr_pct,
    )


def equity_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=df.index, y=df["equity"], mode="lines", name="Equity", line=dict(color="#1f77b4", width=2))
    )
    fig.update_layout(
        title="Equity Curve",
        xaxis_title="Date",
        yaxis_title="Equity",
        template="plotly_white",
        height=380,
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig


def price_chart(df: pd.DataFrame) -> go.Figure:
    entries = df[df["entry"]]
    exits = df[df["exit"]]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["price"], mode="lines", name="Price", line=dict(color="#111111", width=2)))
    fig.add_trace(
        go.Scatter(x=df.index, y=df["fast_sma"], mode="lines", name="Fast SMA", line=dict(color="#2ca02c", width=1.8))
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df["slow_sma"], mode="lines", name="Slow SMA", line=dict(color="#ff7f0e", width=1.8))
    )

    fig.add_trace(
        go.Scatter(
            x=entries.index,
            y=entries["price"],
            mode="markers",
            name="Entry",
            marker=dict(symbol="triangle-up", size=10, color="#2ca02c"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=exits.index,
            y=exits["price"],
            mode="markers",
            name="Exit",
            marker=dict(symbol="triangle-down", size=10, color="#d62728"),
        )
    )

    fig.update_layout(
        title="Price + SMA Signals",
        xaxis_title="Date",
        yaxis_title="Price",
        template="plotly_white",
        height=420,
        margin=dict(l=40, r=20, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig


def reset_defaults() -> None:
    st.session_state["ticker_input"] = DEFAULTS["ticker"]
    st.session_state["period_input"] = DEFAULTS["period"]
    st.session_state["fast_input"] = DEFAULTS["fast"]
    st.session_state["slow_input"] = DEFAULTS["slow"]


def main() -> None:
    st.set_page_config(page_title="Quant Strategy Backtester", layout="wide")
    st.title("Quant Strategy Backtester")
    st.caption("SMA crossover backtest using yfinance daily data.")

    if "ticker_input" not in st.session_state:
        reset_defaults()

    left, right = st.columns([1, 2.5], gap="large")

    with left:
        st.subheader("Inputs")
        ticker = st.text_input("Ticker", key="ticker_input")
        period = st.selectbox("Period", ["6mo", "1y", "2y", "5y"], key="period_input")
        fast = st.number_input("Fast SMA length", min_value=1, step=1, key="fast_input")
        slow = st.number_input("Slow SMA length", min_value=2, step=1, key="slow_input")

        invalid_sma = int(fast) >= int(slow)
        if invalid_sma:
            st.error("Fast SMA length must be smaller than Slow SMA length.")

        c1, c2 = st.columns(2)
        with c1:
            run_clicked = st.button("Run Backtest", type="primary", width="stretch", disabled=invalid_sma)
        with c2:
            if st.button("Reset to defaults", width="stretch"):
                reset_defaults()
                st.rerun()

    with right:
        if run_clicked:
            try:
                with st.spinner("Running backtest..."):
                    result = run_backtest(
                        ticker=ticker.strip(), period=period, fast=int(fast), slow=int(slow)
                    )

                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("Total Return %", f"{result.total_return_pct:.2f}%")
                m2.metric("Max Drawdown %", f"{result.max_drawdown_pct:.2f}%")
                win_text = "N/A" if result.win_rate_pct is None else f"{result.win_rate_pct:.2f}%"
                m3.metric("Win Rate %", win_text)
                m4.metric("Number of Trades", f"{result.num_trades}")
                cagr_text = "N/A" if result.cagr_pct is None else f"{result.cagr_pct:.2f}%"
                m5.metric("CAGR %", cagr_text)

                st.plotly_chart(equity_chart(result.data), width="stretch")
                st.plotly_chart(price_chart(result.data), width="stretch")

                st.subheader("Trades")
                if result.trades.empty:
                    st.info("No trades generated for this configuration.")
                else:
                    st.dataframe(result.trades, width="stretch", hide_index=True)

            except ValueError as e:
                st.error(f"Invalid ticker or no data returned: {e}")
            except Exception as e:  # network/API and unexpected failures
                st.error(f"Network/API failure or unexpected error: {e}")
        else:
            st.info("Set parameters and click **Run Backtest** to generate results.")


if __name__ == "__main__":
    main()
