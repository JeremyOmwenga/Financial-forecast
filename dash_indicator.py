# Source: @DeepCharts Youtube Channel (https://www.youtube.com/@DeepCharts)

from datetime import datetime, timedelta

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import ta
import yfinance as yf


##########################################################################################
## PART 1: Define Functions for Pulling, Processing, and Creating Technical Indicators ##
##########################################################################################

@st.cache_data(ttl=60)
def fetch_stock_data(ticker, period, interval):
    """Download prices for a single ticker with flat (non-MultiIndex) columns."""
    if period == "1wk":
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)
        return yf.download(ticker, start=start_date, end=end_date, interval=interval,
                           multi_level_index=False, progress=False)
    return yf.download(ticker, period=period, interval=interval, multi_level_index=False, progress=False)


def process_data(data):
    """Make the index US/Eastern timezone-aware and expose it as a 'Datetime' column."""
    data = data.copy()
    if data.index.tzinfo is None:
        data.index = data.index.tz_localize("UTC")
    data.index = data.index.tz_convert("US/Eastern")
    data.reset_index(inplace=True)
    data.rename(columns={"Date": "Datetime"}, inplace=True)
    return data


def calculate_metrics(data):
    last_close = data["Close"].iloc[-1]
    prev_close = data["Close"].iloc[0]
    change = last_close - prev_close
    pct_change = (change / prev_close) * 100
    high = data["High"].max()
    low = data["Low"].min()
    volume = data["Volume"].sum()
    return last_close, change, pct_change, high, low, volume


def add_technical_indicators(data):
    data["SMA_20"] = ta.trend.sma_indicator(data["Close"], window=20)
    data["EMA_20"] = ta.trend.ema_indicator(data["Close"], window=20)
    return data


###############################################
## PART 2: Creating the Dashboard App layout ##
###############################################

st.title("Real Time Stock Dashboard")

# 2A: SIDEBAR PARAMETERS ############

st.sidebar.header("Chart Parameters")
ticker = st.sidebar.text_input("Ticker", "ADBE")
time_period = st.sidebar.selectbox("Time Period", ["1d", "1wk", "1mo", "1y", "max"])
chart_type = st.sidebar.selectbox("Chart Type", ["Candlestick", "Line"])
indicators = st.sidebar.multiselect("Technical Indicators", ["SMA 20", "EMA 20"])

# Mapping of time periods to data intervals
interval_mapping = {
    "1d": "1m",
    "1wk": "30m",
    "1mo": "1d",
    "1y": "1wk",
    "max": "1wk",
}

# 2B: MAIN CONTENT AREA ############

if st.sidebar.button("Update"):
    data = fetch_stock_data(ticker, time_period, interval_mapping[time_period])
    if data is None or data.empty:
        st.warning(f"No data returned for '{ticker}'.")
    else:
        data = add_technical_indicators(process_data(data))
        last_close, change, pct_change, high, low, volume = calculate_metrics(data)

        st.metric(label=f"{ticker} Last Price", value=f"{last_close:.2f} USD",
                  delta=f"{change:.2f} ({pct_change:.2f}%)")

        col1, col2, col3 = st.columns(3)
        col1.metric("High", f"{high:.2f} USD")
        col2.metric("Low", f"{low:.2f} USD")
        col3.metric("Volume", f"{volume:,}")

        if chart_type == "Candlestick":
            fig = go.Figure(go.Candlestick(x=data["Datetime"], open=data["Open"], high=data["High"],
                                           low=data["Low"], close=data["Close"]))
        else:
            fig = px.line(data, x="Datetime", y="Close")

        for indicator in indicators:
            column = indicator.replace(" ", "_")  # "SMA 20" -> "SMA_20"
            fig.add_trace(go.Scatter(x=data["Datetime"], y=data[column], name=indicator))

        fig.update_layout(title=f"{ticker} {time_period.upper()} Chart",
                          xaxis_title="Time", yaxis_title="Price (USD)", height=600)
        st.plotly_chart(fig, width="stretch")

        st.subheader("Historical Data")
        st.dataframe(data[["Datetime", "Open", "High", "Low", "Close", "Volume"]])

        st.subheader("Technical Indicators")
        st.dataframe(data[["Datetime", "SMA_20", "EMA_20"]])

# 2C: SIDEBAR PRICES ############

st.sidebar.header("Real-Time Stock Prices")
for symbol in ["AAPL", "GOOGL", "AMZN", "MSFT"]:
    real_time_data = fetch_stock_data(symbol, "1d", "1m")
    if real_time_data is None or real_time_data.empty:
        continue
    last_price = real_time_data["Close"].iloc[-1]
    change = last_price - real_time_data["Open"].iloc[0]
    pct_change = (change / real_time_data["Open"].iloc[0]) * 100
    st.sidebar.metric(f"{symbol}", f"{last_price:.2f} USD", f"{change:.2f} ({pct_change:.2f}%)")

st.sidebar.subheader("About")
st.sidebar.info("This dashboard provides stock data and technical indicators for various time periods. "
                "Use the sidebar to customize your view.")
