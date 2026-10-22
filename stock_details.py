import altair as alt
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf


@st.cache_data
def fetch_stock_info(symbol):
    return yf.Ticker(symbol).info


@st.cache_data
def fetch_quarterly_financials(symbol):
    return yf.Ticker(symbol).quarterly_financials.T


@st.cache_data
def fetch_annual_financials(symbol):
    return yf.Ticker(symbol).financials.T


@st.cache_data
def fetch_weekly_price_history(symbol):
    return yf.Ticker(symbol).history(period="1y", interval="1wk")


st.title("Stock Dashboard")
symbol = st.text_input("Enter a stock symbol", "AAPL")

information = fetch_stock_info(symbol)
st.subheader(f'Name: {information.get("longName", symbol)}')
st.subheader(f'Market Cap: ${information.get("marketCap", 0):,}')
st.subheader(f'Sector: {information.get("sector", "N/A")}')

st.header("Price Trend Chart")
price_history = fetch_weekly_price_history(symbol).rename_axis("Date").reset_index()
candle_stick_chart = go.Figure(data=[go.Candlestick(
    x=price_history["Date"],
    open=price_history["Open"],
    low=price_history["Low"],
    high=price_history["High"],
    close=price_history["Close"],
)])
st.plotly_chart(candle_stick_chart, width="stretch")

st.header("Financials")
selection = st.segmented_control(label="Period", options=["Quarterly", "Annual"], default="Quarterly")

if selection == "Quarterly":
    financials = fetch_quarterly_financials(symbol).rename_axis("Quarter").reset_index()
    financials["Quarter"] = financials["Quarter"].astype(str)
    x_field = "Quarter"
else:
    financials = fetch_annual_financials(symbol).rename_axis("Year").reset_index()
    financials["Year"] = financials["Year"].astype(str).str.split("-").str[0]
    x_field = "Year"

revenue_chart = alt.Chart(financials).mark_bar().encode(x=x_field, y="Total Revenue")
st.altair_chart(revenue_chart, width="stretch")
