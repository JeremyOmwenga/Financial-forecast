import datetime as dt

import financedatabase as fd
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

st.title("Portfolio Analysis")

# Yahoo exchange codes for US listings (Nasdaq, NYSE, NYSE American/Arca, Cboe BZX)
US_EXCHANGES = ["NMS", "NGM", "NCM", "NYQ", "ASE", "PCX", "BTS"]
US_MARKET = "United States"


@st.cache_data
def load_tickers():
    # All ETFs and equities known to financedatabase (~125k rows)
    ticker_list = pd.concat([
        fd.ETFs().select().reset_index()[["symbol", "name", "exchange"]],
        fd.Equities().select().reset_index()[["symbol", "name", "exchange"]],
    ])
    ticker_list = ticker_list[ticker_list.symbol.notna() & ticker_list.exchange.notna()]
    ticker_list["symbol_name"] = ticker_list.symbol + " - " + ticker_list.name.fillna("")
    return ticker_list


def tickers_for_market(ticker_list, market):
    # The full list is too large for the browser-side multiselect, so show one market at a time
    exchanges = US_EXCHANGES if market == US_MARKET else [market]
    return ticker_list[ticker_list.exchange.isin(exchanges)]


@st.cache_data
def load_prices(tickers, start, end):
    prices = yf.download(list(tickers), start=start, end=end, progress=False)["Close"]
    prices = prices.reset_index().melt(id_vars=["Date"], var_name="ticker", value_name="price")
    prices["price_start"] = prices.groupby("ticker").price.transform("first")
    prices["price_pct"] = (prices.price - prices.price_start) / prices.price_start
    return prices


ticker_list = load_tickers()

with st.sidebar:
    other_exchanges = sorted(set(ticker_list.exchange) - set(US_EXCHANGES))
    market = st.selectbox("Market", [US_MARKET] + other_exchanges, help="Other markets are listed by Yahoo exchange code.")
    market_tickers = tickers_for_market(ticker_list, market)

    sel_tickers = st.multiselect("Portfolio Builder", placeholder="Search tickers", options=market_tickers.symbol_name)
    sel_tickers_list = list(market_tickers[market_tickers.symbol_name.isin(sel_tickers)].symbol)

    cols = st.columns(2)
    sel_gt1 = cols[0].date_input("Start Date", value=dt.date(2024, 1, 1), format="YYYY-MM-DD")
    sel_gt2 = cols[1].date_input("End Date", format="YYYY-MM-DD")

if not sel_tickers_list:
    st.info("Select tickers to view plots")
    st.stop()

yfdata = load_prices(tuple(sel_tickers_list), sel_gt1, sel_gt2)

tab1, tab2 = st.tabs(["Portfolio", "Calculator"])

with tab1:
    st.subheader("All Stocks")
    fig = px.line(yfdata, x="Date", y="price_pct", color="ticker", markers=True)
    fig.add_hline(y=0, line_dash="dash", line_color="white")
    fig.update_layout(xaxis_title=None, yaxis_title=None)
    fig.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig, width="stretch")

    st.subheader("Individual Stock")
    cols = st.columns(3)
    for i, ticker in enumerate(sel_tickers_list):
        col = cols[i % 3]
        ticker_prices = yfdata[yfdata.ticker == ticker].price
        col.subheader(ticker)

        metrics = col.columns(3)
        metrics[0].metric(label="50-Day Average", value=round(ticker_prices.tail(50).mean(), 2))
        metrics[1].metric(label="1-Year Low", value=round(ticker_prices.tail(365).min(), 2))
        metrics[2].metric(label="1-Year High", value=round(ticker_prices.tail(365).max(), 2))

        fig = px.line(yfdata[yfdata.ticker == ticker], x="Date", y="price", markers=True)
        fig.update_layout(xaxis_title=None, yaxis_title=None)
        col.plotly_chart(fig, width="stretch")

with tab2:
    cols_tab2 = st.columns((0.2, 0.8))
    amounts = {}
    for ticker in sel_tickers_list:
        amounts[ticker] = cols_tab2[0].number_input(f"{ticker} amount", key=ticker, step=50)
    total_inv = sum(amounts.values())

    cols_tab2[1].subheader(f"Total Investment: {total_inv}")
    goal = cols_tab2[1].number_input("Goal", key="goal", step=50)

    df = yfdata.copy()
    df["amount"] = df.ticker.map(amounts) * (1 + df.price_pct)
    dfsum = df.groupby("Date").amount.sum().reset_index()
    reached = dfsum[dfsum.amount >= goal]

    fig = px.area(df, x="Date", y="amount", color="ticker")
    fig.add_hline(y=goal, line_color="rgb(57,255,20)", line_dash="dash", line_width=3)
    if reached.empty:
        cols_tab2[1].warning("The goal can't be reached within this time frame. Either change the goal amount or the time frame.")
    else:
        goal_date = reached.Date.iloc[0]
        fig.add_vline(x=goal_date, line_color="rgb(57,255,20)", line_dash="dash", line_width=3)
        fig.add_trace(go.Scatter(
            x=[goal_date + dt.timedelta(days=7)],
            y=[goal * 1.1],
            text=[goal_date.date()],
            mode="text",
            name="Goal",
            textfont=dict(color="rgb(57,255,20)", size=20),
        ))
    fig.update_layout(xaxis_title=None, yaxis_title=None)
    cols_tab2[1].plotly_chart(fig, width="stretch")
