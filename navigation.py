import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import matplotlib.pyplot as plt
import yfinance as yf
import financedatabase as fd
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import altair as alt
import datetime as dt
from sklearn.preprocessing import MinMaxScaler 


st.set_page_config(
	layout="wide",
	
	)

selected = option_menu(
	menu_title = "Main menu",
	options = ["Stock Forecast", "Portfolio & Savings", "Stock Details"],
	orientation = "horizontal",)

if selected == "Stock Forecast":
	model = load_model('C:\\Users\\jerem\\OneDrive\\Desktop\\sav2\\Stock Predictions Model.keras')

	st.header('Stock Forecast')

	stock = st.text_input('Enter Stock Symbol', 'GOOG')
	start = '2012-01-01'
	end = '2022-12-31'

	data = yf.download(stock, start, end)

	st.subheader('Stock Data')
	st.write(data)

	data_train = pd.DataFrame(data.Close[0: int(len(data)*0.8)])
	data_test = pd.DataFrame(data.Close[int(len(data)*0.8): len(data)])


	scaler = MinMaxScaler(feature_range = (0,1))

	pas_100_days = data_train.tail(100)
	data_test = pd.concat([pas_100_days, data_test], ignore_index = True)
	data_test_scale = scaler.fit_transform(data_test)

	st.subheader('Moving Average - 50D')
	ma_50_days = data.Close.rolling(50).mean()
	fig1 = plt.figure(figsize = (10,8))
	plt.plot(ma_50_days, 'r')
	plt.plot(data.Close, 'g')
	plt.show()
	st.pyplot(fig1)

	st.subheader('Price vs Moving Average - 50D vs Moving Average - 100D')
	ma_100_days = data.Close.rolling(100).mean()
	fig2 = plt.figure(figsize = (10,8))
	plt.plot(ma_50_days, 'r')
	plt.plot(ma_100_days, 'b')
	plt.plot(data.Close, 'g')
	plt.show()
	st.pyplot(fig2)

	st.subheader('Price vs Moving Average - 100D vs Moving Average - 200D')
	ma_200_days = data.Close.rolling(200).mean()
	fig3 = plt.figure(figsize = (10,8))
	plt.plot(ma_100_days, 'r')
	plt.plot(ma_200_days, 'b')
	plt.plot(data.Close, 'g')
	plt.show()
	st.pyplot(fig3)

	x = []
	y = []

	for i in range(100, data_test_scale.shape[0]):
	    x.append(data_test_scale[i-100:i])
	    y.append(data_test_scale[i,0])

	x, y = np.array(x), np.array(y)

	predict = model.predict(x)

	scale = 1/scaler.scale_

	predict = predict * scale

	y = y * scale

	st.subheader('Original Price vs Predicted Price')
	fig4 = plt.figure(figsize = (10,8))
	plt.plot(predict, 'r', label = 'Original Price')
	plt.plot(y, 'g', label = 'Predicted Price')
	plt.xlabel('Time')
	plt.ylabel('Price')
	plt.show()
	st.pyplot(fig4)

if selected == "Portfolio & Savings":


	# App title
	st.title("Portfolio Analysis")

	# Import ticker_list
	@st.cache_data
	def load_data():
	    # Pulling list of all EIFs and Equities from financedatabase
	    ticker_list = pd.concat([fd.ETFs().select().reset_index()[['symbol', 'name']],
	    fd.Equities().select().reset_index()[['symbol', 'name']]])
	    ticker_list = ticker_list[ticker_list.symbol.notna()]
	    ticker_list['symbol_name'] = ticker_list.symbol + ' - ' + ticker_list.name

	    return ticker_list
	ticker_list = load_data()

	# Side bar
	with st.sidebar:
	    # Portfolio builder
	    sel_tickers = st.multiselect('Portfolio Builder', placeholder="Search tickers", options=ticker_list.symbol_name)
	    sel_tickers_list = ticker_list[ticker_list.symbol_name.isin(sel_tickers)].symbol

	    cols = st.columns(4)
	    for i, ticker in enumerate(sel_tickers_list):
	        try:
	            cols[i % 4].image('https://logo.clearbit.com/' + yf.Ticker(ticker).info['website'].replace('https://www.', ''), width=65)
	        except:
	            cols[i % 4].subheader(ticker)

	    # Date selector
	    cols = st.columns(2)
	    sel_gt1 = cols[0].date_input('Start Date', value=dt.datetime(2024,1,1), format='YYYY-MM-DD')
	    sel_gt2 = cols[1].date_input('End Date', format='YYYY-MM-DD')

	    # Select tickers data
	    if len(sel_tickers) != 0:
	        yfdata = yf.download(list(sel_tickers_list), start=sel_gt1, end=sel_gt2)['Close'].reset_index().melt(id_vars=['Date'], var_name='ticker', value_name='price')
	        yfdata['price_start'] = yfdata.groupby('ticker').price.transform('first')
	        yfdata['price_pct_maily'] = yfdata.groupby('ticker').price.pct_change()
	        yfdata['price_pct'] = (yfdata.price - yfdata.price_start) / yfdata.price_start
	        
	# Tabs
	tab1, tab2 = st.tabs(['Portfolio', 'Calculator'])

	if len(sel_tickers) == 0:
	    st.info('Select tickers to view plots')
	else:
	    st.empty()
	    # Tab 1
	    #---
	    with tab1:
	        # All stocks plot
	        st.subheader('All Stocks')
	        fig = px.line(yfdata, x="Date", y="price_pct", color='ticker', markers=True)
	        fig.add_hline(y=0, line_dash='dash', line_color='white')
	        fig.update_layout(xaxis_title=None, yaxis_title=None)
	        fig.update_yaxes(tickformat='.,.0%')
	        st.plotly_chart(fig, use_container_width=True)

	        # Individual stock plots
	        # Add code for individual stock plots here

	        # Individual stock plots
	st.subheader('Individual Stock')
	cols = st.columns(3)
	for i, ticker in enumerate(sel_tickers_list):
	    # Adding logo
	    try:
	        cols[i % 3].image('https://logo.clearbit.com/' + yf.Ticker(ticker).info['website'].replace('https://www.', ''), width=65)
	    except:
	        cols[i % 3].subheader(ticker)

	    # Stock metrics
	    cols2 = cols[i % 3].columns(3)
	    ticker = 'Close' if len(sel_tickers_list) == 1 else ticker
	    cols2[0].metric(label='50-Day Average', value=round(yfdata[yfdata.ticker == ticker].price.tail(50).mean(),2))
	    cols2[1].metric(label='1-Year Low', value=round(yfdata[yfdata.ticker == ticker].price.tail(365).min(),2))
	    cols2[2].metric(label='1-Year High', value=round(yfdata[yfdata.ticker == ticker].price.tail(365).max(),2))

	    # Stock plot
	    fig = px.line(yfdata[yfdata.ticker == ticker], x="Date", y="price", markers=True)
	    fig.update_layout(xaxis_title=None, yaxis_title=None)
	    cols[i % 3].plotly_chart(fig, use_container_width=True)

	    # Tab 2
	#---
	with tab2:
	    # Amounts input
	    cols_tab2 = st.columns((0.2,0.8))
	    total_inv = 0
	    amounts = {}
	    for i, ticker in enumerate(sel_tickers_list):
	        cols = cols_tab2[0].columns((0.1,0.3))
	        try:
	            cols[0].image('https://logo.clearbit.com/' + yf.Ticker(ticker).info['website'].replace('https://www.', ''), width=65)
	        except:
	            cols[0].subheader(ticker)

	        amount = cols[1].number_input('', key=ticker, step=50)
	        total_inv = total_inv + amount
	        amounts[ticker] = amount

	# Investment goals
	cols_tab2[1].subheader('Total Investment: ' + str(total_inv))
	cols_goal = cols_tab2[1].columns((0.06,0.20,0.7))
	cols_goal[0].text('')
	cols_goal[0].subheader('Goal: ')
	goal = cols_goal[1].number_input('', key='goal', step=50)

	# Plot
	df = yfdata.copy()
	df['amount'] = df.ticker.map(amounts) * (1 + df.price_pct)

	dfsum = df.groupby('Date').amount.sum().reset_index()
	fig = px.area(df, x='Date', y='amount', color='ticker')
	fig.add_hline(y=goal, line_color='rgb(57,255,20)', line_dash='dash', line_width=3)
	if dfsum[dfsum.amount >= goal].shape[0] == 0:
	    cols_tab2[1].warning("The goal can't be reached within this time frame. Either change the goal amount or the time frame.")
	else:
	    fig.add_vline(x=dfsum[dfsum.amount >= goal].Date.iloc[0], line_color='rgb(57,255,20)', line_dash='dash', line_width=3)
	    fig.add_trace(go.Scatter(x=[dfsum[dfsum.amount >= goal].Date.iloc[0] + dt.timedelta(days=7)], y=[goal*1.1],
	    text=[dfsum[dfsum.amount >= goal].Date.dt.date.iloc[0]],
	    mode='text',
	    name='Goal',
	    textfont=dict(color='rgb(57,255,20)',
	    size=20)))
	fig.update_layout(xaxis_title=None, yaxis_title=None)
	cols_tab2[1].plotly_chart(fig, use_container_width=True)


if selected == "Stock Details":

	@st.cache_data
	def fetch_stock_info(symbol):
	    stock = yf.Ticker(symbol)
	    return stock.info

	@st.cache_data
	def fetch_quarterly_financials(symbol):
	    stock = yf.Ticker(symbol)
	    return stock.quarterly_financials.T

	@st.cache_data
	def fetch_annual_financials(symbol):
	    stock = yf.Ticker(symbol)
	    return stock.financials.T

	@st.cache_data
	def fetch_weekly_price_history(symbol):
	    stock = yf.Ticker(symbol)
	    return stock.history(period='1y', interval='1wk')

	st.title('Stock Dashboard')
	symbol = st.text_input('Enter a stock symbol', 'AAPL')

	information = fetch_stock_info(symbol)

	st.subheader(f'Name: {information["longName"]}')
	st.subheader(f'Market Cap: ${information["marketCap"]:,}')
	st.subheader(f'Sector: {information["sector"]}')

	price_history = fetch_weekly_price_history(symbol)

	st.header('Price Trend Chart')

	price_history = price_history.rename_axis('Date').reset_index()
	candle_stick_chart = go.Figure(data=[go.Candlestick(x=price_history['Date'],
	    open=price_history['Open'],
	    low=price_history['Low'],
	    high=price_history['High'],
	    close=price_history['Close'])])

	st.plotly_chart(candle_stick_chart, use_container_width=True)

	quarterly_financials = fetch_quarterly_financials(symbol)
	annual_financials = fetch_annual_financials(symbol)

	st.header('Financials')
	selection = st.segmented_control(label='Period', options=['Quarterly', 'Annual'], default='Quarterly')

	if selection == 'Quarterly':
	    quarterly_financials = quarterly_financials.rename_axis('Quarter').reset_index()
	    quarterly_financials['Quarter'] = quarterly_financials['Quarter'].astype(str)


	    revenue_chart = alt.Chart(quarterly_financials).mark_bar().encode(
	        x='Quarter',
	        y='Total Revenue'
	    )
	    st.altair_chart(revenue_chart, use_container_width=True)
	    





	if selection == 'Annual':
	    annual_financials = annual_financials.rename_axis('Year').reset_index()
	    annual_financials['Year'] = annual_financials['Year'].astype(str).transform(lambda year: year.split('-')[0])
	    revenue_chart = alt.Chart(annual_financials).mark_bar().encode(
	        x='Year',
	        y='Total Revenue'
	    )
	    st.altair_chart(revenue_chart, use_container_width=True)
