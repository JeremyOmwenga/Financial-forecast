import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler 


model = load_model('C:\\Users\\jerem\\OneDrive\\Desktop\\sav2\\Stock Predictions Model.keras')

st.header('Stock Price Predictor')

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