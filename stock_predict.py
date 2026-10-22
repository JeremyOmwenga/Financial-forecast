from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

MODEL_PATH = Path(__file__).parent / "Stock Predictions Model.keras"
WINDOW = 100  # days of history the model looks at per prediction


@st.cache_resource
def get_model():
    return load_model(MODEL_PATH)


@st.cache_data
def get_prices(symbol, start, end):
    return yf.download(symbol, start=start, end=end, multi_level_index=False, progress=False)


st.header("Stock Price Predictor")

stock = st.text_input("Enter Stock Symbol", "GOOG")
start = "2012-01-01"
end = "2022-12-31"

data = get_prices(stock, start, end)
if data is None or data.empty:
    st.warning(f"No price data found for '{stock}'.")
    st.stop()

st.subheader("Stock Data")
st.write(data)

split = int(len(data) * 0.8)
data_train = pd.DataFrame(data.Close[:split])
data_test = pd.DataFrame(data.Close[split:])

# Prepend the last WINDOW training days so the first test prediction has full history
data_test = pd.concat([data_train.tail(WINDOW), data_test], ignore_index=True)
scaler = MinMaxScaler(feature_range=(0, 1))
data_test_scale = scaler.fit_transform(data_test)

ma_50_days = data.Close.rolling(50).mean()
ma_100_days = data.Close.rolling(100).mean()
ma_200_days = data.Close.rolling(200).mean()


def plot_lines(*series):
    fig = plt.figure(figsize=(10, 8))
    for values, color in series:
        plt.plot(values, color)
    st.pyplot(fig)


st.subheader("Moving Average - 50D")
plot_lines((ma_50_days, "r"), (data.Close, "g"))

st.subheader("Price vs Moving Average - 50D vs Moving Average - 100D")
plot_lines((ma_50_days, "r"), (ma_100_days, "b"), (data.Close, "g"))

st.subheader("Price vs Moving Average - 100D vs Moving Average - 200D")
plot_lines((ma_100_days, "r"), (ma_200_days, "b"), (data.Close, "g"))

x = np.array([data_test_scale[i - WINDOW:i] for i in range(WINDOW, data_test_scale.shape[0])])
y = data_test_scale[WINDOW:, 0]

predict = scaler.inverse_transform(get_model().predict(x))
actual = scaler.inverse_transform(y.reshape(-1, 1))

st.subheader("Original Price vs Predicted Price")
fig4 = plt.figure(figsize=(10, 8))
plt.plot(actual, "g", label="Original Price")
plt.plot(predict, "r", label="Predicted Price")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
st.pyplot(fig4)
