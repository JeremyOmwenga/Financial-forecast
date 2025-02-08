# Financial Forecasting Model

## Overview
This project implements a financial forecasting model using Long Short-Term Memory (LSTM) Recurrent Neural Networks to forecast stock prices. LSTM proved the most accurate compared to Random Forest and Naive Bayes. The model is deployed through a Streamlit web application that provides an interactive interface for viewing stock information and forecasts.

## Features
- Real-time stock data fetching using Yahoo Finance API
- Historical price analysis and visualization
- LSTM-based price prediction model
- Technical indicators calculation (Moving Averages, RSI, MACD)
- Interactive charts and visualizations
- Custom date range selection for analysis
- Multiple stock symbol support
- Performance metrics dashboard

## Tech Stack
- **Python 3.8+**
- **Deep Learning**: TensorFlow, Keras
- **Data Processing**: Pandas, NumPy
- **Data Visualization**: Plotly
- **Web Application**: Streamlit
- **Stock Data**: yfinance
- **Version Control**: Git

## Installation

1. Clone the repository:
```bash
git clone https://github.com/JeremyOmwenga/Financial-forecasting.git
cd financial-forecasting
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the Streamlit application:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to `http://localhost:8501`

3. Enter a stock symbol and select your desired date range

4. View the analysis and forecasts


## Model Architecture
The LSTM model architecture consists of:
- Input layer with sequence length of 60, 120 days
- 2 LSTM layers with 50 units each
- Dropout layers (0.2) for regularization
- Dense output layer for final prediction

## Data Preprocessing
- Min-max scaling of features
- Sequence creation (60-day windows)
- Train-test split (80-20)
- Feature engineering including:
  - Technical indicators
  - Volume metrics
  - Price momentum features

## Performance Metrics
The model is evaluated using:
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Percentage Error (MAPE)
- R-squared (R²) score

## Contributing
1. Fork the repository
2. Create a new branch (`git checkout -b feature/improvement`)
3. Make your changes
4. Commit your changes (`git commit -am 'Add new feature'`)
5. Push to the branch (`git push origin feature/improvement`)
6. Create a Pull Request

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- Yahoo Finance for providing stock data
- Streamlit for the web application framework
- TensorFlow team for the deep learning tools


## Future Improvements
- Implementation of additional technical indicators
- Support for cryptocurrency price prediction
- Advanced portfolio optimization features
- Sentiment analysis integration
- Real-time model retraining
- Enhanced visualization options
- API endpoint for predictions
