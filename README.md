# Cryptocurrency Analysis Dashboard

A comprehensive web application for cryptocurrency price analysis, technical indicators, sentiment analysis, and price prediction.

## Features

- Real-time price data from Yahoo Finance
- Technical indicators: RSI, SMA, MACD, Bollinger Bands
- Price prediction using linear regression
- News sentiment analysis from Google News (English & Turkish)
- Crypto Fear & Greed Index integration
- Interactive charts with Plotly

## Installation

1. Install required packages:
```bash
pip install -r requirements_crypto.txt
```

2. Run the application:
```bash
streamlit run crypto_prediction_app.py
```

The application will open in your browser at `http://localhost:8501`

## Usage

1. Select a cryptocurrency from the sidebar
2. Choose time period for historical data
3. Enable technical indicators as needed
4. Enable news sentiment analysis to see market sentiment
5. View price predictions and technical analysis

## Technical Details

### Technologies Used
- Streamlit - Web interface
- YFinance - Financial data fetching
- Pandas - Data processing
- Plotly - Interactive charts
- Scikit-learn - Machine learning (prediction model)
- TextBlob - Sentiment analysis
- PyGoogleNews - News fetching
- NLTK - Natural language processing

### Technical Indicators
- **RSI**: 14-day default period (adjustable)
  - > 70: Overbought
  - < 30: Oversold

- **SMA**: Simple moving average
  - Short SMA > Long SMA: Uptrend
  - Short SMA < Long SMA: Downtrend

- **MACD**: Trend following indicator
  - MACD > Signal: Bullish signal
  - MACD < Signal: Bearish signal

### Prediction Model
The application uses a simple Linear Regression model based on the last 30 days of price data to predict future prices. This model is for educational purposes and does not guarantee actual market movements.

## Disclaimer

This application is for educational and analysis purposes only. 

- This is NOT investment advice
- Cryptocurrency investments carry high risk
- Make investment decisions at your own risk
- Past performance does not guarantee future results

## License

This project is open source and available for educational purposes.

