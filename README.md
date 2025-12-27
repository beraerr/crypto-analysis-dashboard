# Cryptocurrency Analysis Dashboard

A comprehensive web application for cryptocurrency price analysis, technical indicators, sentiment analysis, and price prediction.

## Features

- **Real-time price data** from Yahoo Finance
- **Technical indicators**: RSI, SMA, MACD, Bollinger Bands
- **🤖 Advanced AI Price Prediction**:
  - LSTM (Long Short-Term Memory) - Captures long-term dependencies
  - GRU (Gated Recurrent Unit) - Lightweight and efficient
  - Transformer - Attention-based model for long-range patterns
  - Hybrid Transformer-GRU - State-of-the-art combination
  - Ensemble Model - Combines all models for best accuracy
- **🧠 Advanced Sentiment Analysis**:
  - FinBERT - Financial domain-specific BERT model
  - Multi-source sentiment fusion with weighted scoring
  - News sentiment analysis from Google News (English & Turkish)
- **Crypto Fear & Greed Index** integration
- **Interactive charts** with Plotly

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
- **Streamlit** - Web interface
- **YFinance** - Financial data fetching
- **Pandas** - Data processing
- **Plotly** - Interactive charts
- **PyTorch** - Deep learning framework (LSTM, GRU models)
- **TensorFlow/Keras** - Deep learning framework (Transformer, Hybrid models)
- **Transformers (HuggingFace)** - FinBERT for financial sentiment analysis
- **Scikit-learn** - Machine learning utilities
- **TextBlob** - Fallback sentiment analysis
- **PyGoogleNews** - News fetching
- **NLTK** - Natural language processing

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

### AI Prediction Models

The application now includes **state-of-the-art deep learning models** for cryptocurrency price prediction:

1. **LSTM (Long Short-Term Memory)**
   - Captures long-term dependencies in price sequences
   - Uses 60-day lookback window
   - Multi-layer architecture with dropout regularization

2. **GRU (Gated Recurrent Unit)**
   - Lighter alternative to LSTM
   - Faster training while maintaining good performance
   - Effective for volatile markets

3. **Transformer Model**
   - Attention mechanism captures long-range dependencies
   - Better understanding of price patterns across time
   - Multi-head attention for complex pattern recognition

4. **Hybrid Transformer-GRU**
   - Combines Transformer's attention with GRU's sequential processing
   - State-of-the-art approach from recent research
   - Best balance of accuracy and efficiency

5. **Ensemble Model**
   - Combines predictions from all models
   - Weighted averaging for optimal results
   - Most robust and accurate predictions

**Note:** All models use multiple features (Close, High, Low, Volume) and are trained on historical data. Models are for educational purposes and do not guarantee actual market movements.

### Advanced Sentiment Analysis

- **FinBERT**: Financial domain-specific BERT model fine-tuned on financial texts
- **Multi-source weighting**: Different weights for trusted sources (Reuters, Bloomberg, etc.)
- **Temporal weighting**: Recent news weighted more heavily
- **BiLSTM support**: Can be extended with BiLSTM for sequential sentiment processing

## Disclaimer

This application is for educational and analysis purposes only. 

- This is NOT investment advice
- Cryptocurrency investments carry high risk
- Make investment decisions at your own risk
- Past performance does not guarantee future results

## License

This project is open source and available for educational purposes.

