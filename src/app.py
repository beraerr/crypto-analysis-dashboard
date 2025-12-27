"""
Main Streamlit application for Cryptocurrency Analysis Dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')

# Import modules
from src.utils.config import config
from src.data.fetchers import CryptoDataFetcher, NewsFetcher, FearGreedIndexFetcher
from src.indicators.technical import TechnicalIndicators
from src.ui.components import (
    render_header, render_price_metrics, render_news_card,
    render_fear_greed_index, render_sentiment_summary, render_prediction_table
)
from src.ui.sidebar import render_sidebar

# Import models with fallback
try:
    from src.models.predictors import (
        LSTMPredictor, GRUPredictor, TransformerPredictor,
        HybridTransformerGRU, EnsemblePredictor
    )
    from sklearn.linear_model import LinearRegression
    PREDICTORS_AVAILABLE = True
except ImportError:
    PREDICTORS_AVAILABLE = False
    from sklearn.linear_model import LinearRegression

try:
    from src.models.sentiment import MultiSourceSentimentAnalyzer, get_sentiment_analyzer
    SENTIMENT_AVAILABLE = True
except ImportError:
    SENTIMENT_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title=config.ui.page_title,
    page_icon=config.ui.page_icon,
    layout=config.ui.layout,
    initial_sidebar_state=config.ui.initial_sidebar_state
)

# Initialize fetchers
crypto_fetcher = CryptoDataFetcher()
news_fetcher = NewsFetcher()
fng_fetcher = FearGreedIndexFetcher()
indicators = TechnicalIndicators()

# Render header
render_header()

# Render sidebar and get user selections
selections = render_sidebar()

# Fetch crypto data
@st.cache_data(ttl=config.data.cache_ttl)
def fetch_data(symbol: str, period: str):
    return crypto_fetcher.fetch(symbol, period)

data, error = fetch_data(selections['symbol'], selections['period'])

if error:
    st.error(error)
    st.stop()

if data is None or data.empty:
    st.error("Data could not be loaded. Please try again.")
    st.stop()

# Sort data
data = data.sort_index()

# Render price metrics
render_price_metrics(data)

# Create chart
fig = make_subplots(
    rows=3, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.03,
    row_heights=[0.5, 0.25, 0.25],
    subplot_titles=(
        'Price Chart and Technical Indicators',
        'RSI' if selections['show_rsi'] else 'Volume',
        'MACD' if selections['show_macd'] else 'Volume'
    )
)

# Candlestick chart
candlestick = go.Candlestick(
    x=data.index,
    open=data['Open'],
    high=data['High'],
    low=data['Low'],
    close=data['Close'],
    name="Price"
)
fig.add_trace(candlestick, row=1, col=1)

# SMA
if selections['show_sma']:
    sma_short_data = indicators.calculate_sma(data, selections['sma_short'])
    sma_long_data = indicators.calculate_sma(data, selections['sma_long'])
    
    fig.add_trace(
        go.Scatter(x=data.index, y=sma_short_data, name=f'SMA {selections["sma_short"]}',
                  line=dict(color='blue', width=1)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=data.index, y=sma_long_data, name=f'SMA {selections["sma_long"]}',
                  line=dict(color='red', width=1)),
        row=1, col=1
    )

# Bollinger Bands
if selections['show_bollinger']:
    upper, middle, lower = indicators.calculate_bollinger_bands(data)
    fig.add_trace(
        go.Scatter(x=data.index, y=upper, name='BB Upper',
                  line=dict(color='gray', width=1, dash='dash')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=data.index, y=middle, name='BB Middle',
                  line=dict(color='gray', width=1)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=data.index, y=lower, name='BB Lower',
                  line=dict(color='gray', width=1, dash='dash'), fill='tonexty'),
        row=1, col=1
    )

# Price prediction
if selections['enable_prediction']:
    future_dates, predictions = None, None
    
    if PREDICTORS_AVAILABLE and selections['model_type'] != "Simple Linear Regression":
        try:
            with st.spinner(f"Training {selections['model_type']} model... This may take a moment."):
                if "Ensemble" in selections['model_type']:
                    ensemble = EnsemblePredictor()
                    ensemble.add_model(LSTMPredictor(), weight=1.0)
                    ensemble.add_model(GRUPredictor(), weight=1.0)
                    ensemble.add_model(TransformerPredictor(), weight=1.2)
                    ensemble.add_model(HybridTransformerGRU(), weight=1.5)
                    ensemble.train(data, epochs=15, batch_size=32)
                    future_dates, predictions = ensemble.predict(data, selections['prediction_days'])
                elif "LSTM" in selections['model_type']:
                    predictor = LSTMPredictor()
                    predictor.train(data, epochs=15, batch_size=32)
                    future_dates, predictions = predictor.predict(data, selections['prediction_days'])
                elif "GRU" in selections['model_type']:
                    predictor = GRUPredictor()
                    predictor.train(data, epochs=15, batch_size=32)
                    future_dates, predictions = predictor.predict(data, selections['prediction_days'])
                elif "Transformer" in selections['model_type'] and "Hybrid" not in selections['model_type']:
                    predictor = TransformerPredictor()
                    predictor.train(data, epochs=15, batch_size=32)
                    future_dates, predictions = predictor.predict(data, selections['prediction_days'])
                elif "Hybrid" in selections['model_type']:
                    predictor = HybridTransformerGRU()
                    predictor.train(data, epochs=15, batch_size=32)
                    future_dates, predictions = predictor.predict(data, selections['prediction_days'])
        except Exception as e:
            st.warning(f"AI model failed: {str(e)}. Using simple linear regression.")
            future_dates, predictions = None, None
    
    # Fallback to simple linear regression
    if future_dates is None or predictions is None:
        recent_data = data.tail(30).copy()
        X = np.array(range(len(recent_data))).reshape(-1, 1)
        y = recent_data['Close'].values
        model = LinearRegression()
        model.fit(X, y)
        future_days = np.array(range(len(recent_data), len(recent_data) + selections['prediction_days'])).reshape(-1, 1)
        predictions = model.predict(future_days)
        last_date = recent_data.index[-1]
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=selections['prediction_days'], freq='D')
    
    if future_dates is not None and predictions is not None:
        fig.add_trace(
            go.Scatter(x=future_dates, y=predictions, name=f'Prediction ({selections["model_type"]})',
                      line=dict(color='green', width=2, dash='dot'),
                      mode='lines+markers'),
            row=1, col=1
        )

# RSI
if selections['show_rsi']:
    rsi = indicators.calculate_rsi(data, selections['rsi_period'])
    fig.add_trace(
        go.Scatter(x=data.index, y=rsi, name='RSI', line=dict(color='purple')),
        row=2, col=1
    )
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1,
                 annotation_text="Overbought (70)")
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1,
                 annotation_text="Oversold (30)")
    fig.add_hline(y=50, line_dash="dot", line_color="gray", row=2, col=1)

# MACD or Volume
if selections['show_macd']:
    macd, signal, histogram = indicators.calculate_macd(data)
    fig.add_trace(
        go.Scatter(x=data.index, y=macd, name='MACD', line=dict(color='blue')),
        row=3, col=1
    )
    fig.add_trace(
        go.Scatter(x=data.index, y=signal, name='Signal', line=dict(color='red')),
        row=3, col=1
    )
    fig.add_trace(
        go.Bar(x=data.index, y=histogram, name='Histogram', marker_color='gray'),
        row=3, col=1
    )
else:
    fig.add_trace(
        go.Bar(x=data.index, y=data['Volume'], name='Volume', marker_color='lightblue'),
        row=3, col=1
    )

# Update layout
fig.update_layout(
    height=900,
    title_text=f"{selections['selected_crypto']} - Technical Analysis",
    xaxis_rangeslider_visible=False,
    hovermode='x unified'
)

fig.update_xaxes(title_text="Date", row=3, col=1)
fig.update_yaxes(title_text="Price ($)", row=1, col=1)
if selections['show_rsi']:
    fig.update_yaxes(title_text="RSI", row=2, col=1, range=[0, 100])
if selections['show_macd']:
    fig.update_yaxes(title_text="MACD", row=3, col=1)
else:
    fig.update_yaxes(title_text="Volume", row=3, col=1)

st.plotly_chart(fig, use_container_width=True)

# Technical analysis notes
st.subheader("Technical Analysis Notes")
analysis_notes = []

if selections['show_rsi']:
    current_rsi = indicators.calculate_rsi(data, selections['rsi_period']).iloc[-1]
    rsi_analysis = indicators.analyze_rsi(current_rsi)
    if rsi_analysis['signal'] != 'unknown':
        st.info(f"**RSI:** {current_rsi:.2f} - {rsi_analysis['message']}")

if selections['show_sma']:
    sma_short_current = indicators.calculate_sma(data, selections['sma_short']).iloc[-1]
    sma_long_current = indicators.calculate_sma(data, selections['sma_long']).iloc[-1]
    sma_analysis = indicators.analyze_sma_crossover(sma_short_current, sma_long_current)
    if sma_analysis['signal'] != 'unknown':
        st.info(f"**SMA {selections['sma_short']}/{selections['sma_long']}:** "
               f"${sma_short_current:.2f} / ${sma_long_current:.2f} - {sma_analysis['message']}")

if selections['show_macd']:
    macd, signal, _ = indicators.calculate_macd(data)
    macd_current = macd.iloc[-1]
    signal_current = signal.iloc[-1]
    macd_analysis = indicators.analyze_macd(macd_current, signal_current)
    if macd_analysis['signal'] != 'unknown':
        st.info(f"**MACD:** {macd_current:.2f} vs {signal_current:.2f} - {macd_analysis['message']}")

# Sentiment Analysis
if selections['enable_sentiment']:
    st.markdown("---")
    st.subheader("News Analysis & Sentiment Analysis")
    
    col_fng, col_sent = st.columns([1, 2])
    
    with col_fng:
        st.markdown("### Fear & Greed Index")
        fng_data = fng_fetcher.fetch()
        render_fear_greed_index(fng_data)
    
    with col_sent:
        st.markdown("### News Sentiment Analysis")
        
        with st.spinner("Analyzing news..."):
            news_list = news_fetcher.fetch_crypto_news(
                selections['selected_crypto'],
                selections['news_count']
            )
            
            if SENTIMENT_AVAILABLE and selections['use_finbert']:
                try:
                    sentiment_analyzer = get_sentiment_analyzer(use_finbert=True)
                    sentiment_result = sentiment_analyzer.analyze_news_list(news_list)
                except Exception as e:
                    st.warning(f"Advanced sentiment failed: {str(e)}")
                    sentiment_result = {
                        'overall_sentiment': 'nötr',
                        'score': 0.0,
                        'message': 'Sentiment analysis unavailable',
                        'positive_count': 0,
                        'negative_count': 0,
                        'neutral_count': 0
                    }
            else:
                # Simple sentiment using TextBlob
                from textblob import TextBlob
                total_score = 0.0
                sentiment_counts = {'pozitif': 0, 'negatif': 0, 'nötr': 0}
                for news in news_list:
                    text = f"{news['title']} {news.get('summary', '')}"
                    blob = TextBlob(text)
                    polarity = blob.sentiment.polarity
                    total_score += polarity
                    if polarity > 0.1:
                        sentiment_counts['pozitif'] += 1
                    elif polarity < -0.1:
                        sentiment_counts['negatif'] += 1
                    else:
                        sentiment_counts['nötr'] += 1
                
                avg_score = total_score / len(news_list) if news_list else 0.0
                sentiment_result = {
                    'overall_sentiment': 'pozitif' if avg_score > 0.1 else 'negatif' if avg_score < -0.1 else 'nötr',
                    'score': avg_score,
                    'message': f"Sentiment score: {avg_score:.3f}",
                    'positive_count': sentiment_counts['pozitif'],
                    'negative_count': sentiment_counts['negatif'],
                    'neutral_count': sentiment_counts['nötr']
                }
        
        if news_list:
            render_sentiment_summary(sentiment_result)
        else:
            st.warning("No news found or could not be fetched.")
    
    # Display news articles
    if news_list:
        st.markdown("---")
        st.markdown("### Latest News")
        
        for news in news_list[:selections['news_count']]:
            text = f"{news['title']} {news.get('summary', '')}"
            if SENTIMENT_AVAILABLE and selections['use_finbert']:
                try:
                    sentiment_analyzer = get_sentiment_analyzer(use_finbert=True)
                    news_sentiment = sentiment_analyzer.analyze_text(text)
                except:
                    from textblob import TextBlob
                    blob = TextBlob(text)
                    news_sentiment = {
                        'polarity': blob.sentiment.polarity,
                        'sentiment': 'pozitif' if blob.sentiment.polarity > 0.1 else 'negatif' if blob.sentiment.polarity < -0.1 else 'nötr'
                    }
            else:
                from textblob import TextBlob
                blob = TextBlob(text)
                news_sentiment = {
                    'polarity': blob.sentiment.polarity,
                    'sentiment': 'pozitif' if blob.sentiment.polarity > 0.1 else 'negatif' if blob.sentiment.polarity < -0.1 else 'nötr'
                }
            
            render_news_card(news, news_sentiment)

# Price Prediction Table
if selections['enable_prediction']:
    st.subheader("Price Prediction")
    if future_dates is not None and predictions is not None:
        render_prediction_table(future_dates, predictions, selections['model_type'])
        st.warning("⚠️ These predictions are AI-generated estimates and do not guarantee actual price movements. "
                  "Past performance does not guarantee future results. Always do your own research.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; padding: 1rem;'>
    <p>Cryptocurrency Analysis Dashboard v2.0</p>
    <p>Educational purposes only - Not investment advice</p>
</div>
""", unsafe_allow_html=True)

