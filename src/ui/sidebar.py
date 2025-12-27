"""
Sidebar UI components and configuration
"""

import streamlit as st
from typing import Dict, Tuple

from src.utils.config import config
from src.utils.helpers import check_dependencies


def render_sidebar() -> Dict:
    """
    Render sidebar and return user selections
    
    Returns:
        Dictionary with user selections
    """
    st.sidebar.header("Settings")
    
    # Check dependencies
    deps = check_dependencies()
    ai_models_available = deps.get('torch', False) or deps.get('tensorflow', False)
    sentiment_available = deps.get('transformers', False) or deps.get('textblob', False)
    
    # Crypto selection
    selected_crypto = st.sidebar.selectbox(
        "Kripto Para Seçin:",
        options=list(config.data.supported_cryptos.keys()),
        index=0
    )
    symbol = config.data.supported_cryptos[selected_crypto]
    
    # Period selection
    selected_period = st.sidebar.selectbox(
        "Zaman Aralığı:",
        options=list(config.data.period_options.keys()),
        index=3  # Default: 1 year
    )
    period = config.data.period_options[selected_period]
    
    # Technical indicators
    st.sidebar.subheader("Technical Indicators")
    show_rsi = st.sidebar.checkbox("RSI (Relative Strength Index)", value=True)
    rsi_period = st.sidebar.slider("RSI Periyodu", 5, 30, 14, disabled=not show_rsi)
    
    show_sma = st.sidebar.checkbox("Hareketli Ortalamalar (SMA)", value=True)
    sma_short = st.sidebar.slider("Kısa SMA", 5, 50, 50, disabled=not show_sma)
    sma_long = st.sidebar.slider("Uzun SMA", 50, 200, 200, disabled=not show_sma)
    
    show_macd = st.sidebar.checkbox("MACD", value=False)
    show_bollinger = st.sidebar.checkbox("Bollinger Bands", value=False)
    
    # Prediction settings
    st.sidebar.subheader("Prediction Settings")
    enable_prediction = st.sidebar.checkbox("Enable Price Prediction", value=True)
    prediction_days = st.sidebar.slider(
        "Prediction Days", 3, 14, config.ui.default_prediction_days, 
        disabled=not enable_prediction
    )
    
    # AI Model Selection
    model_type = "Simple Linear Regression"
    if ai_models_available and enable_prediction:
        st.sidebar.subheader("🤖 AI Model Selection")
        model_type = st.sidebar.selectbox(
            "Prediction Model:",
            options=[
                "Simple Linear Regression",
                "LSTM (Long Short-Term Memory)",
                "GRU (Gated Recurrent Unit)",
                "Transformer",
                "Hybrid Transformer-GRU",
                "Ensemble (All Models)"
            ],
            index=5 if ai_models_available else 0,
            disabled=not ai_models_available
        )
    
    # Sentiment Analysis
    st.sidebar.subheader("Sentiment Analysis")
    enable_sentiment = st.sidebar.checkbox("Enable News Analysis", value=True)
    news_count = st.sidebar.slider(
        "Number of Articles", 5, 20, config.ui.default_news_count, 
        disabled=not enable_sentiment
    )
    
    # Advanced Sentiment Option
    use_finbert = False
    if sentiment_available and enable_sentiment:
        use_finbert = st.sidebar.checkbox(
            "Use FinBERT (Advanced AI)", 
            value=config.sentiment.use_finbert,
            help="Uses financial domain-specific BERT model for better sentiment analysis"
        )
    
    # Refresh button
    if st.sidebar.button("Refresh Data", type="primary"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()
    
    return {
        'symbol': symbol,
        'selected_crypto': selected_crypto,
        'period': period,
        'show_rsi': show_rsi,
        'rsi_period': rsi_period,
        'show_sma': show_sma,
        'sma_short': sma_short,
        'sma_long': sma_long,
        'show_macd': show_macd,
        'show_bollinger': show_bollinger,
        'enable_prediction': enable_prediction,
        'prediction_days': prediction_days,
        'model_type': model_type,
        'enable_sentiment': enable_sentiment,
        'news_count': news_count,
        'use_finbert': use_finbert,
        'ai_models_available': ai_models_available,
        'sentiment_available': sentiment_available
    }

