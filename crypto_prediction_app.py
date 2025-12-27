"""
Cryptocurrency Price Analysis and Prediction Tool
Technical analysis and sentiment analysis application for cryptocurrency trading
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import warnings
import requests
import feedparser
from textblob import TextBlob
from pygooglenews import GoogleNews
import time
warnings.filterwarnings('ignore')

# Import advanced AI models
try:
    from ai_models import LSTMPredictor, GRUPredictor, TransformerPredictor, HybridTransformerGRU, EnsemblePredictor
    AI_MODELS_AVAILABLE = True
except ImportError:
    AI_MODELS_AVAILABLE = False
    st.warning("⚠️ Advanced AI models not available. Install dependencies: pip install torch tensorflow transformers")

# Import advanced sentiment analysis
try:
    from sentiment_ai import MultiSourceSentimentAnalyzer, get_sentiment_analyzer
    ADVANCED_SENTIMENT_AVAILABLE = True
except ImportError:
    ADVANCED_SENTIMENT_AVAILABLE = False

# TextBlob için NLTK verilerini kontrol et ve indir
try:
    from textblob.download_corpora import download_all
    import nltk
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)
except:
    pass  # NLTK verileri zaten yüklü veya otomatik indirilecek

st.set_page_config(
    page_title="Crypto Analysis Dashboard",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .warning-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="warning-box">
    <strong>DISCLAIMER:</strong> This application is for educational and analysis purposes only. 
    The information provided does not constitute investment advice. Cryptocurrency investments carry high risk. 
    Make investment decisions at your own risk.
</div>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">Cryptocurrency Price Analysis & Technical Indicators</h1>', unsafe_allow_html=True)

st.sidebar.header("Settings")

# Kripto sembolü seçimi
crypto_symbols = {
    "Bitcoin (BTC-USD)": "BTC-USD",
    "Ethereum (ETH-USD)": "ETH-USD",
    "Binance Coin (BNB-USD)": "BNB-USD",
    "Cardano (ADA-USD)": "ADA-USD",
    "Solana (SOL-USD)": "SOL-USD",
    "Ripple (XRP-USD)": "XRP-USD",
    "Polkadot (DOT-USD)": "DOT-USD",
    "Dogecoin (DOGE-USD)": "DOGE-USD"
}

selected_crypto = st.sidebar.selectbox(
    "Kripto Para Seçin:",
    options=list(crypto_symbols.keys()),
    index=0
)

symbol = crypto_symbols[selected_crypto]

# Zaman aralığı seçimi
period_options = {
    "1 Ay": "1mo",
    "3 Ay": "3mo",
    "6 Ay": "6mo",
    "1 Yıl": "1y",
    "2 Yıl": "2y",
    "5 Yıl": "5y"
}

selected_period = st.sidebar.selectbox(
    "Zaman Aralığı:",
    options=list(period_options.keys()),
    index=3  # Varsayılan: 1 yıl
)

period = period_options[selected_period]

st.sidebar.subheader("Technical Indicators")

show_rsi = st.sidebar.checkbox("RSI (Relative Strength Index)", value=True)
rsi_period = st.sidebar.slider("RSI Periyodu", 5, 30, 14, disabled=not show_rsi)

show_sma = st.sidebar.checkbox("Hareketli Ortalamalar (SMA)", value=True)
sma_short = st.sidebar.slider("Kısa SMA", 5, 50, 50, disabled=not show_sma)
sma_long = st.sidebar.slider("Uzun SMA", 50, 200, 200, disabled=not show_sma)

show_macd = st.sidebar.checkbox("MACD", value=False)
show_bollinger = st.sidebar.checkbox("Bollinger Bands", value=False)

st.sidebar.subheader("Prediction Settings")
enable_prediction = st.sidebar.checkbox("Enable Price Prediction", value=True)
prediction_days = st.sidebar.slider("Prediction Days", 3, 14, 7, disabled=not enable_prediction)

# AI Model Selection
if AI_MODELS_AVAILABLE and enable_prediction:
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
        index=5 if AI_MODELS_AVAILABLE else 0,
        disabled=not AI_MODELS_AVAILABLE
    )
    
    if "Ensemble" in model_type:
        use_ensemble = True
    else:
        use_ensemble = False
        selected_model_type = model_type.lower()
else:
    model_type = "Simple Linear Regression"
    use_ensemble = False

st.sidebar.subheader("Sentiment Analysis")
enable_sentiment = st.sidebar.checkbox("Enable News Analysis", value=True)
news_count = st.sidebar.slider("Number of Articles", 5, 20, 10, disabled=not enable_sentiment)

# Advanced Sentiment Analysis Option
if ADVANCED_SENTIMENT_AVAILABLE and enable_sentiment:
    use_finbert = st.sidebar.checkbox("Use FinBERT (Advanced AI)", value=True, 
                                      help="Uses financial domain-specific BERT model for better sentiment analysis")
else:
    use_finbert = False

@st.cache_data(ttl=300)
def fetch_crypto_data(symbol, period):
    """Fetch cryptocurrency price data from Yahoo Finance"""
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=period)
        if data.empty:
            return None, "Veri bulunamadı. Lütfen sembolü kontrol edin."
        return data, None
    except Exception as e:
        return None, f"Hata: {str(e)}"

def calculate_rsi(data, period=14):
    """Calculate Relative Strength Index"""
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_sma(data, period):
    """Calculate Simple Moving Average"""
    return data['Close'].rolling(window=period).mean()

def calculate_ema(data, period):
    """Calculate Exponential Moving Average"""
    return data['Close'].ewm(span=period, adjust=False).mean()

def calculate_macd(data, fast=12, slow=26, signal=9):
    """Calculate MACD indicator"""
    ema_fast = calculate_ema(data, fast)
    ema_slow = calculate_ema(data, slow)
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    return macd, signal_line, histogram

def calculate_bollinger_bands(data, period=20, std_dev=2):
    """Calculate Bollinger Bands"""
    sma = calculate_sma(data, period)
    std = data['Close'].rolling(window=period).std()
    upper_band = sma + (std * std_dev)
    lower_band = sma - (std * std_dev)
    return upper_band, sma, lower_band

# ==================== SENTIMENT ANALYSIS FUNCTIONS ====================

SOURCE_WEIGHTS = {
    'reuters': 1.5,
    'bloomberg': 1.5,
    'coindesk': 1.3,
    'cointelegraph': 1.3,
    'forbes': 1.2,
    'cnbc': 1.2,
    'bbc': 1.2,
    'default': 1.0
}

@st.cache_data(ttl=600)
def fetch_crypto_news(crypto_name, max_articles=10):
    """Fetch cryptocurrency news in Turkish and English from Google News (last 24 hours)"""
    try:
        clean_name = crypto_name.split('(')[0].strip()
        
        gn_en = GoogleNews(lang='en', country='US')
        gn_tr = GoogleNews(lang='tr', country='TR')
        
        query = f"{clean_name} cryptocurrency"
        
        news_list = []
        seen_titles = set()
        
        # Fetch English news
        try:
            search_en = gn_en.search(query, when='1d')  # Son 24 saat
            if search_en and 'entries' in search_en:
                for entry in search_en['entries'][:max_articles//2]:
                    title = entry.get('title', '')
                    if title and title not in seen_titles:
                        seen_titles.add(title)
                        link = entry.get('link', '')
                        source = entry.get('source', {}).get('title', 'Unknown')
                        source_lower = source.lower()
                        
                        # Kaynak ağırlığını belirle
                        weight = SOURCE_WEIGHTS.get('default', 1.0)
                        for key, value in SOURCE_WEIGHTS.items():
                            if key in source_lower:
                                weight = value
                                break
                        
                        news_list.append({
                            'title': title,
                            'link': link,
                            'published': entry.get('published', ''),
                            'source': source,
                            'summary': entry.get('title', ''),
                        'weight': weight,
                        'language': 'en'
                        })
        except Exception as e:
            st.warning(f"Error fetching English news: {str(e)}")
        
        # Fetch Turkish news
        try:
            search_tr = gn_tr.search(query, when='1d')  # Son 24 saat
            if search_tr and 'entries' in search_tr:
                for entry in search_tr['entries'][:max_articles//2]:
                    title = entry.get('title', '')
                    if title and title not in seen_titles:
                        seen_titles.add(title)
                        link = entry.get('link', '')
                        source = entry.get('source', {}).get('title', 'Unknown')
                        source_lower = source.lower()
                        
                        # Kaynak ağırlığını belirle
                        weight = SOURCE_WEIGHTS.get('default', 1.0)
                        for key, value in SOURCE_WEIGHTS.items():
                            if key in source_lower:
                                weight = value
                                break
                        
                        news_list.append({
                            'title': title,
                            'link': link,
                            'published': entry.get('published', ''),
                            'source': source,
                            'summary': entry.get('title', ''),
                        'weight': weight,
                        'language': 'tr'
                        })
        except Exception as e:
            st.warning(f"Error fetching Turkish news: {str(e)}")
        
        return news_list[:max_articles]
        
    except Exception as e:
        # Fallback: Eski feedparser yöntemi
        try:
            clean_name = crypto_name.split('(')[0].strip()
            query = f"{clean_name} cryptocurrency"
            url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
            feed = feedparser.parse(url)
            
            news_list = []
            for entry in feed.entries[:max_articles]:
                source = entry.get('source', {}).get('title', 'Unknown')
                source_lower = source.lower()
                weight = SOURCE_WEIGHTS.get('default', 1.0)
                for key, value in SOURCE_WEIGHTS.items():
                    if key in source_lower:
                        weight = value
                        break
                
                news_list.append({
                    'title': entry.title,
                    'link': entry.link,
                    'published': entry.get('published', ''),
                    'source': source,
                    'summary': entry.get('summary', entry.title),
                    'weight': weight,
                    'language': 'en'
                })
            return news_list
        except:
            st.warning(f"Error fetching news: {str(e)}")
            return []

# Initialize advanced sentiment analyzer if available
if ADVANCED_SENTIMENT_AVAILABLE:
    try:
        advanced_sentiment_analyzer = get_sentiment_analyzer(use_finbert=True)
    except:
        advanced_sentiment_analyzer = None
else:
    advanced_sentiment_analyzer = None

def analyze_sentiment(text, use_advanced=None):
    """Perform sentiment analysis using FinBERT or TextBlob"""
    if use_advanced is None:
        use_advanced = use_finbert if 'use_finbert' in globals() else False
    
    # Try advanced sentiment if available and requested
    if use_advanced and advanced_sentiment_analyzer:
        try:
            result = advanced_sentiment_analyzer.analyze_text(text)
            return {
                'polarity': result.get('polarity', 0.0),
                'subjectivity': 0.0,  # FinBERT doesn't provide subjectivity
                'sentiment': result.get('sentiment', 'nötr'),
                'score': result.get('score', 0.0)
            }
        except Exception as e:
            st.warning(f"Advanced sentiment analysis failed: {str(e)}. Falling back to TextBlob.")
    
    # Fallback to TextBlob
    try:
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity  # -1 (negatif) ile +1 (pozitif) arası
        subjectivity = blob.sentiment.subjectivity  # 0 (objektif) ile 1 (subjektif) arası
        
        return {
            'polarity': polarity,
            'subjectivity': subjectivity,
            'sentiment': 'pozitif' if polarity > 0.1 else 'negatif' if polarity < -0.1 else 'nötr',
            'score': abs(polarity)
        }
    except:
        return {
            'polarity': 0.0,
            'subjectivity': 0.0,
            'sentiment': 'nötr',
            'score': 0.0
        }

def analyze_news_sentiment(news_list):
    """Analyze sentiment of news list using advanced AI or TextBlob"""
    if not news_list:
        return {
            'overall_sentiment': 'nötr',
            'total_score': 0.0,
            'weighted_score': 0.0,
            'positive_count': 0,
            'negative_count': 0,
            'neutral_count': 0,
            'message': 'Haber bulunamadı'
        }
    
    # Use advanced sentiment analyzer if available
    if use_finbert and advanced_sentiment_analyzer:
        try:
            return advanced_sentiment_analyzer.analyze_news_list(news_list)
        except Exception as e:
            st.warning(f"Advanced sentiment analysis failed: {str(e)}. Using TextBlob fallback.")
    
    # Fallback to original TextBlob method
    total_score = 0.0
    weighted_score = 0.0
    total_weight = 0.0
    sentiment_counts = {'pozitif': 0, 'negatif': 0, 'nötr': 0}
    
    for news in news_list:
        # Başlık ve özeti birleştir
        text = f"{news['title']} {news.get('summary', '')}"
        sentiment_data = analyze_sentiment(text, use_advanced=False)
        
        polarity = sentiment_data['polarity']
        weight = news.get('weight', 1.0)
        
        total_score += polarity
        weighted_score += polarity * weight
        total_weight += weight
        
        sentiment_counts[sentiment_data['sentiment']] += 1
    
    # Ağırlıklı ortalama
    if total_weight > 0:
        weighted_avg = weighted_score / total_weight
    else:
        weighted_avg = total_score / len(news_list) if news_list else 0.0
    
    if weighted_avg > 0.1:
        overall = 'pozitif'
        message = "Positive: News sentiment supports upward trend. Market appears optimistic."
    elif weighted_avg < -0.1:
        overall = 'negatif'
        message = "Negative: Market sentiment shows concern. Exercise caution."
    else:
        overall = 'nötr'
        message = "Neutral: News flow is balanced. Market appears calm."
    
    return {
        'overall_sentiment': overall,
        'total_score': total_score / len(news_list) if news_list else 0.0,
        'weighted_score': weighted_avg,
        'positive_count': sentiment_counts['pozitif'],
        'negative_count': sentiment_counts['negatif'],
        'neutral_count': sentiment_counts['nötr'],
        'message': message,
        'score': weighted_avg
    }

@st.cache_data(ttl=3600)
def fetch_fear_greed_index():
    """Fetch Crypto Fear & Greed Index data"""
    try:
        # Alternative.me API'si
        url = "https://api.alternative.me/fng/"
        response = requests.get(url, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            if 'data' in data and len(data['data']) > 0:
                latest = data['data'][0]
                value = int(latest['value'])
                classification = latest['value_classification']
                
                return {
                    'value': value,
                    'classification': classification,
                    'timestamp': latest.get('timestamp', ''),
                    'success': True
                }
    except Exception as e:
        pass
    
    return {
        'value': None,
        'classification': 'Unknown',
        'success': False
    }

def get_fear_greed_color(value):
    """Return color based on Fear & Greed Index value"""
    if value is None:
        return 'gray'
    if value >= 75:
        return 'red'  # Aşırı açgözlülük
    elif value >= 55:
        return 'orange'
    elif value >= 45:
        return 'yellow'
    elif value >= 25:
        return 'lightgreen'
    else:
        return 'green'  # Aşırı korku

def get_fear_greed_emoji(value):
    """Return emoji based on Fear & Greed Index value"""
    if value is None:
        return ''
    if value >= 75:
        return 'Extreme Greed'
    elif value >= 55:
        return 'Greed'
    elif value >= 45:
        return 'Neutral'
    elif value >= 25:
        return 'Fear'
    else:
        return 'Extreme Fear'

# ==================== PREDICTION FUNCTIONS ====================

def simple_price_prediction(data, days=7):
    """Simple linear regression price prediction"""
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler
    
    recent_data = data.tail(30).copy()
    
    X = np.array(range(len(recent_data))).reshape(-1, 1)
    y = recent_data['Close'].values
    
    model = LinearRegression()
    model.fit(X, y)
    
    future_days = np.array(range(len(recent_data), len(recent_data) + days)).reshape(-1, 1)
    predictions = model.predict(future_days)
    
    last_date = recent_data.index[-1]
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=days, freq='D')
    
    return future_dates, predictions

@st.cache_resource
def get_ai_predictor(model_type, data):
    """Get and train AI predictor model"""
    if not AI_MODELS_AVAILABLE:
        return None
    
    try:
        if "lstm" in model_type.lower():
            predictor = LSTMPredictor(sequence_length=60, hidden_units=50, num_layers=2)
        elif "gru" in model_type.lower():
            predictor = GRUPredictor(sequence_length=60, hidden_units=50, num_layers=2)
        elif "transformer" in model_type.lower() and "hybrid" not in model_type.lower():
            predictor = TransformerPredictor(sequence_length=60, d_model=64, num_heads=4)
        elif "hybrid" in model_type.lower():
            predictor = HybridTransformerGRU(sequence_length=60, d_model=64, gru_units=50)
        else:
            return None
        
        # Train the model (with reduced epochs for faster response)
        predictor.train(data, epochs=20, batch_size=32, validation_split=0.2)
        return predictor
    except Exception as e:
        st.error(f"Error training {model_type}: {str(e)}")
        return None

@st.cache_resource
def get_ensemble_predictor(data):
    """Get and train ensemble predictor"""
    if not AI_MODELS_AVAILABLE:
        return None
    
    try:
        ensemble = EnsemblePredictor()
        
        # Add multiple models
        ensemble.add_model(LSTMPredictor(sequence_length=60, hidden_units=50, num_layers=2), weight=1.0)
        ensemble.add_model(GRUPredictor(sequence_length=60, hidden_units=50, num_layers=2), weight=1.0)
        ensemble.add_model(TransformerPredictor(sequence_length=60, d_model=64, num_heads=4), weight=1.2)
        ensemble.add_model(HybridTransformerGRU(sequence_length=60, d_model=64, gru_units=50), weight=1.5)
        
        # Train ensemble
        ensemble.train(data, epochs=15, batch_size=32)
        return ensemble
    except Exception as e:
        st.error(f"Error training ensemble: {str(e)}")
        return None

def advanced_price_prediction(data, days=7, model_type="Ensemble"):
    """Advanced AI-based price prediction"""
    if not AI_MODELS_AVAILABLE:
        return None, None
    
    try:
        if "Ensemble" in model_type or "ensemble" in model_type.lower():
            predictor = get_ensemble_predictor(data)
            if predictor:
                return predictor.predict(data, days)
        else:
            predictor = get_ai_predictor(model_type, data)
            if predictor:
                return predictor.predict(data, days)
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
    
    return None, None

if st.sidebar.button("Refresh Data", type="primary"):
    st.cache_data.clear()

# Veri çekme
data, error = fetch_crypto_data(symbol, period)

if error:
    st.error(error)
    st.stop()

if data is None or data.empty:
    st.error("Veri yüklenemedi. Lütfen tekrar deneyin.")
    st.stop()

# Veri işleme
data = data.sort_index()
latest_price = data['Close'].iloc[-1]
previous_price = data['Close'].iloc[-2] if len(data) > 1 else latest_price
price_change = latest_price - previous_price
price_change_pct = (price_change / previous_price) * 100

# Ana metrikler
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Current Price",
        value=f"${latest_price:,.2f}",
        delta=f"{price_change_pct:.2f}%"
    )

with col2:
    st.metric(
        label="24h Change",
        value=f"${price_change:,.2f}",
        delta=f"{price_change_pct:.2f}%"
    )

with col3:
    st.metric(
        label="Daily High",
        value=f"${data['High'].iloc[-1]:,.2f}"
    )

with col4:
    st.metric(
        label="Daily Low",
        value=f"${data['Low'].iloc[-1]:,.2f}"
    )

# Grafik oluşturma
fig = make_subplots(
    rows=3, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.03,
    row_heights=[0.5, 0.25, 0.25],
    subplot_titles=('Price Chart and Technical Indicators', 'RSI', 'MACD' if show_macd else 'Volume')
)

# 1. Mum grafiği
candlestick = go.Candlestick(
    x=data.index,
    open=data['Open'],
    high=data['High'],
    low=data['Low'],
    close=data['Close'],
    name="Fiyat"
)
fig.add_trace(candlestick, row=1, col=1)

# Hareketli ortalamalar
if show_sma:
    sma_short_data = calculate_sma(data, sma_short)
    sma_long_data = calculate_sma(data, sma_long)
    
    fig.add_trace(
        go.Scatter(x=data.index, y=sma_short_data, name=f'SMA {sma_short}', 
                  line=dict(color='blue', width=1)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=data.index, y=sma_long_data, name=f'SMA {sma_long}', 
                  line=dict(color='red', width=1)),
        row=1, col=1
    )

# Bollinger Bands
if show_bollinger:
    upper, middle, lower = calculate_bollinger_bands(data)
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

# Fiyat tahmini
if enable_prediction:
    if AI_MODELS_AVAILABLE and model_type != "Simple Linear Regression":
        with st.spinner(f"Training {model_type} model... This may take a moment."):
            future_dates, predictions = advanced_price_prediction(data, prediction_days, model_type)
            if future_dates is None or predictions is None:
                # Fallback to simple prediction
                future_dates, predictions = simple_price_prediction(data, prediction_days)
                st.info(f"⚠️ {model_type} model unavailable. Using simple linear regression.")
    else:
        future_dates, predictions = simple_price_prediction(data, prediction_days)
    
    if future_dates is not None and predictions is not None:
        fig.add_trace(
            go.Scatter(x=future_dates, y=predictions, name=f'Prediction ({model_type})', 
                      line=dict(color='green', width=2, dash='dot'),
                      mode='lines+markers'),
            row=1, col=1
        )

# 2. RSI
if show_rsi:
    rsi = calculate_rsi(data, rsi_period)
    fig.add_trace(
        go.Scatter(x=data.index, y=rsi, name='RSI', line=dict(color='purple')),
        row=2, col=1
    )
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1, 
                  annotation_text="Overbought (70)")
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1, 
                  annotation_text="Oversold (30)")
    fig.add_hline(y=50, line_dash="dot", line_color="gray", row=2, col=1)

# 3. MACD veya Hacim
if show_macd:
    macd, signal, histogram = calculate_macd(data)
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
        go.Bar(x=data.index, y=data['Volume'], name='Hacim', marker_color='lightblue'),
        row=3, col=1
    )

# Grafik düzenleme
fig.update_layout(
    height=900,
    title_text=f"{selected_crypto} - Technical Analysis",
    xaxis_rangeslider_visible=False,
    hovermode='x unified'
)

fig.update_xaxes(title_text="Date", row=3, col=1)
fig.update_yaxes(title_text="Price ($)", row=1, col=1)
if show_rsi:
    fig.update_yaxes(title_text="RSI", row=2, col=1, range=[0, 100])
if show_macd:
    fig.update_yaxes(title_text="MACD", row=3, col=1)
else:
    fig.update_yaxes(title_text="Hacim", row=3, col=1)

st.plotly_chart(fig, use_container_width=True)

st.subheader("Technical Analysis Notes")

analysis_notes = []

# RSI analizi
if show_rsi:
    current_rsi = calculate_rsi(data, rsi_period).iloc[-1]
    if not pd.isna(current_rsi):
        if current_rsi > 70:
            analysis_notes.append({
                "indicator": "RSI",
                "value": f"{current_rsi:.2f}",
                "signal": "Overbought - Price at high level, exercise caution",
                "color": "red"
            })
        elif current_rsi < 30:
            analysis_notes.append({
                "indicator": "RSI",
                "value": f"{current_rsi:.2f}",
                "signal": "Oversold - Potential opportunity",
                "color": "green"
            })
        else:
            analysis_notes.append({
                "indicator": "RSI",
                "value": f"{current_rsi:.2f}",
                "signal": "Normal level",
                "color": "blue"
            })

# SMA analizi
if show_sma:
    sma_short_current = calculate_sma(data, sma_short).iloc[-1]
    sma_long_current = calculate_sma(data, sma_long).iloc[-1]
    if not pd.isna(sma_short_current) and not pd.isna(sma_long_current):
        if sma_short_current > sma_long_current:
            analysis_notes.append({
                "indicator": f"SMA {sma_short}/{sma_long}",
                "value": f"${sma_short_current:.2f} / ${sma_long_current:.2f}",
                "signal": "Uptrend - Short MA above Long MA",
                "color": "green"
            })
        else:
            analysis_notes.append({
                "indicator": f"SMA {sma_short}/{sma_long}",
                "value": f"${sma_short_current:.2f} / ${sma_long_current:.2f}",
                "signal": "Downtrend - Short MA below Long MA",
                "color": "red"
            })

# MACD analizi
if show_macd:
    macd, signal, _ = calculate_macd(data)
    macd_current = macd.iloc[-1]
    signal_current = signal.iloc[-1]
    if not pd.isna(macd_current) and not pd.isna(signal_current):
        if macd_current > signal_current:
            analysis_notes.append({
                "indicator": "MACD",
                "value": f"{macd_current:.2f} > {signal_current:.2f}",
                "signal": "Bullish signal",
                "color": "green"
            })
        else:
            analysis_notes.append({
                "indicator": "MACD",
                "value": f"{macd_current:.2f} < {signal_current:.2f}",
                "signal": "Bearish signal",
                "color": "red"
            })

# Analiz notlarını göster
if analysis_notes:
    for note in analysis_notes:
        st.info(f"**{note['indicator']}:** {note['value']} - {note['signal']}")
else:
    st.info("Enable technical indicators to see analysis notes.")

# ==================== SENTIMENT ANALYSIS SECTION ====================
if enable_sentiment:
    st.markdown("---")
    st.subheader("News Analysis & Sentiment Analysis")
    
    col_fng, col_sent = st.columns([1, 2])
    
    with col_fng:
        st.markdown("### Fear & Greed Index")
        fng_data = fetch_fear_greed_index()
        
        if fng_data['success']:
            fng_value = fng_data['value']
            fng_class = fng_data['classification']
            fng_color = get_fear_greed_color(fng_value)
            fng_emoji = get_fear_greed_emoji(fng_value)
            
            st.metric(
                label="Current Value",
                value=f"{fng_value}",
                delta=fng_class
            )
            
            st.progress(fng_value / 100)
            st.caption(f"**Classification:** {fng_class}")
            
            if fng_value >= 75:
                st.warning("Extreme Greed: Market is overly optimistic. Exercise caution!")
            elif fng_value <= 25:
                st.info("Extreme Fear: Market is overly pessimistic. Potential opportunity!")
        else:
            st.info("Fear & Greed Index data is currently unavailable.")
    
    with col_sent:
        st.markdown("### News Sentiment Analysis")
        
        with st.spinner("Analyzing news..."):
            news_list = fetch_crypto_news(selected_crypto, news_count)
            sentiment_result = analyze_news_sentiment(news_list)
        
        if news_list:
            # Genel duygu skoru
            score = sentiment_result['score']
            sentiment_color = 'green' if score > 0.1 else 'red' if score < -0.1 else 'gray'
            
            st.markdown(f"**Overall Sentiment:** {sentiment_result['message']}")
            
            col_pos, col_neg, col_neut = st.columns(3)
            with col_pos:
                st.metric("Positive", sentiment_result['positive_count'])
            with col_neg:
                st.metric("Negative", sentiment_result['negative_count'])
            with col_neut:
                st.metric("Neutral", sentiment_result['neutral_count'])
            
            st.metric(
                "Weighted Sentiment Score",
                f"{score:.3f}",
                delta=f"{'Positive' if score > 0 else 'Negative' if score < 0 else 'Neutral'}"
            )
        else:
            st.warning("No news found or could not be fetched.")
    
    if news_list:
        st.markdown("---")
        st.markdown("### Latest News")
        
        for i, news in enumerate(news_list[:news_count], 1):
            news_text = f"{news['title']} {news.get('summary', '')}"
            news_sentiment = analyze_sentiment(news_text, use_advanced=use_finbert if 'use_finbert' in globals() else False)
            
            if news_sentiment['polarity'] > 0.1:
                border_color = "4px solid #28a745"
                sentiment_label = "Positive"
            elif news_sentiment['polarity'] < -0.1:
                border_color = "4px solid #dc3545"
                sentiment_label = "Negative"
            else:
                border_color = "4px solid #6c757d"
                sentiment_label = "Neutral"
            
            card_html = f"""
            <div style="
                border-left: {border_color};
                padding: 1rem;
                margin: 0.5rem 0;
                background-color: #f8f9fa;
                border-radius: 0.25rem;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            ">
                <h4 style="margin-top: 0;">
                    {news['title']}
                </h4>
                <p style="color: #6c757d; font-size: 0.9rem; margin-bottom: 0.5rem;">
                    <strong>Source:</strong> {news['source']} | 
                    <strong>Language:</strong> {'TR' if news.get('language') == 'tr' else 'EN'} | 
                    <strong>Weight:</strong> {news['weight']:.1f}x | 
                    <strong>Sentiment:</strong> {sentiment_label} ({news_sentiment['polarity']:.2f})
                </p>
                <p style="font-size: 0.85rem; color: #495057;">
                    {news.get('summary', '')[:200]}...
                </p>
                <a href="{news['link']}" target="_blank" style="color: #007bff; text-decoration: none;">
                    Read Article →
                </a>
            </div>
            """
            st.markdown(card_html, unsafe_allow_html=True)
    
    if news_list and fng_data['success']:
        st.markdown("---")
        st.markdown("### Combined Analysis Summary")
        
        combined_analysis = []
        
        fng_value = fng_data['value']
        if fng_value >= 75:
            combined_analysis.append("Fear & Greed Index shows extreme greed - exercise caution")
        elif fng_value <= 25:
            combined_analysis.append("Fear & Greed Index shows extreme fear - potential opportunity")
        
        if sentiment_result['score'] > 0.1:
            combined_analysis.append("News sentiment is generally positive - upward trend may be supported")
        elif sentiment_result['score'] < -0.1:
            combined_analysis.append("News sentiment is generally negative - exercise caution")
        
        if combined_analysis:
            for analysis in combined_analysis:
                st.info(analysis)
        else:
            st.info("Market appears balanced. Both technical and sentiment analysis are at neutral levels.")

if enable_prediction:
    st.subheader("Price Prediction")
    
    # Get predictions based on selected model
    if AI_MODELS_AVAILABLE and model_type != "Simple Linear Regression":
        with st.spinner(f"Generating predictions using {model_type}..."):
            future_dates, predictions = advanced_price_prediction(data, prediction_days, model_type)
            if future_dates is None or predictions is None:
                # Fallback
                future_dates, predictions = simple_price_prediction(data, prediction_days)
                model_display = "Simple Linear Regression (Fallback)"
            else:
                model_display = model_type
    else:
        future_dates, predictions = simple_price_prediction(data, prediction_days)
        model_display = "Simple Linear Regression"
    
    if future_dates is not None and predictions is not None:
        pred_df = pd.DataFrame({
            'Date': future_dates,
            'Predicted Price': predictions
        })
        
        st.dataframe(pred_df.style.format({
            'Predicted Price': '${:,.2f}'
        }), use_container_width=True)
        
        # Show model info
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"🤖 **Model Used:** {model_display}")
        with col2:
            if len(predictions) > 0:
                price_change = predictions[-1] - data['Close'].iloc[-1]
                price_change_pct = (price_change / data['Close'].iloc[-1]) * 100
                st.metric("Predicted Change", f"{price_change_pct:.2f}%", 
                         delta=f"${price_change:,.2f}")
        
        st.warning("⚠️ These predictions are AI-generated estimates and do not guarantee actual price movements. Past performance does not guarantee future results. Always do your own research.")

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; padding: 1rem;'>
    <p>Cryptocurrency Analysis Dashboard</p>
    <p>Educational purposes only - Not investment advice</p>
</div>
""", unsafe_allow_html=True)

