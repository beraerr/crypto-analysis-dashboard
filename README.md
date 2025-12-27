# Cryptocurrency Analysis Dashboard

A comprehensive web application for cryptocurrency price analysis, technical indicators, sentiment analysis, and AI-powered price prediction.

## Features

- Real-time price data from Yahoo Finance
- Technical indicators: RSI, SMA, MACD, Bollinger Bands
- Advanced AI Price Prediction using deep learning models
- Advanced Sentiment Analysis using FinBERT
- Crypto Fear & Greed Index integration
- Interactive charts with Plotly
- Multi-source news analysis (English & Turkish)

## Dependencies

### Required Dependencies

These dependencies are essential for the application to run:

- **streamlit** (>=1.28.0): Web framework for building the dashboard interface
- **yfinance** (>=0.2.28): Fetches real-time cryptocurrency price data from Yahoo Finance
- **pandas** (>=2.0.0): Data manipulation and analysis
- **plotly** (>=5.17.0): Interactive charting library
- **numpy** (>=1.24.0): Numerical computing
- **scikit-learn** (>=1.3.0): Machine learning utilities and simple linear regression
- **textblob** (>=0.17.1): Text processing and basic sentiment analysis
- **feedparser** (>=6.0.10): RSS feed parsing for news
- **requests** (>=2.31.0): HTTP library for API calls
- **nltk** (>=3.8.1): Natural language processing toolkit
- **pygooglenews** (>=0.1.2): Google News API wrapper

### Optional AI Dependencies

These dependencies enable advanced AI features. The application will work without them but with limited functionality:

- **torch** (>=2.0.0): PyTorch deep learning framework for LSTM models
- **tensorflow** (>=2.13.0): TensorFlow/Keras for GRU, Transformer, and Hybrid models
- **transformers** (>=4.30.0): HuggingFace Transformers library for FinBERT sentiment analysis
- **sentencepiece** (>=0.1.99): Tokenization library required by FinBERT
- **protobuf** (>=3.20.0): Protocol buffers required by transformers

**Note:** Without these AI dependencies, the application will use:
- Simple Linear Regression instead of deep learning models
- TextBlob instead of FinBERT for sentiment analysis

## Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/beraerr/crypto-analysis-dashboard.git
cd crypto-analysis-dashboard
```

### Step 2: Create Virtual Environment (Recommended)

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies

**Basic Installation (Core Features Only):**
```bash
pip install -r requirements.txt
```

**Full Installation (With All AI Features):**
```bash
pip install -r requirements.txt
pip install torch tensorflow transformers sentencepiece protobuf
```

**Note:** PyTorch and TensorFlow are large packages. Installation may take several minutes.

### Step 4: Download NLTK Data

The application will automatically download required NLTK data on first run, but you can pre-download:

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

## Running the Application

### Option 1: Using run.py (Recommended)

```bash
python run.py
```

### Option 2: Direct Streamlit Command

```bash
streamlit run src/app.py
```

### Option 3: Install as Package

```bash
pip install -e .
crypto-dashboard
```

The application will automatically open in your browser at `http://localhost:8501`

If it doesn't open automatically, navigate to:
- Local: `http://localhost:8501`
- Network: Check the terminal output for the network URL

## Usage Guide

1. **Select Cryptocurrency**: Choose from the sidebar (Bitcoin, Ethereum, etc.)
2. **Choose Time Period**: Select historical data range (1 month to 5 years)
3. **Enable Technical Indicators**: Toggle RSI, SMA, MACD, Bollinger Bands
4. **Configure AI Prediction**: 
   - Enable price prediction
   - Select prediction model (LSTM, GRU, Transformer, Hybrid, Ensemble)
   - Set prediction days (3-14 days)
5. **Enable Sentiment Analysis**: 
   - Toggle news analysis
   - Enable FinBERT for advanced sentiment (if installed)
   - Set number of articles to analyze
6. **View Results**: 
   - Interactive charts with predictions
   - Technical analysis notes
   - Sentiment scores and news articles
   - Fear & Greed Index

## AI Models Explained

### Price Prediction Models

#### 1. Simple Linear Regression (Baseline)

**What it does:**
- Fits a straight line to the last 30 days of closing prices
- Extrapolates the trend forward
- Uses only time as input feature

**When to use:**
- Quick estimates
- Baseline comparison
- When deep learning libraries are not available

**Limitations:**
- Assumes linear trend (rarely true for crypto)
- Ignores volatility and market cycles
- No feature engineering
- Low accuracy for volatile markets

#### 2. LSTM (Long Short-Term Memory)

**What it does:**
- Recurrent neural network designed for sequential data
- Uses 60-day lookback window to capture patterns
- Processes multiple features: Close, High, Low, Volume
- Maintains internal memory of past patterns
- 2-layer architecture with dropout regularization (0.2 dropout rate)
- Uses 50 hidden units per layer

**How it works:**
1. Normalizes input data (MinMaxScaler to 0-1 range)
2. Creates sequences of 60 days
3. Each sequence contains: [Close, High, Low, Volume] for each day
4. LSTM cells process sequences, learning temporal patterns
5. Final layer outputs predicted price
6. Predictions are inverse-transformed back to original scale

**Training:**
- 20 epochs (reduced for faster response)
- Batch size: 32
- Adam optimizer (learning rate: 0.001)
- Early stopping (patience: 10 epochs)
- Learning rate reduction on plateau

**When to use:**
- Capturing long-term price dependencies
- When you need good accuracy with reasonable training time
- For cryptocurrencies with clear trends

**Advantages:**
- Handles sequential patterns well
- Can remember information from 60+ days ago
- Works with multiple input features

**Limitations:**
- Training takes 30-60 seconds
- Requires PyTorch or TensorFlow
- May overfit on small datasets

#### 3. GRU (Gated Recurrent Unit)

**What it does:**
- Similar to LSTM but with simpler architecture
- Uses 60-day lookback window
- Processes Close, High, Low, Volume features
- 2-layer architecture with dropout
- 50 hidden units per layer

**How it works:**
- Similar to LSTM but with fewer gates (2 vs 3)
- Update gate: decides what information to keep
- Reset gate: decides what information to forget
- Simpler than LSTM, faster to train

**Training:**
- Same as LSTM: 20 epochs, batch size 32
- Uses TensorFlow/Keras
- Early stopping and learning rate reduction

**When to use:**
- Faster training needed
- Similar accuracy to LSTM but lighter
- Good balance of speed and performance

**Advantages:**
- Faster training than LSTM
- Lower memory usage
- Good performance on volatile markets

**Limitations:**
- Slightly less powerful than LSTM for very long sequences
- Requires TensorFlow

#### 4. Transformer Model

**What it does:**
- Attention-based architecture (no recurrence)
- Uses multi-head attention (4 heads) to focus on important time steps
- Processes 60-day sequences
- Uses Close, High, Low, Volume features
- Attention mechanism captures relationships between any two time steps

**How it works:**
1. Input sequences are embedded to 64 dimensions (d_model)
2. Multi-head attention (4 heads) computes attention weights
3. Each head focuses on different aspects of the data
4. Attention weights show which past days are most relevant
5. Feed-forward network processes attended features
6. Global average pooling combines all time steps
7. Final dense layer outputs prediction

**Architecture:**
- 2 transformer encoder layers
- 64-dimensional model (d_model)
- 4 attention heads
- 256-dimensional feed-forward network (4x d_model)
- Layer normalization and residual connections

**Training:**
- 20 epochs, batch size 32
- Adam optimizer (learning rate: 0.001)
- Early stopping and learning rate reduction

**When to use:**
- Long-range dependencies needed
- Complex pattern recognition
- When you want to understand which past days matter most

**Advantages:**
- Can attend to any past time step directly
- Better for long-range patterns
- Attention weights provide interpretability

**Limitations:**
- More parameters than LSTM/GRU
- Requires TensorFlow
- Training can be slower

#### 5. Hybrid Transformer-GRU

**What it does:**
- Combines Transformer attention mechanism with GRU sequential processing
- Two parallel branches:
  - Transformer branch: Captures long-range dependencies via attention
  - GRU branch: Processes sequential patterns
- Concatenates outputs from both branches
- Final dense layers combine features

**How it works:**
1. Input data goes to both branches simultaneously
2. Transformer branch:
   - Applies attention mechanism
   - Global average pooling
   - Outputs 64-dimensional vector
3. GRU branch:
   - Processes sequences with GRU layers
   - Outputs 50-dimensional vector
4. Concatenation: Combines both (64 + 50 = 114 dimensions)
5. Dense layers: 64 units with ReLU activation
6. Final output: Single price prediction

**Architecture:**
- Transformer: 64 d_model, 4 heads, 2 layers
- GRU: 50 units, 2 layers
- Combined: 114 -> 64 -> 1

**Training:**
- 20 epochs, batch size 32
- Same optimization as other models

**When to use:**
- Best accuracy needed
- State-of-the-art performance
- When you can wait for training

**Advantages:**
- Best of both worlds (attention + sequential)
- Highest accuracy among single models
- Research-proven architecture

**Limitations:**
- Slowest training (combines two models)
- Requires TensorFlow
- More complex architecture

#### 6. Ensemble Model

**What it does:**
- Combines predictions from all available models
- Uses weighted averaging:
  - Hybrid Transformer-GRU: 1.5x weight (most accurate)
  - Transformer: 1.2x weight
  - LSTM: 1.0x weight
  - GRU: 1.0x weight
- Trains all models, then averages their predictions

**How it works:**
1. Trains each model independently
2. Each model makes predictions
3. Predictions are weighted by model importance
4. Final prediction = weighted average

**Training:**
- Trains all models (15 epochs each, reduced for speed)
- Can take 2-3 minutes total

**When to use:**
- Maximum accuracy needed
- Most robust predictions
- Production use

**Advantages:**
- Most accurate (combines strengths of all models)
- Most robust (reduces individual model errors)
- Best for real trading decisions

**Limitations:**
- Slowest (trains multiple models)
- Requires both PyTorch and TensorFlow
- High memory usage

### Sentiment Analysis Models

#### FinBERT (Financial BERT)

**What it does:**
- BERT model fine-tuned specifically on financial texts
- Pre-trained on large corpus of financial news and reports
- Understands financial terminology and context
- Classifies sentiment as: positive, negative, or neutral
- Provides confidence scores (0-1)

**How it works:**
1. Tokenizes input text using SentencePiece
2. Adds special tokens ([CLS], [SEP])
3. Processes through 12 transformer layers
4. [CLS] token contains sentence-level representation
5. Classification head outputs sentiment label and score
6. Converts to polarity score (-1 to +1)

**Model Details:**
- Base model: ProsusAI/finbert (from HuggingFace)
- 110M parameters
- Trained on financial news, earnings reports, analyst reports
- Better than general BERT for financial texts

**When to use:**
- Analyzing financial news
- When accuracy matters
- For production sentiment analysis

**Advantages:**
- Domain-specific (understands financial context)
- High accuracy on financial texts
- Pre-trained (no training needed)

**Limitations:**
- Requires transformers library (~500MB download)
- Slower than TextBlob
- Requires internet for first-time model download

#### Multi-Source Sentiment Fusion

**What it does:**
- Combines sentiment from multiple news sources
- Applies source credibility weights:
  - Reuters, Bloomberg: 1.5x (most trusted)
  - Financial Times, Wall Street Journal: 1.4x
  - CoinDesk, CoinTelegraph: 1.3x
  - Forbes, CNBC, BBC: 1.2x
  - Others: 1.0x
- Temporal weighting: Recent news weighted more heavily
- Calculates weighted average sentiment score

**How it works:**
1. Fetches news from multiple sources (Google News)
2. Analyzes each article with FinBERT or TextBlob
3. Applies source weight based on credibility
4. Applies temporal weight (recent = higher)
5. Calculates weighted average: sum(polarity * source_weight * temporal_weight) / sum(weights)
6. Provides overall sentiment score and breakdown

**When to use:**
- Comprehensive market sentiment analysis
- When you need reliable sentiment scores
- For trading decisions

**Advantages:**
- Accounts for source credibility
- Reduces bias from single sources
- More reliable than single-source analysis

**Limitations:**
- Requires multiple news sources
- Slower than single-source analysis

#### TextBlob (Fallback)

**What it does:**
- Simple rule-based sentiment analysis
- Uses pattern matching and word lists
- Provides polarity (-1 to +1) and subjectivity (0 to 1)

**How it works:**
- Analyzes word patterns
- Matches against sentiment lexicons
- Calculates average polarity

**When to use:**
- When FinBERT is not available
- Quick sentiment analysis
- Lightweight applications

**Advantages:**
- Fast
- No external dependencies (except TextBlob)
- Works offline

**Limitations:**
- Less accurate than FinBERT
- Doesn't understand financial context
- Simple pattern matching

## Technical Indicators

### RSI (Relative Strength Index)

- **Default Period**: 14 days
- **Range**: 0-100
- **Overbought**: > 70 (price may decline)
- **Oversold**: < 30 (price may rise)
- **Calculation**: Compares magnitude of recent gains to recent losses

### SMA (Simple Moving Average)

- **Short SMA**: Default 50 days
- **Long SMA**: Default 200 days
- **Uptrend**: Short SMA > Long SMA
- **Downtrend**: Short SMA < Long SMA
- **Calculation**: Average of closing prices over period

### MACD (Moving Average Convergence Divergence)

- **Fast EMA**: 12 days
- **Slow EMA**: 26 days
- **Signal Line**: 9-day EMA of MACD
- **Bullish**: MACD > Signal Line
- **Bearish**: MACD < Signal Line
- **Calculation**: Difference between fast and slow EMAs

### Bollinger Bands

- **Period**: 20 days
- **Standard Deviation**: 2
- **Upper Band**: SMA + (2 * StdDev)
- **Lower Band**: SMA - (2 * StdDev)
- **Volatility Indicator**: Bands widen during high volatility

## Project Structure

```
src/
├── app.py              # Main Streamlit application
├── models/             # AI prediction and sentiment models
│   ├── predictors.py   # LSTM, GRU, Transformer, Hybrid, Ensemble
│   └── sentiment.py    # FinBERT, Multi-source sentiment
├── data/               # Data fetching and processing
│   ├── fetchers.py     # Yahoo Finance, News, Fear & Greed
│   └── processors.py   # Data preprocessing
├── indicators/         # Technical indicators
│   └── technical.py    # RSI, SMA, MACD, Bollinger Bands
├── ui/                 # UI components
│   ├── sidebar.py      # Sidebar configuration
│   └── components.py   # Reusable UI components
└── utils/              # Utilities and configuration
    ├── config.py       # Configuration management
    └── helpers.py      # Helper functions
```

See [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) for detailed architecture documentation.

## Performance Notes

### First Run

- FinBERT model download: ~500MB (one-time, cached)
- Model training: 30-60 seconds per model
- Ensemble training: 2-3 minutes (trains all models)

### Subsequent Runs

- Models are cached (no re-training)
- FinBERT is cached (no re-download)
- Much faster loading

### Memory Requirements

- Basic (no AI): ~500MB RAM
- With AI models: ~2-4GB RAM
- Ensemble: ~4GB RAM

### CPU vs GPU

- Current setup: CPU-only (works on any machine)
- GPU support: Can be enabled by installing GPU versions of PyTorch/TensorFlow
- GPU speeds up training significantly (5-10x faster)

## Troubleshooting

### "ModuleNotFoundError: No module named 'streamlit'"

**Solution:** Install dependencies:
```bash
pip install -r requirements.txt
```

### "FinBERT download failed"

**Solution:** 
- Check internet connection
- Try: `pip install --upgrade transformers`
- Application will fallback to TextBlob automatically

### "Out of memory"

**Solution:**
- Reduce batch_size in model training (edit src/models/predictors.py)
- Use fewer models in ensemble
- Close other applications

### "Model training too slow"

**Solution:**
- Reduce epochs (default: 20, can reduce to 10)
- Use simpler models (GRU instead of Hybrid)
- Install GPU versions if you have NVIDIA GPU

### "CUDA not available"

**Solution:**
- This is normal if you don't have NVIDIA GPU
- Models will use CPU (slower but works)
- Install CUDA toolkit if you want GPU acceleration

## Disclaimer

This application is for educational and analysis purposes only.

- This is NOT investment advice
- Cryptocurrency investments carry high risk
- Make investment decisions at your own risk
- Past performance does not guarantee future results
- AI predictions are estimates, not guarantees

## License

This project is open source and available for educational purposes.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## References

- Hybrid Transformer-GRU: Research paper on combining attention and sequential models
- FinBERT: ProsusAI/finbert on HuggingFace
- Technical Indicators: Standard financial analysis indicators
