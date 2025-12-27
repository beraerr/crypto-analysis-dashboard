# 📁 Project Structure

This document describes the architecture and organization of the Cryptocurrency Analysis Dashboard.

## 🏗️ Architecture Overview

```
ai_crypto_analyzer/
├── src/                          # Main source code
│   ├── __init__.py              # Package initialization
│   ├── app.py                   # Main Streamlit application
│   │
│   ├── models/                  # AI Models
│   │   ├── __init__.py
│   │   ├── predictors.py        # Price prediction models (LSTM, GRU, Transformer, etc.)
│   │   └── sentiment.py         # Sentiment analysis models (FinBERT, BiLSTM)
│   │
│   ├── data/                    # Data Layer
│   │   ├── __init__.py
│   │   ├── fetchers.py          # Data fetching (Yahoo Finance, News, Fear & Greed)
│   │   └── processors.py        # Data preprocessing and transformation
│   │
│   ├── indicators/              # Technical Indicators
│   │   ├── __init__.py
│   │   └── technical.py         # RSI, SMA, MACD, Bollinger Bands
│   │
│   ├── ui/                      # UI Components
│   │   ├── __init__.py
│   │   ├── sidebar.py           # Sidebar configuration and controls
│   │   └── components.py        # Reusable UI components
│   │
│   └── utils/                   # Utilities
│       ├── __init__.py
│       ├── config.py            # Configuration management
│       └── helpers.py           # Helper functions
│
├── config/                      # Configuration files (future)
│
├── tests/                        # Unit tests (future)
│
├── run.py                        # Application entry point
├── setup.py                     # Package setup script
├── requirements_crypto.txt       # Python dependencies
├── README.md                     # Main documentation
├── AI_MODELS_GUIDE.md           # AI models documentation
├── IMPLEMENTATION_SUMMARY.md    # Implementation details
├── PROJECT_STRUCTURE.md         # This file
└── .gitignore                   # Git ignore rules
```

## 📦 Module Descriptions

### `src/app.py`
Main Streamlit application that orchestrates all components:
- Renders UI using components from `ui/`
- Fetches data using `data/fetchers.py`
- Calculates indicators using `indicators/technical.py`
- Runs predictions using `models/predictors.py`
- Analyzes sentiment using `models/sentiment.py`

### `src/models/`
**predictors.py**: Contains all price prediction models
- `LSTMPredictor`: Long Short-Term Memory network
- `GRUPredictor`: Gated Recurrent Unit
- `TransformerPredictor`: Attention-based Transformer
- `HybridTransformerGRU`: Hybrid model combining Transformer + GRU
- `EnsemblePredictor`: Combines all models

**sentiment.py**: Contains sentiment analysis models
- `FinBERTSentimentAnalyzer`: Financial domain-specific BERT
- `MultiSourceSentimentAnalyzer`: Multi-source sentiment fusion
- `BiLSTMSentimentModel`: Bidirectional LSTM for sentiment

### `src/data/`
**fetchers.py**: Data fetching classes
- `CryptoDataFetcher`: Fetches price data from Yahoo Finance
- `NewsFetcher`: Fetches news from Google News
- `FearGreedIndexFetcher`: Fetches Fear & Greed Index

**processors.py**: Data preprocessing
- `DataPreprocessor`: Normalization, sequence creation, feature engineering

### `src/indicators/`
**technical.py**: Technical indicator calculations
- `TechnicalIndicators`: Class with static methods for:
  - RSI (Relative Strength Index)
  - SMA/EMA (Moving Averages)
  - MACD (Moving Average Convergence Divergence)
  - Bollinger Bands
  - Analysis methods for each indicator

### `src/ui/`
**sidebar.py**: Sidebar configuration
- `render_sidebar()`: Renders sidebar and returns user selections

**components.py**: Reusable UI components
- `render_header()`: Main header with disclaimer
- `render_price_metrics()`: Price metrics display
- `render_news_card()`: News article card
- `render_fear_greed_index()`: Fear & Greed Index display
- `render_sentiment_summary()`: Sentiment analysis summary
- `render_prediction_table()`: Prediction results table

### `src/utils/`
**config.py**: Configuration management
- `ModelConfig`: AI model configuration
- `SentimentConfig`: Sentiment analysis configuration
- `DataConfig`: Data fetching configuration
- `UIConfig`: UI configuration
- `AppConfig`: Main application configuration

**helpers.py**: Utility functions
- `safe_divide()`: Safe division
- `calculate_percentage_change()`: Percentage calculations
- `format_currency()`: Currency formatting
- `validate_dataframe()`: Data validation
- `check_dependencies()`: Dependency checking

## 🔄 Data Flow

```
User Input (Sidebar)
    ↓
app.py (Main Application)
    ↓
┌─────────────────────────────────────┐
│  Data Layer                         │
│  - CryptoDataFetcher               │
│  - NewsFetcher                     │
│  - FearGreedIndexFetcher           │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  Processing Layer                   │
│  - DataPreprocessor                 │
│  - TechnicalIndicators             │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  AI Models Layer                    │
│  - Predictors (LSTM, GRU, etc.)     │
│  - Sentiment Analyzers (FinBERT)   │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  UI Layer                           │
│  - Components (charts, tables)      │
│  - Streamlit rendering              │
└─────────────────────────────────────┘
    ↓
User Output (Dashboard)
```

## 🎯 Design Principles

1. **Separation of Concerns**: Each module has a single responsibility
2. **Modularity**: Components can be used independently
3. **Configuration Management**: Centralized configuration in `utils/config.py`
4. **Error Handling**: Graceful fallbacks for missing dependencies
5. **Reusability**: UI components and utilities are reusable
6. **Testability**: Clear interfaces make testing easier

## 🚀 Running the Application

### Option 1: Using run.py
```bash
python run.py
```

### Option 2: Direct Streamlit
```bash
streamlit run src/app.py
```

### Option 3: Using setup.py (after installation)
```bash
pip install -e .
crypto-dashboard
```

## 📝 Adding New Features

### Adding a New Indicator
1. Add method to `src/indicators/technical.py`
2. Add UI option in `src/ui/sidebar.py`
3. Use in `src/app.py`

### Adding a New Model
1. Add class to `src/models/predictors.py` or `src/models/sentiment.py`
2. Export in `src/models/__init__.py`
3. Add option in `src/ui/sidebar.py`
4. Integrate in `src/app.py`

### Adding a New Data Source
1. Add fetcher class to `src/data/fetchers.py`
2. Add configuration to `src/utils/config.py`
3. Use in `src/app.py`

## 🔧 Configuration

Configuration is managed in `src/utils/config.py`:
- Model hyperparameters
- API endpoints
- UI settings
- Feature flags

Can be extended to support:
- YAML configuration files
- Environment variables
- Command-line arguments

## 🧪 Testing (Future)

Tests should be organized in `tests/`:
```
tests/
├── test_models/
├── test_data/
├── test_indicators/
└── test_ui/
```

## 📚 Documentation

- `README.md`: Main documentation
- `AI_MODELS_GUIDE.md`: AI models guide
- `IMPLEMENTATION_SUMMARY.md`: Implementation details
- `PROJECT_STRUCTURE.md`: This file

## 🔐 Security Considerations

- API keys should be in environment variables
- User input validation
- Rate limiting for API calls
- Error messages should not expose sensitive information

## 🎨 Future Improvements

- [ ] Add configuration files (YAML/JSON)
- [ ] Add unit tests
- [ ] Add integration tests
- [ ] Add logging system
- [ ] Add monitoring/metrics
- [ ] Add API layer for backend services
- [ ] Add database for caching
- [ ] Add CI/CD pipeline

