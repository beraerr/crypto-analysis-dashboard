"""
Configuration management for the application
"""

import os
from typing import Dict, Any
from dataclasses import dataclass, field

@dataclass
class ModelConfig:
    """Configuration for AI prediction models"""
    sequence_length: int = 60
    hidden_units: int = 50
    num_layers: int = 2
    dropout: float = 0.2
    d_model: int = 64
    num_heads: int = 4
    gru_units: int = 50
    epochs: int = 20
    batch_size: int = 32
    validation_split: float = 0.2
    early_stopping_patience: int = 10
    learning_rate: float = 0.001

@dataclass
class SentimentConfig:
    """Configuration for sentiment analysis"""
    finbert_model: str = 'ProsusAI/finbert'
    use_finbert: bool = True
    source_weights: Dict[str, float] = field(default_factory=lambda: {
        'reuters': 1.5,
        'bloomberg': 1.5,
        'coindesk': 1.3,
        'cointelegraph': 1.3,
        'forbes': 1.2,
        'cnbc': 1.2,
        'bbc': 1.2,
        'financial times': 1.4,
        'wall street journal': 1.4,
        'default': 1.0
    })
    temporal_weighting: bool = True
    max_articles: int = 10

@dataclass
class DataConfig:
    """Configuration for data fetching"""
    cache_ttl: int = 300  # seconds
    default_period: str = "1y"
    supported_cryptos: Dict[str, str] = field(default_factory=lambda: {
        "Bitcoin (BTC-USD)": "BTC-USD",
        "Ethereum (ETH-USD)": "ETH-USD",
        "Binance Coin (BNB-USD)": "BNB-USD",
        "Cardano (ADA-USD)": "ADA-USD",
        "Solana (SOL-USD)": "SOL-USD",
        "Ripple (XRP-USD)": "XRP-USD",
        "Polkadot (DOT-USD)": "DOT-USD",
        "Dogecoin (DOGE-USD)": "DOGE-USD"
    })
    period_options: Dict[str, str] = field(default_factory=lambda: {
        "1 Ay": "1mo",
        "3 Ay": "3mo",
        "6 Ay": "6mo",
        "1 Yıl": "1y",
        "2 Yıl": "2y",
        "5 Yıl": "5y"
    })

@dataclass
class UIConfig:
    """Configuration for UI settings"""
    page_title: str = "Crypto Analysis Dashboard"
    page_icon: str = "📈"
    layout: str = "wide"
    initial_sidebar_state: str = "expanded"
    default_prediction_days: int = 7
    default_news_count: int = 10

@dataclass
class AppConfig:
    """Main application configuration"""
    model: ModelConfig = field(default_factory=ModelConfig)
    sentiment: SentimentConfig = field(default_factory=SentimentConfig)
    data: DataConfig = field(default_factory=DataConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    
    # Environment variables
    debug: bool = os.getenv('DEBUG', 'False').lower() == 'true'
    log_level: str = os.getenv('LOG_LEVEL', 'INFO')
    
    @classmethod
    def from_env(cls) -> 'AppConfig':
        """Create config from environment variables"""
        return cls()

# Global configuration instance
config = AppConfig.from_env()

