"""
Advanced Sentiment Analysis for Financial News
Implements:
- FinBERT (Financial domain-specific BERT)
- BiLSTM for sentiment processing
- Multi-source sentiment fusion
- Weighted sentiment scoring
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from transformers import pipeline
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Transformers library not available. Install with: pip install transformers")

try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False
    print("TextBlob not available. Install with: pip install textblob")

try:
    from tensorflow import keras
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Embedding, Dropout, Input
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("TensorFlow not available for BiLSTM sentiment model")


class FinBERTSentimentAnalyzer:
    """
    FinBERT-based sentiment analyzer for financial news
    Uses pre-trained FinBERT model fine-tuned on financial texts
    """
    
    def __init__(self, model_name='ProsusAI/finbert'):
        """
        Initialize FinBERT model
        model_name: HuggingFace model identifier
        Options:
        - 'ProsusAI/finbert' (default, sentiment analysis)
        - 'yiyanghkust/finbert-tone' (tone analysis)
        """
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        self.initialized = False
        
        if TRANSFORMERS_AVAILABLE:
            try:
                self._initialize_model()
            except Exception as e:
                print(f"Warning: Could not initialize FinBERT: {str(e)}")
                print("Falling back to TextBlob sentiment analysis")
    
    def _initialize_model(self):
        """Initialize FinBERT model and tokenizer"""
        try:
            # Try to load FinBERT
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self.pipeline = pipeline("sentiment-analysis", 
                                    model=self.model, 
                                    tokenizer=self.tokenizer,
                                    device=-1)  # Use CPU, set to 0 for GPU
            self.initialized = True
        except Exception as e:
            # Fallback to general BERT or TextBlob
            print(f"FinBERT initialization failed: {str(e)}")
            try:
                # Try general BERT sentiment
                self.pipeline = pipeline("sentiment-analysis", device=-1)
                self.initialized = True
            except:
                self.initialized = False
    
    def analyze(self, text):
        """
        Analyze sentiment of financial text
        
        Returns:
        {
            'label': 'positive'/'negative'/'neutral',
            'score': confidence score (0-1),
            'polarity': normalized polarity (-1 to 1),
            'sentiment': 'pozitif'/'negatif'/'nötr'
        }
        """
        if not text or len(text.strip()) == 0:
            return {
                'label': 'neutral',
                'score': 0.5,
                'polarity': 0.0,
                'sentiment': 'nötr'
            }
        
        if self.initialized and self.pipeline:
            try:
                # FinBERT/BERT analysis
                result = self.pipeline(text[:512])[0]  # Limit to 512 tokens
                
                label = result['label'].lower()
                score = result['score']
                
                # Convert to polarity (-1 to 1)
                if 'positive' in label or 'bullish' in label:
                    polarity = score
                    sentiment = 'pozitif'
                elif 'negative' in label or 'bearish' in label:
                    polarity = -score
                    sentiment = 'negatif'
                else:
                    polarity = 0.0
                    sentiment = 'nötr'
                
                return {
                    'label': label,
                    'score': score,
                    'polarity': polarity,
                    'sentiment': sentiment
                }
            except Exception as e:
                # Fallback to TextBlob
                return self._textblob_fallback(text)
        else:
            return self._textblob_fallback(text)
    
    def _textblob_fallback(self, text):
        """Fallback to TextBlob if FinBERT is not available"""
        if not TEXTBLOB_AVAILABLE:
            return {
                'label': 'neutral',
                'score': 0.5,
                'polarity': 0.0,
                'sentiment': 'nötr'
            }
        
        try:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            
            if polarity > 0.1:
                sentiment = 'pozitif'
                label = 'positive'
            elif polarity < -0.1:
                sentiment = 'negatif'
                label = 'negative'
            else:
                sentiment = 'nötr'
                label = 'neutral'
            
            return {
                'label': label,
                'score': abs(polarity) if abs(polarity) > 0.1 else 0.5,
                'polarity': polarity,
                'sentiment': sentiment
            }
        except:
            return {
                'label': 'neutral',
                'score': 0.5,
                'polarity': 0.0,
                'sentiment': 'nötr'
            }
    
    def analyze_batch(self, texts):
        """Analyze multiple texts"""
        results = []
        for text in texts:
            results.append(self.analyze(text))
        return results


class BiLSTMSentimentModel:
    """
    Bidirectional LSTM model for sentiment analysis
    Can be trained on financial news data
    """
    
    def __init__(self, vocab_size=10000, embedding_dim=128, lstm_units=64, max_length=200):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        self.max_length = max_length
        self.tokenizer = Tokenizer(num_words=vocab_size, oov_token='<OOV>')
        self.model = None
        
    def build_model(self):
        """Build BiLSTM sentiment model"""
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow required for BiLSTM model")
        
        model = Sequential([
            Embedding(self.vocab_size, self.embedding_dim, input_length=self.max_length),
            Bidirectional(LSTM(self.lstm_units, return_sequences=True)),
            Dropout(0.5),
            Bidirectional(LSTM(self.lstm_units)),
            Dropout(0.5),
            Dense(64, activation='relu'),
            Dropout(0.5),
            Dense(1, activation='sigmoid')  # Binary classification: positive/negative
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def prepare_data(self, texts, labels=None):
        """Prepare text data for training/prediction"""
        if labels is None:
            # Prediction mode
            sequences = self.tokenizer.texts_to_sequences(texts)
            padded = pad_sequences(sequences, maxlen=self.max_length, padding='post')
            return padded
        else:
            # Training mode
            self.tokenizer.fit_on_texts(texts)
            sequences = self.tokenizer.texts_to_sequences(texts)
            padded = pad_sequences(sequences, maxlen=self.max_length, padding='post')
            return padded, np.array(labels)
    
    def train(self, texts, labels, epochs=10, batch_size=32, validation_split=0.2):
        """Train the BiLSTM model"""
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow required")
        
        X, y = self.prepare_data(texts, labels)
        self.model = self.build_model()
        
        self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=0
        )
        
        return self.model
    
    def predict(self, texts):
        """Predict sentiment for texts"""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        X = self.prepare_data(texts)
        predictions = self.model.predict(X, verbose=0)
        
        results = []
        for pred in predictions:
            score = pred[0]
            if score > 0.6:
                sentiment = 'pozitif'
                polarity = score
            elif score < 0.4:
                sentiment = 'negatif'
                polarity = -(1 - score)
            else:
                sentiment = 'nötr'
                polarity = 0.0
            
            results.append({
                'sentiment': sentiment,
                'polarity': polarity,
                'score': score
            })
        
        return results


class MultiSourceSentimentAnalyzer:
    """
    Multi-source sentiment analyzer combining:
    - FinBERT for news articles
    - TextBlob as fallback
    - Source weighting
    - Temporal weighting (recent news weighted more)
    """
    
    def __init__(self):
        self.finbert = FinBERTSentimentAnalyzer()
        self.source_weights = {
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
        }
    
    def get_source_weight(self, source):
        """Get weight for news source"""
        source_lower = source.lower()
        for key, weight in self.source_weights.items():
            if key in source_lower:
                return weight
        return self.source_weights['default']
    
    def analyze_news_list(self, news_list, use_temporal_weighting=True):
        """
        Analyze a list of news articles with multi-source weighting
        
        Args:
            news_list: List of dicts with 'title', 'summary', 'source', 'published', etc.
            use_temporal_weighting: Weight recent news more heavily
        
        Returns:
            {
                'overall_sentiment': 'pozitif'/'negatif'/'nötr',
                'weighted_score': float,
                'total_score': float,
                'positive_count': int,
                'negative_count': int,
                'neutral_count': int,
                'message': str,
                'individual_scores': list
            }
        """
        if not news_list:
            return {
                'overall_sentiment': 'nötr',
                'weighted_score': 0.0,
                'total_score': 0.0,
                'positive_count': 0,
                'negative_count': 0,
                'neutral_count': 0,
                'message': 'No news available',
                'individual_scores': []
            }
        
        total_weighted_score = 0.0
        total_weight = 0.0
        sentiment_counts = {'pozitif': 0, 'negatif': 0, 'nötr': 0}
        individual_scores = []
        
        for news in news_list:
            # Combine title and summary
            text = f"{news.get('title', '')} {news.get('summary', '')}"
            
            # Analyze sentiment
            sentiment_result = self.finbert.analyze(text)
            polarity = sentiment_result['polarity']
            
            # Get source weight
            source_weight = self.get_source_weight(news.get('source', 'Unknown'))
            
            # Temporal weight (if published date available)
            temporal_weight = 1.0
            if use_temporal_weighting and 'published' in news:
                # Recent news gets higher weight (simplified)
                temporal_weight = 1.2  # Could be more sophisticated
            
            # Combined weight
            combined_weight = source_weight * temporal_weight
            
            # Accumulate scores
            total_weighted_score += polarity * combined_weight
            total_weight += combined_weight
            
            # Count sentiments
            sentiment_counts[sentiment_result['sentiment']] += 1
            
            # Store individual score
            individual_scores.append({
                'title': news.get('title', ''),
                'source': news.get('source', 'Unknown'),
                'sentiment': sentiment_result['sentiment'],
                'polarity': polarity,
                'weight': combined_weight
            })
        
        # Calculate weighted average
        if total_weight > 0:
            weighted_avg = total_weighted_score / total_weight
        else:
            weighted_avg = sum(s['polarity'] for s in individual_scores) / len(individual_scores) if individual_scores else 0.0
        
        # Determine overall sentiment
        if weighted_avg > 0.1:
            overall = 'pozitif'
            message = f"Positive sentiment ({weighted_avg:.3f}): Market appears optimistic. {sentiment_counts['pozitif']} positive articles."
        elif weighted_avg < -0.1:
            overall = 'negatif'
            message = f"Negative sentiment ({weighted_avg:.3f}): Market shows concern. {sentiment_counts['negatif']} negative articles."
        else:
            overall = 'nötr'
            message = f"Neutral sentiment ({weighted_avg:.3f}): Market appears balanced. {sentiment_counts['nötr']} neutral articles."
        
        return {
            'overall_sentiment': overall,
            'weighted_score': weighted_avg,
            'total_score': sum(s['polarity'] for s in individual_scores) / len(individual_scores) if individual_scores else 0.0,
            'positive_count': sentiment_counts['pozitif'],
            'negative_count': sentiment_counts['negatif'],
            'neutral_count': sentiment_counts['nötr'],
            'message': message,
            'individual_scores': individual_scores,
            'score': weighted_avg
        }
    
    def analyze_text(self, text):
        """Analyze single text"""
        return self.finbert.analyze(text)


def get_sentiment_analyzer(use_finbert=True):
    """
    Factory function to get appropriate sentiment analyzer
    
    Args:
        use_finbert: Whether to use FinBERT (requires transformers library)
    
    Returns:
        Sentiment analyzer instance
    """
    if use_finbert and TRANSFORMERS_AVAILABLE:
        return MultiSourceSentimentAnalyzer()
    elif TEXTBLOB_AVAILABLE:
        # Fallback to TextBlob
        class TextBlobAnalyzer:
            def analyze(self, text):
                blob = TextBlob(text)
                polarity = blob.sentiment.polarity
                if polarity > 0.1:
                    sentiment = 'pozitif'
                elif polarity < -0.1:
                    sentiment = 'negatif'
                else:
                    sentiment = 'nötr'
                return {
                    'sentiment': sentiment,
                    'polarity': polarity,
                    'score': abs(polarity)
                }
        return TextBlobAnalyzer()
    else:
        raise ImportError("No sentiment analysis library available. Install transformers or textblob.")

