# AI Models Guide

This document explains the advanced AI models implemented in this cryptocurrency analysis dashboard.

## 🚀 Quick Start

1. Install dependencies:
```bash
pip install -r requirements_crypto.txt
```

2. Run the application:
```bash
streamlit run crypto_prediction_app.py
```

3. Select an AI model from the sidebar under "🤖 AI Model Selection"

## 📊 Available Models

### 1. Simple Linear Regression (Baseline)
- **Speed**: ⚡⚡⚡⚡⚡ Very Fast
- **Accuracy**: ⭐⭐ Low
- **Use Case**: Quick estimates, baseline comparison
- **Best For**: Simple trend extrapolation

### 2. LSTM (Long Short-Term Memory)
- **Speed**: ⚡⚡⚡ Medium
- **Accuracy**: ⭐⭐⭐⭐ High
- **Architecture**: 2-layer LSTM with dropout
- **Features**: Uses Close, High, Low, Volume
- **Lookback**: 60 days
- **Best For**: Capturing long-term price patterns

### 3. GRU (Gated Recurrent Unit)
- **Speed**: ⚡⚡⚡⚡ Fast
- **Accuracy**: ⭐⭐⭐⭐ High
- **Architecture**: 2-layer GRU with dropout
- **Features**: Uses Close, High, Low, Volume
- **Lookback**: 60 days
- **Best For**: Faster training with good accuracy

### 4. Transformer
- **Speed**: ⚡⚡⚡ Medium
- **Accuracy**: ⭐⭐⭐⭐⭐ Very High
- **Architecture**: Multi-head attention with 4 heads
- **Features**: Uses Close, High, Low, Volume
- **Lookback**: 60 days
- **Best For**: Long-range dependency patterns

### 5. Hybrid Transformer-GRU ⭐ RECOMMENDED
- **Speed**: ⚡⚡⚡ Medium
- **Accuracy**: ⭐⭐⭐⭐⭐ Very High
- **Architecture**: Combines Transformer attention + GRU sequential processing
- **Features**: Uses Close, High, Low, Volume
- **Lookback**: 60 days
- **Best For**: Best balance of accuracy and efficiency (State-of-the-art)

### 6. Ensemble (All Models Combined)
- **Speed**: ⚡⚡ Slow (trains all models)
- **Accuracy**: ⭐⭐⭐⭐⭐ Very High
- **Architecture**: Weighted average of all models
- **Weights**: Hybrid (1.5x), Transformer (1.2x), LSTM/GRU (1.0x)
- **Best For**: Most robust predictions, production use

## 🧠 Sentiment Analysis Models

### FinBERT (Financial BERT)
- **Model**: `ProsusAI/finbert` from HuggingFace
- **Accuracy**: ⭐⭐⭐⭐⭐ Very High for financial texts
- **Features**:
  - Pre-trained on financial news
  - Understands financial terminology
  - Better than general sentiment models
- **Fallback**: TextBlob if FinBERT unavailable

### Multi-Source Sentiment Fusion
- Combines sentiment from multiple news sources
- Weighted by source credibility:
  - Reuters, Bloomberg: 1.5x
  - CoinDesk, CoinTelegraph: 1.3x
  - Forbes, CNBC, BBC: 1.2x
  - Others: 1.0x
- Temporal weighting (recent news weighted more)

## 📈 Model Performance Tips

1. **For Real-time Trading**: Use Ensemble or Hybrid Transformer-GRU
2. **For Quick Analysis**: Use GRU or LSTM
3. **For Research**: Compare all models
4. **For Sentiment**: Always enable FinBERT if available

## 🔧 Technical Details

### Data Preprocessing
- **Normalization**: MinMaxScaler (0-1 range)
- **Features**: Close, High, Low, Volume
- **Sequence Length**: 60 days
- **Train/Validation Split**: 80/20

### Training Parameters
- **Epochs**: 15-20 (reduced for faster response)
- **Batch Size**: 32
- **Optimizer**: Adam (learning_rate=0.001)
- **Regularization**: Dropout (0.2-0.5)
- **Early Stopping**: Patience=10 epochs
- **Learning Rate Reduction**: Factor=0.5, Patience=5

### Model Evaluation
- **Metrics**: MAE, MSE, RMSE, R², MAPE
- **Validation**: Uses validation split during training
- **Backtesting**: Can be extended with historical data

## 🚨 Important Notes

1. **First Run**: Models will download pre-trained weights (FinBERT ~500MB)
2. **Training Time**: First prediction takes longer (model training)
3. **Caching**: Models are cached after first training
4. **GPU Support**: Set `device=0` in code for GPU acceleration
5. **Memory**: Requires ~2-4GB RAM for all models

## 🔮 Future Enhancements

- [ ] Add more features (on-chain metrics, social signals)
- [ ] Implement backtesting framework
- [ ] Add confidence intervals for predictions
- [ ] Real-time model retraining
- [ ] Model performance comparison dashboard
- [ ] Export predictions to CSV/JSON
- [ ] Alert system for price predictions

## 📚 References

- **Hybrid Transformer-GRU**: [arXiv:2504.17079](https://arxiv.org/abs/2504.17079)
- **FinBERT**: [ProsusAI/finbert](https://huggingface.co/ProsusAI/finbert)
- **BiLSTM Sentiment**: [arXiv:2411.12748](https://arxiv.org/abs/2411.12748)

## 🐛 Troubleshooting

### "TensorFlow/PyTorch not available"
```bash
pip install torch tensorflow transformers
```

### "FinBERT download failed"
- Check internet connection
- Try: `pip install --upgrade transformers`
- Fallback to TextBlob will be used automatically

### "Out of memory"
- Reduce batch_size in model training
- Use fewer models in ensemble
- Close other applications

### "Model training too slow"
- Reduce epochs (default: 15-20)
- Use GPU if available
- Start with simpler models (GRU/LSTM)

