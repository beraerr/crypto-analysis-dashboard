"""
Advanced AI Models for Cryptocurrency Price Prediction
Implements state-of-the-art deep learning models:
- LSTM (Long Short-Term Memory)
- GRU (Gated Recurrent Unit)
- Transformer-based models
- Hybrid Transformer-GRU
- Ensemble methods
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available. Install with: pip install torch")

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Input, MultiHeadAttention, LayerNormalization, Add, Concatenate, GlobalAveragePooling1D
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("TensorFlow not available. Install with: pip install tensorflow")

class TimeSeriesDataset(Dataset):
    """PyTorch Dataset for time series data"""
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class LSTMPredictor:
    """
    LSTM (Long Short-Term Memory) model for cryptocurrency price prediction
    Effective for capturing long-term dependencies in time series data
    """
    
    def __init__(self, sequence_length=60, hidden_units=50, num_layers=2, dropout=0.2):
        self.sequence_length = sequence_length
        self.hidden_units = hidden_units
        self.num_layers = num_layers
        self.dropout = dropout
        self.scaler = MinMaxScaler()
        self.model = None
        self.use_torch = TORCH_AVAILABLE
        
    def prepare_data(self, data, lookback=None):
        """Prepare data for LSTM training"""
        if lookback is None:
            lookback = self.sequence_length
            
        # Use multiple features: Close, High, Low, Volume
        features = ['Close', 'High', 'Low', 'Volume']
        df = data[features].copy()
        
        # Normalize data
        scaled_data = self.scaler.fit_transform(df)
        
        X, y = [], []
        for i in range(lookback, len(scaled_data)):
            X.append(scaled_data[i-lookback:i])
            y.append(scaled_data[i, 0])  # Predict Close price
        
        return np.array(X), np.array(y)
    
    def build_model_tf(self, input_shape):
        """Build LSTM model using TensorFlow"""
        model = Sequential([
            LSTM(self.hidden_units, return_sequences=True, input_shape=input_shape),
            Dropout(self.dropout),
            LSTM(self.hidden_units, return_sequences=True),
            Dropout(self.dropout),
            LSTM(self.hidden_units),
            Dropout(self.dropout),
            Dense(1)
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        return model
    
    def build_model_torch(self, input_size):
        """Build LSTM model using PyTorch"""
        class LSTMModel(nn.Module):
            def __init__(self, input_size, hidden_units, num_layers, dropout):
                super(LSTMModel, self).__init__()
                self.hidden_units = hidden_units
                self.num_layers = num_layers
                self.lstm = nn.LSTM(input_size, hidden_units, num_layers, 
                                   batch_first=True, dropout=dropout if num_layers > 1 else 0)
                self.dropout = nn.Dropout(dropout)
                self.fc = nn.Linear(hidden_units, 1)
            
            def forward(self, x):
                lstm_out, _ = self.lstm(x)
                lstm_out = self.dropout(lstm_out[:, -1, :])
                output = self.fc(lstm_out)
                return output
        
        return LSTMModel(input_size, self.hidden_units, self.num_layers, self.dropout)
    
    def train(self, data, epochs=50, batch_size=32, validation_split=0.2):
        """Train the LSTM model"""
        X, y = self.prepare_data(data)
        
        if len(X) == 0:
            return None
        
        # Split data
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        if TF_AVAILABLE:
            self.model = self.build_model_tf((X_train.shape[1], X_train.shape[2]))
            
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7)
            ]
            
            self.model.fit(
                X_train, y_train,
                validation_data=(X_test, y_test),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=0
            )
            return self.model
        elif TORCH_AVAILABLE:
            self.model = self.build_model_torch(X_train.shape[2])
            criterion = nn.MSELoss()
            optimizer = optim.Adam(self.model.parameters(), lr=0.001)
            
            train_dataset = TimeSeriesDataset(X_train, y_train)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            
            self.model.train()
            for epoch in range(epochs):
                for batch_X, batch_y in train_loader:
                    optimizer.zero_grad()
                    outputs = self.model(batch_X)
                    loss = criterion(outputs.squeeze(), batch_y)
                    loss.backward()
                    optimizer.step()
            
            self.model.eval()
            return self.model
        else:
            raise ImportError("Neither TensorFlow nor PyTorch is available")
    
    def predict(self, data, days=7):
        """Make predictions"""
        if self.model is None:
            return None, None
        
        X, _ = self.prepare_data(data)
        if len(X) == 0:
            return None, None
        
        last_sequence = X[-1:]
        predictions = []
        
        if TF_AVAILABLE:
            current_input = last_sequence.copy()
            for _ in range(days):
                pred = self.model.predict(current_input, verbose=0)[0, 0]
                predictions.append(pred)
                # Update input sequence (simplified - in production, use actual features)
                new_row = np.append(current_input[0, 1:, :], 
                                   [[pred, pred, pred, current_input[0, -1, 3]]], axis=0)
                current_input = np.array([new_row])
        elif TORCH_AVAILABLE:
            self.model.eval()
            current_input = torch.FloatTensor(last_sequence)
            with torch.no_grad():
                for _ in range(days):
                    pred = self.model(current_input)[0, 0].item()
                    predictions.append(pred)
                    # Update input sequence
                    new_row = torch.cat([current_input[0, 1:, :], 
                                        torch.tensor([[pred, pred, pred, current_input[0, -1, 3]]])], dim=0)
                    current_input = new_row.unsqueeze(0)
        
        # Inverse transform predictions
        dummy_data = np.zeros((len(predictions), 4))
        dummy_data[:, 0] = predictions
        predictions_scaled = self.scaler.inverse_transform(dummy_data)[:, 0]
        
        last_date = data.index[-1]
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days, freq='D')
        
        return future_dates, predictions_scaled


class GRUPredictor:
    """
    GRU (Gated Recurrent Unit) model for cryptocurrency price prediction
    Lighter and faster than LSTM while maintaining good performance
    """
    
    def __init__(self, sequence_length=60, hidden_units=50, num_layers=2, dropout=0.2):
        self.sequence_length = sequence_length
        self.hidden_units = hidden_units
        self.num_layers = num_layers
        self.dropout = dropout
        self.scaler = MinMaxScaler()
        self.model = None
    
    def prepare_data(self, data, lookback=None):
        """Prepare data for GRU training"""
        if lookback is None:
            lookback = self.sequence_length
            
        features = ['Close', 'High', 'Low', 'Volume']
        df = data[features].copy()
        scaled_data = self.scaler.fit_transform(df)
        
        X, y = [], []
        for i in range(lookback, len(scaled_data)):
            X.append(scaled_data[i-lookback:i])
            y.append(scaled_data[i, 0])
        
        return np.array(X), np.array(y)
    
    def build_model(self, input_shape):
        """Build GRU model"""
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow required for GRU model")
        
        model = Sequential([
            GRU(self.hidden_units, return_sequences=True, input_shape=input_shape),
            Dropout(self.dropout),
            GRU(self.hidden_units, return_sequences=True),
            Dropout(self.dropout),
            GRU(self.hidden_units),
            Dropout(self.dropout),
            Dense(1)
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        return model
    
    def train(self, data, epochs=50, batch_size=32, validation_split=0.2):
        """Train the GRU model"""
        X, y = self.prepare_data(data)
        
        if len(X) == 0:
            return None
        
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        self.model = self.build_model((X_train.shape[1], X_train.shape[2]))
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7)
        ]
        
        self.model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=0
        )
        
        return self.model
    
    def predict(self, data, days=7):
        """Make predictions"""
        if self.model is None:
            return None, None
        
        X, _ = self.prepare_data(data)
        if len(X) == 0:
            return None, None
        
        last_sequence = X[-1:]
        predictions = []
        current_input = last_sequence.copy()
        
        for _ in range(days):
            pred = self.model.predict(current_input, verbose=0)[0, 0]
            predictions.append(pred)
            new_row = np.append(current_input[0, 1:, :], 
                               [[pred, pred, pred, current_input[0, -1, 3]]], axis=0)
            current_input = np.array([new_row])
        
        dummy_data = np.zeros((len(predictions), 4))
        dummy_data[:, 0] = predictions
        predictions_scaled = self.scaler.inverse_transform(dummy_data)[:, 0]
        
        last_date = data.index[-1]
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days, freq='D')
        
        return future_dates, predictions_scaled


class TransformerPredictor:
    """
    Transformer-based model for cryptocurrency price prediction
    Uses attention mechanism to capture long-range dependencies
    """
    
    def __init__(self, sequence_length=60, d_model=64, num_heads=4, num_layers=2, dropout=0.1):
        self.sequence_length = sequence_length
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout
        self.scaler = MinMaxScaler()
        self.model = None
    
    def prepare_data(self, data, lookback=None):
        """Prepare data for Transformer training"""
        if lookback is None:
            lookback = self.sequence_length
            
        features = ['Close', 'High', 'Low', 'Volume']
        df = data[features].copy()
        scaled_data = self.scaler.fit_transform(df)
        
        X, y = [], []
        for i in range(lookback, len(scaled_data)):
            X.append(scaled_data[i-lookback:i])
            y.append(scaled_data[i, 0])
        
        return np.array(X), np.array(y)
    
    def transformer_encoder(self, inputs, head_size, num_heads, ff_dim, dropout=0):
        """Transformer encoder block"""
        # Multi-head attention
        attention_output = MultiHeadAttention(
            key_dim=head_size, num_heads=num_heads, dropout=dropout
        )(inputs, inputs)
        attention_output = Dropout(dropout)(attention_output)
        out1 = LayerNormalization(epsilon=1e-6)(inputs + attention_output)
        
        # Feed forward
        ffn_output = Dense(ff_dim, activation="relu")(out1)
        ffn_output = Dense(inputs.shape[-1])(ffn_output)
        ffn_output = Dropout(dropout)(ffn_output)
        out2 = LayerNormalization(epsilon=1e-6)(out1 + ffn_output)
        
        return out2
    
    def build_model(self, input_shape):
        """Build Transformer model"""
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow required for Transformer model")
        
        inputs = Input(shape=input_shape)
        x = inputs
        
        # Add positional encoding (simplified)
        x = Dense(self.d_model)(x)
        
        # Transformer encoder layers
        for _ in range(self.num_layers):
            x = self.transformer_encoder(x, self.d_model, self.num_heads, self.d_model * 4, self.dropout)
        
        # Global average pooling
        x = GlobalAveragePooling1D()(x)
        x = Dropout(self.dropout)(x)
        outputs = Dense(1)(x)
        
        model = Model(inputs, outputs)
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        return model
    
    def train(self, data, epochs=50, batch_size=32, validation_split=0.2):
        """Train the Transformer model"""
        X, y = self.prepare_data(data)
        
        if len(X) == 0:
            return None
        
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        self.model = self.build_model((X_train.shape[1], X_train.shape[2]))
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7)
        ]
        
        self.model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=0
        )
        
        return self.model
    
    def predict(self, data, days=7):
        """Make predictions"""
        if self.model is None:
            return None, None
        
        X, _ = self.prepare_data(data)
        if len(X) == 0:
            return None, None
        
        last_sequence = X[-1:]
        predictions = []
        current_input = last_sequence.copy()
        
        for _ in range(days):
            pred = self.model.predict(current_input, verbose=0)[0, 0]
            predictions.append(pred)
            new_row = np.append(current_input[0, 1:, :], 
                               [[pred, pred, pred, current_input[0, -1, 3]]], axis=0)
            current_input = np.array([new_row])
        
        dummy_data = np.zeros((len(predictions), 4))
        dummy_data[:, 0] = predictions
        predictions_scaled = self.scaler.inverse_transform(dummy_data)[:, 0]
        
        last_date = data.index[-1]
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days, freq='D')
        
        return future_dates, predictions_scaled


class HybridTransformerGRU:
    """
    Hybrid Transformer-GRU model combining attention mechanism with sequential processing
    State-of-the-art approach for cryptocurrency prediction
    """
    
    def __init__(self, sequence_length=60, d_model=64, num_heads=4, gru_units=50, dropout=0.2):
        self.sequence_length = sequence_length
        self.d_model = d_model
        self.num_heads = num_heads
        self.gru_units = gru_units
        self.dropout = dropout
        self.scaler = MinMaxScaler()
        self.model = None
    
    def prepare_data(self, data, lookback=None):
        """Prepare data for Hybrid model training"""
        if lookback is None:
            lookback = self.sequence_length
            
        features = ['Close', 'High', 'Low', 'Volume']
        df = data[features].copy()
        scaled_data = self.scaler.fit_transform(df)
        
        X, y = [], []
        for i in range(lookback, len(scaled_data)):
            X.append(scaled_data[i-lookback:i])
            y.append(scaled_data[i, 0])
        
        return np.array(X), np.array(y)
    
    def build_model(self, input_shape):
        """Build Hybrid Transformer-GRU model"""
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow required for Hybrid model")
        
        inputs = Input(shape=input_shape)
        
        # Transformer branch
        x_transformer = Dense(self.d_model)(inputs)
        x_transformer = self.transformer_encoder(x_transformer, self.d_model, self.num_heads, 
                                                self.d_model * 4, self.dropout)
        x_transformer = GlobalAveragePooling1D()(x_transformer)
        
        # GRU branch
        x_gru = GRU(self.gru_units, return_sequences=True)(inputs)
        x_gru = Dropout(self.dropout)(x_gru)
        x_gru = GRU(self.gru_units)(x_gru)
        x_gru = Dropout(self.dropout)(x_gru)
        
        # Combine both branches
        combined = Concatenate()([x_transformer, x_gru])
        combined = Dense(64, activation='relu')(combined)
        combined = Dropout(self.dropout)(combined)
        outputs = Dense(1)(combined)
        
        model = Model(inputs, outputs)
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        return model
    
    def transformer_encoder(self, inputs, head_size, num_heads, ff_dim, dropout=0):
        """Transformer encoder block"""
        attention_output = MultiHeadAttention(
            key_dim=head_size, num_heads=num_heads, dropout=dropout
        )(inputs, inputs)
        attention_output = Dropout(dropout)(attention_output)
        out1 = LayerNormalization(epsilon=1e-6)(inputs + attention_output)
        
        ffn_output = Dense(ff_dim, activation="relu")(out1)
        ffn_output = Dense(inputs.shape[-1])(ffn_output)
        ffn_output = Dropout(dropout)(ffn_output)
        out2 = LayerNormalization(epsilon=1e-6)(out1 + ffn_output)
        
        return out2
    
    def train(self, data, epochs=50, batch_size=32, validation_split=0.2):
        """Train the Hybrid model"""
        X, y = self.prepare_data(data)
        
        if len(X) == 0:
            return None
        
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        self.model = self.build_model((X_train.shape[1], X_train.shape[2]))
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7)
        ]
        
        self.model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=0
        )
        
        return self.model
    
    def predict(self, data, days=7):
        """Make predictions"""
        if self.model is None:
            return None, None
        
        X, _ = self.prepare_data(data)
        if len(X) == 0:
            return None, None
        
        last_sequence = X[-1:]
        predictions = []
        current_input = last_sequence.copy()
        
        for _ in range(days):
            pred = self.model.predict(current_input, verbose=0)[0, 0]
            predictions.append(pred)
            new_row = np.append(current_input[0, 1:, :], 
                               [[pred, pred, pred, current_input[0, -1, 3]]], axis=0)
            current_input = np.array([new_row])
        
        dummy_data = np.zeros((len(predictions), 4))
        dummy_data[:, 0] = predictions
        predictions_scaled = self.scaler.inverse_transform(dummy_data)[:, 0]
        
        last_date = data.index[-1]
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days, freq='D')
        
        return future_dates, predictions_scaled


class EnsemblePredictor:
    """
    Ensemble model combining multiple prediction algorithms
    Uses weighted average of predictions from different models
    """
    
    def __init__(self, models=None, weights=None):
        self.models = models or []
        self.weights = weights or [1.0 / len(models)] * len(models) if models else []
        self.scaler = MinMaxScaler()
    
    def add_model(self, model, weight=1.0):
        """Add a model to the ensemble"""
        self.models.append(model)
        # Normalize weights
        total_weight = sum(self.weights) + weight
        self.weights = [w / total_weight for w in self.weights] + [weight / total_weight]
    
    def train(self, data, epochs=50, batch_size=32):
        """Train all models in the ensemble"""
        trained_models = []
        for model in self.models:
            try:
                trained_model = model.train(data, epochs=epochs, batch_size=batch_size)
                if trained_model is not None:
                    trained_models.append(model)
            except Exception as e:
                print(f"Error training model {type(model).__name__}: {str(e)}")
                continue
        
        self.models = trained_models
        # Re-normalize weights
        if self.models:
            self.weights = [w / sum(self.weights[:len(self.models)]) 
                          for w in self.weights[:len(self.models)]]
        
        return len(trained_models) > 0
    
    def predict(self, data, days=7):
        """Make ensemble predictions"""
        if not self.models:
            return None, None
        
        all_predictions = []
        valid_models = []
        valid_weights = []
        
        for model, weight in zip(self.models, self.weights):
            try:
                dates, predictions = model.predict(data, days)
                if predictions is not None and len(predictions) == days:
                    all_predictions.append(predictions)
                    valid_models.append(model)
                    valid_weights.append(weight)
            except Exception as e:
                print(f"Error predicting with {type(model).__name__}: {str(e)}")
                continue
        
        if not all_predictions:
            return None, None
        
        # Normalize weights for valid models
        total_weight = sum(valid_weights)
        valid_weights = [w / total_weight for w in valid_weights]
        
        # Weighted average
        ensemble_predictions = np.zeros(days)
        for pred, weight in zip(all_predictions, valid_weights):
            ensemble_predictions += np.array(pred) * weight
        
        last_date = data.index[-1]
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days, freq='D')
        
        return future_dates, ensemble_predictions


def evaluate_predictions(y_true, y_pred):
    """Evaluate prediction performance"""
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    return {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R2': r2,
        'MAPE': mape
    }

