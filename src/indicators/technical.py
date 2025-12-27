"""
Technical indicators for cryptocurrency analysis
RSI, SMA, EMA, MACD, Bollinger Bands
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional


class TechnicalIndicators:
    """Calculate various technical indicators"""
    
    @staticmethod
    def calculate_rsi(data: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index (RSI)
        
        Args:
            data: DataFrame with 'Close' column
            period: RSI period (default 14)
        
        Returns:
            Series with RSI values
        """
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def calculate_sma(data: pd.DataFrame, period: int) -> pd.Series:
        """
        Calculate Simple Moving Average (SMA)
        
        Args:
            data: DataFrame with 'Close' column
            period: SMA period
        
        Returns:
            Series with SMA values
        """
        return data['Close'].rolling(window=period).mean()
    
    @staticmethod
    def calculate_ema(data: pd.DataFrame, period: int) -> pd.Series:
        """
        Calculate Exponential Moving Average (EMA)
        
        Args:
            data: DataFrame with 'Close' column
            period: EMA period
        
        Returns:
            Series with EMA values
        """
        return data['Close'].ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def calculate_macd(data: pd.DataFrame, fast: int = 12, slow: int = 26, 
                      signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate MACD (Moving Average Convergence Divergence)
        
        Args:
            data: DataFrame with 'Close' column
            fast: Fast EMA period
            slow: Slow EMA period
            signal: Signal line period
        
        Returns:
            Tuple of (MACD line, Signal line, Histogram)
        """
        ema_fast = TechnicalIndicators.calculate_ema(data, fast)
        ema_slow = TechnicalIndicators.calculate_ema(data, slow)
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        histogram = macd - signal_line
        return macd, signal_line, histogram
    
    @staticmethod
    def calculate_bollinger_bands(data: pd.DataFrame, period: int = 20, 
                                 std_dev: int = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate Bollinger Bands
        
        Args:
            data: DataFrame with 'Close' column
            period: Moving average period
            std_dev: Standard deviation multiplier
        
        Returns:
            Tuple of (Upper band, Middle band (SMA), Lower band)
        """
        sma = TechnicalIndicators.calculate_sma(data, period)
        std = data['Close'].rolling(window=period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        return upper_band, sma, lower_band
    
    @staticmethod
    def analyze_rsi(rsi_value: float) -> dict:
        """Analyze RSI value and return signal"""
        if pd.isna(rsi_value):
            return {'signal': 'unknown', 'color': 'gray', 'message': 'Insufficient data'}
        
        if rsi_value > 70:
            return {
                'signal': 'overbought',
                'color': 'red',
                'message': 'Overbought - Price at high level, exercise caution'
            }
        elif rsi_value < 30:
            return {
                'signal': 'oversold',
                'color': 'green',
                'message': 'Oversold - Potential opportunity'
            }
        else:
            return {
                'signal': 'normal',
                'color': 'blue',
                'message': 'Normal level'
            }
    
    @staticmethod
    def analyze_sma_crossover(sma_short: float, sma_long: float) -> dict:
        """Analyze SMA crossover"""
        if pd.isna(sma_short) or pd.isna(sma_long):
            return {'signal': 'unknown', 'color': 'gray', 'message': 'Insufficient data'}
        
        if sma_short > sma_long:
            return {
                'signal': 'uptrend',
                'color': 'green',
                'message': 'Uptrend - Short MA above Long MA'
            }
        else:
            return {
                'signal': 'downtrend',
                'color': 'red',
                'message': 'Downtrend - Short MA below Long MA'
            }
    
    @staticmethod
    def analyze_macd(macd_value: float, signal_value: float) -> dict:
        """Analyze MACD crossover"""
        if pd.isna(macd_value) or pd.isna(signal_value):
            return {'signal': 'unknown', 'color': 'gray', 'message': 'Insufficient data'}
        
        if macd_value > signal_value:
            return {
                'signal': 'bullish',
                'color': 'green',
                'message': 'Bullish signal - MACD above Signal'
            }
        else:
            return {
                'signal': 'bearish',
                'color': 'red',
                'message': 'Bearish signal - MACD below Signal'
            }

