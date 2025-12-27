"""
Helper utility functions
"""

import warnings
from typing import Optional, Tuple
import pandas as pd
import numpy as np

warnings.filterwarnings('ignore')

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers, returning default if denominator is zero"""
    if denominator == 0:
        return default
    return numerator / denominator

def calculate_percentage_change(old_value: float, new_value: float) -> Tuple[float, float]:
    """Calculate percentage change between two values"""
    if old_value == 0:
        return 0.0, 0.0
    change = new_value - old_value
    change_pct = (change / old_value) * 100
    return change, change_pct

def format_currency(value: float, currency: str = "$") -> str:
    """Format a number as currency"""
    return f"{currency}{value:,.2f}"

def format_percentage(value: float, decimals: int = 2) -> str:
    """Format a number as percentage"""
    return f"{value:.{decimals}f}%"

def validate_dataframe(df: pd.DataFrame, required_columns: list) -> Tuple[bool, Optional[str]]:
    """Validate that dataframe has required columns"""
    if df is None or df.empty:
        return False, "DataFrame is empty or None"
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        return False, f"Missing required columns: {missing_columns}"
    
    return True, None

def get_latest_value(series: pd.Series, default: float = 0.0) -> float:
    """Get the latest non-null value from a series"""
    if series.empty:
        return default
    valid_values = series.dropna()
    if valid_values.empty:
        return default
    return float(valid_values.iloc[-1])

def check_dependencies() -> dict:
    """Check which optional dependencies are available"""
    dependencies = {
        'torch': False,
        'tensorflow': False,
        'transformers': False,
        'textblob': False
    }
    
    try:
        import torch
        dependencies['torch'] = True
    except ImportError:
        pass
    
    try:
        import tensorflow
        dependencies['tensorflow'] = True
    except ImportError:
        pass
    
    try:
        import transformers
        dependencies['transformers'] = True
    except ImportError:
        pass
    
    try:
        import textblob
        dependencies['textblob'] = True
    except ImportError:
        pass
    
    return dependencies

