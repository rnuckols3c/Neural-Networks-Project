"""
Utilities for time series feature engineering and processing.
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL

def create_time_features(df):
    """Create time-based features from datetime index."""
    df = df.copy()
    df['month'] = df.index.month
    df['quarter'] = df.index.quarter
    df['year'] = df.index.year
    df['dayofyear'] = df.index.dayofyear
    df['sin_month'] = np.sin(2 * np.pi * df.index.month / 12)
    df['cos_month'] = np.cos(2 * np.pi * df.index.month / 12)
    
    # Add trend feature
    df['time_idx'] = np.arange(len(df))
    
    # Add lagged features
    for lag in [1, 3, 6, 12, 24]:  # Various lag values in months
        df[f'temp_lag_{lag}'] = df['temperature'].shift(lag)
    
    # Add rolling statistics
    for window in [3, 6, 12, 36, 60]:  # Various window sizes
        df[f'temp_roll_mean_{window}'] = df['temperature'].rolling(window=window).mean()
        df[f'temp_roll_std_{window}'] = df['temperature'].rolling(window=window).std()
    
    return df

def decompose_time_series(df, column='temperature', period=12):
    """Decompose time series into trend, seasonal, and residual components."""
    try:
        df = df.copy()
        stl = STL(df[column], period=period)
        result = stl.fit()
        df['trend_component'] = result.trend
        df['seasonal_component'] = result.seasonal
        df['residual_component'] = result.resid
    except Exception as e:
        print(f"Warning: STL decomposition failed - {str(e)}")
        
    return df

def create_sequences(data, target_col, seq_length, horizon=1, stride=1):
    """Create sliding window sequences for time series modeling."""
    xs, ys = [], []
    
    for i in range(0, len(data) - seq_length - horizon + 1, stride):
        x = data.iloc[i:(i + seq_length)]
        y = data.iloc[i + seq_length:i + seq_length + horizon][target_col].values
        xs.append(x.values)
        ys.append(y)
        
    return np.array(xs), np.array(ys)

def create_multitask_sequences(data, target_cols, seq_length, horizon=1):
    """Create sequences for predicting multiple targets."""
    xs, ys = [], []
    
    for i in range(len(data) - seq_length - horizon + 1):
        x = data.iloc[i:(i + seq_length)]
        y = data.iloc[i + seq_length:i + seq_length + horizon][target_cols].values
        xs.append(x.values)
        ys.append(y)
        
    return np.array(xs), np.array(ys)
