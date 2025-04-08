"""
Utilities for data normalization and standardization.
"""

import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.base import clone

def normalize_time_series(train_data, val_data, test_data, norm_cols, method='standard'):
    """Normalize time series data using different methods."""
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    elif method == 'robust':
        scaler = RobustScaler()
    
    # Fit scaler only on training data
    train_data_norm = train_data.copy()
    val_data_norm = val_data.copy()
    test_data_norm = test_data.copy()
    
    scalers = {}
    for col in norm_cols:
        scaler = clone(scaler)
        train_data_norm[col] = scaler.fit_transform(train_data[col].values.reshape(-1, 1)).flatten()
        val_data_norm[col] = scaler.transform(val_data[col].values.reshape(-1, 1)).flatten()
        test_data_norm[col] = scaler.transform(test_data[col].values.reshape(-1, 1)).flatten()
        scalers[col] = scaler
    
    return train_data_norm, val_data_norm, test_data_norm, scalers

def normalize_sequences(X_train, X_val, X_test):
    """Normalize 3D sequence data along feature dimension."""
    # Calculate mean and std from training data
    mean = np.mean(X_train, axis=(0, 1))
    std = np.std(X_train, axis=(0, 1))
    
    # Normalize all datasets
    X_train_norm = (X_train - mean) / (std + 1e-8)
    X_val_norm = (X_val - mean) / (std + 1e-8)
    X_test_norm = (X_test - mean) / (std + 1e-8)
    
    return X_train_norm, X_val_norm, X_test_norm, mean, std

def create_missing_data_masks(grid_data):
    """Create masks for missing data in grid."""
    # Create masks (1 for valid data, 0 for missing)
    masks = ~np.isnan(grid_data)
    
    # Calculate valid data percentage
    valid_percentage = np.mean(masks)
    print(f"Data completeness: {valid_percentage:.2%}")
    
    return masks
