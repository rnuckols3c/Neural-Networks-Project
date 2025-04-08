"""
Utilities for splitting data into training, validation, and test sets.
"""

import numpy as np
import pandas as pd

def temporal_split(data, train_pct=0.7, val_pct=0.15):
    """Split time series data maintaining temporal order."""
    n = len(data)
    train_size = int(n * train_pct)
    val_size = int(n * val_pct)
    
    train_data = data.iloc[:train_size]
    val_data = data.iloc[train_size:train_size + val_size]
    test_data = data.iloc[train_size + val_size:]
    
    return train_data, val_data, test_data

def blocked_rolling_window_split(data, training_years=30, validation_years=5, test_years=5):
    """Create multiple train-val-test splits using a rolling window approach."""
    # Assuming data has a datetime index
    min_year = data.index.year.min()
    max_year = data.index.year.max()
    
    splits = []
    for start_year in range(min_year, max_year - (training_years + validation_years + test_years) + 1, 5):
        train_start = start_year
        train_end = start_year + training_years
        val_start = train_end
        val_end = val_start + validation_years
        test_start = val_end
        test_end = test_start + test_years
        
        train_mask = (data.index.year >= train_start) & (data.index.year < train_end)
        val_mask = (data.index.year >= val_start) & (data.index.year < val_end)
        test_mask = (data.index.year >= test_start) & (data.index.year < test_end)
        
        train_data = data[train_mask]
        val_data = data[val_mask]
        test_data = data[test_mask]
        
        splits.append((train_data, val_data, test_data))
    
    return splits

def data_augmentation(X, y, noise_level=0.05, n_samples=1):
    """Augment time series data with noise and time warping."""
    from scipy.interpolate import interp1d
    
    X_aug, y_aug = [], []
    
    # Add original data
    X_aug.append(X)
    y_aug.append(y)
    
    for i in range(n_samples):
        # Add noise
        noise = np.random.normal(0, noise_level, X.shape)
        X_noisy = X + noise
        X_aug.append(X_noisy)
        y_aug.append(y)
        
        # Time warping (stretch or compress)
        if X.shape[1] > 10:  # Only if sequence is long enough
            time_warp_idx = np.linspace(0, X.shape[1]-1, X.shape[1])
            warp_factor = np.random.uniform(0.8, 1.2)
            warped_idx = np.linspace(0, X.shape[1]-1, int(X.shape[1]*warp_factor))
            
            # Interpolate to original length
            X_warped = np.zeros_like(X)
            for j in range(X.shape[2]):
                for k in range(X.shape[0]):
                    f = interp1d(warped_idx, X[k, :, j], 
                                 bounds_error=False, fill_value="extrapolate")
                    X_warped[k, :, j] = f(time_warp_idx)
            
            X_aug.append(X_warped)
            y_aug.append(y)
    
    return np.vstack(X_aug), np.vstack(y_aug)
