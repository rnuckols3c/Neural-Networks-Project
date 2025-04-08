"""
Utilities for data cleaning and handling missing values.
"""

import numpy as np
import pandas as pd

def handle_missing_and_uncertainty(df):
    """Handle missing values and incorporate uncertainty information."""
    # Fill missing values using interpolation
    df = df.copy()
    if 'temperature' in df.columns:
        df['temperature'] = df['temperature'].interpolate(method='time')
        
        # Flag imputed values
        df['is_imputed'] = df['temperature'].isna().astype(int)
    
    # Weight observations by inverse of uncertainty
    if 'uncertainty' in df.columns:
        # Cap uncertainty to avoid division by zero or extreme weights
        min_uncertainty = 0.01
        df['uncertainty'] = df['uncertainty'].clip(lower=min_uncertainty)
        
        # Create confidence weights (higher for more certain measurements)
        df['confidence_weight'] = 1 / df['uncertainty']
        df['confidence_weight'] = df['confidence_weight'] / df['confidence_weight'].mean()
    
    # Add flags for pre-1850 data (higher uncertainty period)
    df['pre_1850_flag'] = (df.index.year < 1850).astype(int)
    
    # Add flags for post-1979 data (satellite era, higher certainty)
    df['satellite_era'] = (df.index.year >= 1979).astype(int)
    
    return df

def fill_spatial_gaps(grid_data, method='interpolate'):
    """Fill missing values in spatial grid data."""
    filled_grid = grid_data.copy()
    
    if method == 'interpolate':
        # For each time step
        for t in range(grid_data.shape[0]):
            # Get current time slice
            time_slice = grid_data[t]
            
            # Create a mask of missing values
            mask = np.isnan(time_slice)
            
            # Only process if there are missing values but not all missing
            if np.any(mask) and not np.all(mask):
                # Get grid coordinates
                y, x = np.indices(time_slice.shape)
                
                # Get coordinates of non-missing values
                y_good = y[~mask]
                x_good = x[~mask]
                
                # Get coordinates of missing values
                y_bad = y[mask]
                x_bad = x[mask]
                
                # Get values at non-missing coordinates
                z_good = time_slice[~mask]
                
                # Interpolate
                from scipy.interpolate import griddata
                z_bad = griddata((y_good, x_good), z_good, (y_bad, x_bad), method='linear')
                
                # Put interpolated values back
                time_slice_filled = time_slice.copy()
                time_slice_filled[mask] = z_bad
                
                filled_grid[t] = time_slice_filled
    
    elif method == 'nearest_neighbor':
        from sklearn.impute import KNNImputer
        
        # Reshape to 2D for imputer
        original_shape = grid_data.shape
        reshaped_data = grid_data.reshape(original_shape[0], -1)
        
        # Impute
        imputer = KNNImputer(n_neighbors=5)
        filled_data = imputer.fit_transform(reshaped_data)
        
        # Reshape back
        filled_grid = filled_data.reshape(original_shape)
    
    return filled_grid
