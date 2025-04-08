"""
Utilities for spatial data processing.
"""

import numpy as np
import pandas as pd

def create_temperature_grid(df, lat_res=5, lon_res=5, method='mean'):
    """Convert irregular temperature data to a regular grid."""
    # Create lat/lon grid
    lat_bins = np.arange(-90, 91, lat_res)
    lon_bins = np.arange(-180, 181, lon_res)
    
    # Create empty grid
    grid_shape = (len(df.index.unique()), 
                  len(lat_bins)-1, 
                  len(lon_bins)-1)
    
    grid_data = np.full(grid_shape, np.nan)
    grid_count = np.zeros((len(lat_bins)-1, len(lon_bins)-1))
    
    # Group data by grid cell and time
    for t_idx, t in enumerate(df.index.unique()):
        time_df = df.loc[t]
        
        for lat_idx in range(len(lat_bins)-1):
            lat_min, lat_max = lat_bins[lat_idx], lat_bins[lat_idx+1]
            
            for lon_idx in range(len(lon_bins)-1):
                lon_min, lon_max = lon_bins[lon_idx], lon_bins[lon_idx+1]
                
                # Filter points in this grid cell
                mask = ((time_df['latitude'] >= lat_min) & 
                        (time_df['latitude'] < lat_max) & 
                        (time_df['longitude'] >= lon_min) & 
                        (time_df['longitude'] < lon_max))
                
                cell_data = time_df[mask]
                
                if len(cell_data) > 0:
                    if method == 'mean':
                        grid_data[t_idx, lat_idx, lon_idx] = cell_data['temperature'].mean()
                    elif method == 'median':
                        grid_data[t_idx, lat_idx, lon_idx] = cell_data['temperature'].median()
                    elif method == 'weighted_mean' and 'uncertainty' in cell_data.columns:
                        weights = 1 / cell_data['uncertainty'].clip(lower=0.01)
                        grid_data[t_idx, lat_idx, lon_idx] = np.average(
                            cell_data['temperature'], weights=weights)
                    
                    grid_count[lat_idx, lon_idx] += 1
    
    # Create metadata
    grid_metadata = {
        'lat_bins': lat_bins,
        'lon_bins': lon_bins,
        'grid_count': grid_count,
        'timestamps': df.index.unique(),
        'coverage': np.mean(grid_count > 0)
    }
    
    return grid_data, grid_metadata

def add_spatial_features(grid_data, grid_metadata):
    """Add spatial features to gridded data."""
    # Get dimensions
    time_steps, n_lat, n_lon = grid_data.shape
    
    # Create output array with additional features
    n_features = 5  # Original data + 4 new features
    enhanced_data = np.zeros((time_steps, n_lat, n_lon, n_features))
    
    # Add original data
    enhanced_data[:, :, :, 0] = grid_data
    
    # Add latitude feature (normalized -1 to 1)
    lat_centers = (grid_metadata['lat_bins'][:-1] + grid_metadata['lat_bins'][1:]) / 2
    lat_normalized = lat_centers / 90.0  # Normalize to [-1, 1]
    
    for i, lat in enumerate(lat_normalized):
        enhanced_data[:, i, :, 1] = lat
    
    # Add longitude feature (normalized -1 to 1)
    lon_centers = (grid_metadata['lon_bins'][:-1] + grid_metadata['lon_bins'][1:]) / 2
    lon_normalized = lon_centers / 180.0  # Normalize to [-1, 1]
    
    for i, lon in enumerate(lon_normalized):
        enhanced_data[:, :, i, 2] = lon
    
    # Add land/sea mask (if available)
    # This would require additional data
    # Placeholder for now
    enhanced_data[:, :, :, 3] = 1.0  # Assume all land for now
    
    # Add elevation data (if available)
    # This would require additional data
    # Placeholder for now
    enhanced_data[:, :, :, 4] = 0.0
    
    return enhanced_data

def create_spatiotemporal_sequences(grid_data, seq_length, horizon=1, stride=1):
    """Create sequences for spatiotemporal modeling."""
    n_samples = grid_data.shape[0] - seq_length - horizon + 1
    n_lat = grid_data.shape[1]
    n_lon = grid_data.shape[2]
    n_features = grid_data.shape[3] if grid_data.ndim > 3 else 1
    
    # Initialize arrays
    X = np.zeros((n_samples, seq_length, n_lat, n_lon, n_features))
    y = np.zeros((n_samples, n_lat, n_lon))
    
    # Create sequences
    for i in range(0, n_samples, stride):
        # Input sequence
        if grid_data.ndim > 3:
            X[i] = grid_data[i:i+seq_length]
        else:
            X[i] = grid_data[i:i+seq_length].reshape(seq_length, n_lat, n_lon, 1)
        
        # Target is the temperature grid at seq_length + horizon
        y[i] = grid_data[i+seq_length+horizon-1] if grid_data.ndim == 3 else grid_data[i+seq_length+horizon-1, :, :, 0]
    
    return X, y
