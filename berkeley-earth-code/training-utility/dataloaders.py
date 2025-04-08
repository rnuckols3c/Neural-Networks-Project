"""
Data loading utilities for neural network models.
"""

import torch
from torch.utils.data import Dataset, DataLoader

class TimeSeriesDataset(Dataset):
    """PyTorch dataset for time series data."""
    def __init__(self, X, y, weights=None):
        """
        Initialize the dataset.
        
        Parameters:
        -----------
        X : numpy.ndarray
            Input sequences.
        y : numpy.ndarray
            Target values.
        weights : numpy.ndarray, optional (default=None)
            Sample weights.
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.weights = torch.tensor(weights, dtype=torch.float32) if weights is not None else None
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        if self.weights is not None:
            return self.X[idx], self.y[idx], self.weights[idx]
        else:
            return self.X[idx], self.y[idx]

class SpatioTemporalDataset(Dataset):
    """PyTorch dataset for spatiotemporal data."""
    def __init__(self, X, y, masks=None):
        """
        Initialize the dataset.
        
        Parameters:
        -----------
        X : numpy.ndarray
            Input sequences with shape (samples, time_steps, height, width, channels).
        y : numpy.ndarray
            Target values with shape (samples, height, width).
        masks : numpy.ndarray, optional (default=None)
            Masks indicating valid data points.
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.masks = torch.tensor(masks, dtype=torch.float32) if masks is not None else None
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        if self.masks is not None:
            return self.X[idx], self.y[idx], self.masks[idx]
        else:
            return self.X[idx], self.y[idx]

def create_data_loaders(X_train, y_train, X_val, y_val, X_test, y_test, 
                       batch_size=32, weights_train=None, weights_val=None, weights_test=None):
    """
    Create PyTorch DataLoaders for training, validation, and testing.
    
    Parameters:
    -----------
    X_train : numpy.ndarray
        Training input data.
    y_train : numpy.ndarray
        Training target data.
    X_val : numpy.ndarray
        Validation input data.
    y_val : numpy.ndarray
        Validation target data.
    X_test : numpy.ndarray
        Test input data.
    y_test : numpy.ndarray
        Test target data.
    batch_size : int, optional (default=32)
        Batch size for DataLoaders.
    weights_train : numpy.ndarray, optional (default=None)
        Sample weights for training data.
    weights_val : numpy.ndarray, optional (default=None)
        Sample weights for validation data.
    weights_test : numpy.ndarray, optional (default=None)
        Sample weights for test data.
        
    Returns:
    --------
    tuple
        Tuple of (train_loader, val_loader, test_loader).
    """
    train_dataset = TimeSeriesDataset(X_train, y_train, weights_train)
    val_dataset = TimeSeriesDataset(X_val, y_val, weights_val)
    test_dataset = TimeSeriesDataset(X_test, y_test, weights_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    return train_loader, val_loader, test_loader
