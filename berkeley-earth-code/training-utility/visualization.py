"""
Visualization utilities for model analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_training_history(history, title='Training History'):
    """Plot training and validation loss curves."""
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    return plt.gcf()

def plot_predictions_vs_actual(y_true, y_pred, dates=None, title='Predictions vs Actual'):
    """Plot model predictions against actual values."""
    plt.figure(figsize=(12, 6))
    
    if dates is not None:
        plt.plot(dates, y_true, label='Actual', alpha=0.7)
        plt.plot(dates, y_pred, label='Predicted', alpha=0.7)
        plt.xlabel('Date')
    else:
        plt.plot(y_true, label='Actual', alpha=0.7)
        plt.plot(y_pred, label='Predicted', alpha=0.7)
        plt.xlabel('Sample Index')
    
    plt.title(title)
    plt.ylabel('Temperature (°C)')
    plt.legend()
    plt.grid(True)
