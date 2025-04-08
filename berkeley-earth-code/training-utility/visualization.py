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
    return plt.gcf()

def plot_prediction_error(y_true, y_pred, dates=None, title='Prediction Error'):
    """Plot prediction errors over time."""
    error = y_true - y_pred
    
    plt.figure(figsize=(12, 6))
    
    if dates is not None:
        plt.plot(dates, error, color='red', alpha=0.7)
        plt.xlabel('Date')
    else:
        plt.plot(error, color='red', alpha=0.7)
        plt.xlabel('Sample Index')
    
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.title(title)
    plt.ylabel('Error (°C)')
    plt.grid(True)
    return plt.gcf()

def plot_attention_weights(attention_weights, input_sequence, dates=None, n_display=10):
    """Visualize attention weights from attention-based models."""
    # Get the most recent prediction if multiple predictions are available
    if len(attention_weights.shape) > 2:
        attention_weights = attention_weights[-1]
    
    # Select a subset of time steps to display if too many
    seq_len = attention_weights.shape[1]
    if seq_len > n_display:
        # Take evenly spaced time steps
        idx = np.linspace(0, seq_len - 1, n_display, dtype=int)
        attention_weights = attention_weights[:, idx]
        if dates is not None:
            dates = [dates[i] for i in idx]
    
    plt.figure(figsize=(12, 6))
    sns.heatmap(attention_weights.T, cmap='viridis')
    
    if dates is not None:
        plt.yticks(np.arange(len(dates)) + 0.5, dates, rotation=0)
    
    plt.xlabel('Samples')
    plt.ylabel('Time Steps')
    plt.title('Attention Weights')
    return plt.gcf()

def plot_spatial_predictions(true_grid, pred_grid, lat_bins, lon_bins, time_idx=0, 
                           title='Spatial Temperature Predictions'):
    """Plot spatial temperature predictions for a specific time step."""
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot actual temperatures
    im1 = axs[0].imshow(true_grid[time_idx], cmap='coolwarm', 
                      extent=[lon_bins[0], lon_bins[-1], lat_bins[0], lat_bins[-1]])
    axs[0].set_title('Actual Temperatures')
    axs[0].set_xlabel('Longitude')
    axs[0].set_ylabel('Latitude')
    plt.colorbar(im1, ax=axs[0], label='Temperature (°C)')
    
    # Plot predicted temperatures
    im2 = axs[1].imshow(pred_grid[time_idx], cmap='coolwarm', 
                      extent=[lon_bins[0], lon_bins[-1], lat_bins[0], lat_bins[-1]])
    axs[1].set_title('Predicted Temperatures')
    axs[1].set_xlabel('Longitude')
    plt.colorbar(im2, ax=axs[1], label='Temperature (°C)')
    
    # Plot error
    error = true_grid[time_idx] - pred_grid[time_idx]
    im3 = axs[2].imshow(error, cmap='RdBu_r', 
                      extent=[lon_bins[0], lon_bins[-1], lat_bins[0], lat_bins[-1]])
    axs[2].set_title('Prediction Error')
    axs[2].set_xlabel('Longitude')
    plt.colorbar(im3, ax=axs[2], label='Error (°C)')
    
    plt.suptitle(title)
    plt.tight_layout()
    return fig

def plot_feature_importance(model, feature_names, title='Feature Importance'):
    """Plot feature importance for interpretable models."""
    if hasattr(model, 'feature_importances_'):
        # For tree-based models
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        # For linear models
        importances = np.abs(model.coef_[0])
    else:
        raise ValueError("Model does not have accessible feature importances")
    
    # Sort features by importance
    sorted_idx = np.argsort(importances)
    sorted_names = [feature_names[i] for i in sorted_idx]
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(sorted_idx)), importances[sorted_idx])
    plt.yticks(range(len(sorted_idx)), sorted_names)
    plt.xlabel('Importance')
    plt.title(title)
    plt.tight_layout()
    return plt.gcf()

def plot_regime_probabilities(regime_probs, dates, regime_names=None, title='Climate Regime Probabilities'):
    """Plot the probability of different climate regimes over time."""
    n_regimes = regime_probs.shape[1]
    
    if regime_names is None:
        regime_names = [f'Regime {i+1}' for i in range(n_regimes)]
    
    plt.figure(figsize=(12, 6))
    
    for i in range(n_regimes):
        plt.plot(dates, regime_probs[:, i], label=regime_names[i])
    
    plt.xlabel('Date')
    plt.ylabel('Probability')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    return plt.gcf()
