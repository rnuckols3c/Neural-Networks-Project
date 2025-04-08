"""
Evaluation metrics for neural network models.
"""

import numpy as np
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score, 
    explained_variance_score, accuracy_score, precision_score,
    recall_score, f1_score, roc_curve, precision_recall_curve, auc
)
from scipy.stats import norm

def regression_metrics(y_true, y_pred):
    """Calculate standard regression metrics."""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # Explained variance ratio
    explained_variance = explained_variance_score(y_true, y_pred)
    
    # Mean absolute percentage error
    mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-8))) * 100
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R²': r2,
        'Explained Variance': explained_variance,
        'MAPE': mape
    }

def climate_specific_metrics(y_true, y_pred, uncertainty=None):
    """Calculate climate-specific evaluation metrics."""
    # Nash-Sutcliffe efficiency (common in hydrology/climate modeling)
    # 1 = perfect match, 0 = prediction no better than mean, negative = worse than mean
    nse = 1 - (np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2))
    
    # Willmott's index of agreement (0-1, higher is better)
    d = 1 - (np.sum((y_pred - y_true)**2) / 
             np.sum((np.abs(y_pred - np.mean(y_true)) + np.abs(y_true - np.mean(y_true)))**2))
    
    # Anomaly Correlation Coefficient
    # Remove seasonal mean to focus on anomalies
    y_true_anom = y_true - np.mean(y_true)
    y_pred_anom = y_pred - np.mean(y_pred)
    acc = np.sum(y_true_anom * y_pred_anom) / (np.sqrt(np.sum(y_true_anom**2) * np.sum(y_pred_anom**2)))
    
    # Calculate metrics weighted by uncertainty if available
    weighted_metrics = {}
    if uncertainty is not None:
        # Convert uncertainty to weights (higher certainty = higher weight)
        weights = 1 / (uncertainty + 1e-8)
        weights = weights / np.sum(weights)
        
        # Weighted MSE
        weighted_mse = np.sum(weights * (y_true - y_pred)**2)
        weighted_metrics['Weighted MSE'] = weighted_mse
    
    return {
        'NSE': nse,
        'Willmott Index': d,
        'ACC': acc,
        **weighted_metrics
    }

def extreme_event_metrics(y_true_binary, y_pred_prob, threshold=0.5):
    """Evaluate model performance for extreme event detection."""
    # Convert probabilities to binary predictions
    y_pred_binary = (y_pred_prob >= threshold).astype(int)
    
    # Basic classification metrics
    accuracy = accuracy_score(y_true_binary, y_pred_binary)
    precision = precision_score(y_true_binary, y_pred_binary)
    recall = recall_score(y_true_binary, y_pred_binary)
    f1 = f1_score(y_true_binary, y_pred_binary)
    
    # ROC and AUC
    fpr, tpr, _ = roc_curve(y_true_binary, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    
    # Precision-Recall curve
    precision_curve, recall_curve, _ = precision_recall_curve(y_true_binary, y_pred_prob)
    pr_auc = auc(recall_curve, precision_curve)
    
    return {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'ROC AUC': roc_auc,
        'PR AUC': pr_auc
    }

def uncertainty_evaluation(y_true, y_pred_mean, y_pred_std):
    """Evaluate probabilistic predictions with uncertainty estimates."""
    # Negative log likelihood (assuming Gaussian distribution)
    nll = -np.mean(norm.logpdf(y_true, loc=y_pred_mean, scale=y_pred_std))
    
    # Prediction intervals
    lower_bound = y_pred_mean - 1.96 * y_pred_std
    upper_bound = y_pred_mean + 1.96 * y_pred_std
    
    # Coverage probability (should be close to 0.95 for well-calibrated models)
    in_interval = np.logical_and(y_true >= lower_bound, y_true <= upper_bound)
    coverage = np.mean(in_interval)
    
    # Mean prediction interval width
    mean_interval_width = np.mean(upper_bound - lower_bound)
    
    return {
        'NLL': nll,
        'Coverage Probability': coverage,
        'Mean Interval Width': mean_interval_width
    }
