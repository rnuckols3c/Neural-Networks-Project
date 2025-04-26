"""
Berkeley Earth Temperature Dataset Neural Network Analysis
==========================================================
Main implementation script to run the neural network models for
temperature pattern analysis and forecasting.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Import data preprocessing utilities
import sys
sys.path.append('./data-preprocessing')
from time_series_processing import create_time_features, create_sequences, decompose_time_series
from data_cleaning import handle_missing_and_uncertainty
from data_normalization import normalize_sequences
from data_splitting import temporal_split

# Import model implementations
sys.path.append('./model-implementation')
from tcn_model import TCNForecaster
from lstm_model import BidirectionalAttentionLSTM, RegimeIdentificationLSTM

# Import training utilities
sys.path.append('./training-utility')
from training import train_with_early_stopping
from evaluation import regression_metrics, climate_specific_metrics
from visualization import (plot_training_history, plot_predictions_vs_actual,
                          plot_prediction_error, plot_attention_weights)
from dataloaders import TimeSeriesDataset

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Set paths
DATA_DIR = r"C:\Users\Richard Nuckols\Desktop\Desktop\Personal\JHU\Neural Networks\Module7\Data"
GLOBAL_TEMP_FILE = os.path.join(DATA_DIR, "GlobalTemperatures.csv")
COUNTRIES_TEMP_FILE = os.path.join(DATA_DIR, "GlobalLandTemperaturesByCountry.csv")
OUTPUT_DIR = os.path.join(DATA_DIR, "Model_Results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Configuration parameters
SEQUENCE_LENGTH = 120  # 10 years of monthly data
HORIZON = 12  # Predict 1 year ahead
TRAIN_PCT = 0.7
VAL_PCT = 0.15
BATCH_SIZE = 32
NUM_EPOCHS = 100
LEARNING_RATE = 0.001
PATIENCE = 10
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

def load_and_preprocess_data(data_file, target_col='LandAverageTemperature'):
    """
    Load and preprocess temperature data.
    
    Parameters:
    -----------
    data_file : str
        Path to data file
    target_col : str
        Name of the target column
        
    Returns:
    --------
    tuple
        Preprocessed data splits and metadata
    """
    print(f"Loading data from {data_file}...")
    
    # Load data
    data = pd.read_csv(data_file, parse_dates=["dt"])
    data.set_index("dt", inplace=True)
    print(f"Data loaded, shape: {data.shape}")
    
    # Rename columns for consistency
    if target_col in data.columns:
        data['temperature'] = data[target_col]
        if f'{target_col}Uncertainty' in data.columns:
            data['uncertainty'] = data[f'{target_col}Uncertainty']
    
    # Handle missing values and uncertainty
    data = handle_missing_and_uncertainty(data)
    print("Missing values handled")
    
    # Create time features
    data = create_time_features(data)
    print("Time features created")
    
    # Decompose time series
    if 'temperature' in data.columns:
        data = decompose_time_series(data, column='temperature')
        print("Time series decomposed")
    
    # Split data
    train_data, val_data, test_data = temporal_split(data, train_pct=TRAIN_PCT, val_pct=VAL_PCT)
    print(f"Data split: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
    
    return train_data, val_data, test_data, data.columns.tolist()

def prepare_sequence_data(train_data, val_data, test_data, target_col='temperature', feature_cols=None):
    """
    Prepare sequence data for time series models.
    
    Parameters:
    -----------
    train_data, val_data, test_data : pd.DataFrame
        Data splits
    target_col : str
        Name of the target column
    feature_cols : list
        List of feature columns to use
        
    Returns:
    --------
    tuple
        Prepared sequence data and feature info
    """
    # Select features (exclude target from input features)
    if feature_cols is None:
        numeric_cols = train_data.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [col for col in numeric_cols if col != target_col and not col.startswith('is_')]
    
    print(f"Selected {len(feature_cols)} features")
    
    # Create sequences
    X_train, y_train = create_sequences(
        train_data[feature_cols + [target_col]], target_col, 
        seq_length=SEQUENCE_LENGTH, horizon=HORIZON, stride=1
    )
    X_val, y_val = create_sequences(
        val_data[feature_cols + [target_col]], target_col,
        seq_length=SEQUENCE_LENGTH, horizon=HORIZON, stride=1
    )
    X_test, y_test = create_sequences(
        test_data[feature_cols + [target_col]], target_col,
        seq_length=SEQUENCE_LENGTH, horizon=HORIZON, stride=1
    )
    
    print(f"Sequences created: X_train={X_train.shape}, y_train={y_train.shape}")
    
    # Normalize sequences
    X_train_norm, X_val_norm, X_test_norm, norm_mean, norm_std = normalize_sequences(
        X_train, X_val, X_test
    )
    print("Sequences normalized")
    
    # Get uncertainty if available (for weighted loss)
    weights_train = weights_val = weights_test = None
    if 'uncertainty' in train_data.columns:
        _, weights_train = create_sequences(
            train_data[['temperature', 'uncertainty']], 'uncertainty',
            seq_length=SEQUENCE_LENGTH, horizon=HORIZON, stride=1
        )
        _, weights_val = create_sequences(
            val_data[['temperature', 'uncertainty']], 'uncertainty',
            seq_length=SEQUENCE_LENGTH, horizon=HORIZON, stride=1
        )
        _, weights_test = create_sequences(
            test_data[['temperature', 'uncertainty']], 'uncertainty',
            seq_length=SEQUENCE_LENGTH, horizon=HORIZON, stride=1
        )
        
        # Convert uncertainties to weights (higher certainty = higher weight)
        weights_train = 1.0 / (weights_train + 1e-5)
        weights_val = 1.0 / (weights_val + 1e-5)
        weights_test = 1.0 / (weights_test + 1e-5)
        
        # Normalize weights
        weights_train = weights_train / np.mean(weights_train)
        weights_val = weights_val / np.mean(weights_val)
        weights_test = weights_test / np.mean(weights_test)
    
    return (X_train_norm, y_train, X_val_norm, y_val, X_test_norm, y_test,
            weights_train, weights_val, weights_test, feature_cols)

def create_dataloaders(X_train, y_train, X_val, y_val, X_test, y_test, weights_train=None,
                     weights_val=None, weights_test=None):
    """Create PyTorch DataLoaders."""
    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
    
    # Create datasets
    if weights_train is not None:
        weights_train_tensor = torch.tensor(weights_train, dtype=torch.float32)
        weights_val_tensor = torch.tensor(weights_val, dtype=torch.float32)
        weights_test_tensor = torch.tensor(weights_test, dtype=torch.float32)
        
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor, weights_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor, weights_val_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor, weights_test_tensor)
    else:
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    return train_loader, val_loader, test_loader

def weighted_mse_loss(output, target, weights=None):
    """MSE loss with optional weighting."""
    if weights is None:
        return torch.mean((output - target) ** 2)
    else:
        return torch.mean(weights * (output - target) ** 2)

def train_tcn_model(train_loader, val_loader, input_size, output_size=HORIZON):
    """Train a TCN model for temperature forecasting."""
    print("\n===== Training TCN Model =====")
    
    # Define model architecture
    num_channels = [64, 128, 256, 128, 64]  # Increasing then decreasing channels
    model = TCNForecaster(
        input_size=input_size,
        output_size=output_size,
        num_channels=num_channels,
        kernel_size=3,
        dropout=0.2
    ).to(DEVICE)
    
    print(f"TCN Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Define loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Train model
    trained_model, history = train_with_early_stopping(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=DEVICE,
        num_epochs=NUM_EPOCHS,
        patience=PATIENCE
    )
    
    # Save model
    torch.save(trained_model.state_dict(), os.path.join(OUTPUT_DIR, 'tcn_model.pth'))
    
    # Plot training history
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('TCN Model Training History')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, 'tcn_training_history.png'))
    
    return trained_model

def train_lstm_model(train_loader, val_loader, input_size, output_size=HORIZON):
    """Train a Bidirectional LSTM model with attention."""
    print("\n===== Training LSTM Model =====")
    
    # Define model architecture
    hidden_size = 128
    num_layers = 2
    model = BidirectionalAttentionLSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        output_size=output_size,
        dropout=0.3
    ).to(DEVICE)
    
    print(f"LSTM Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Define loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Learning rate scheduler with warmup
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Custom training loop for attention model
    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(NUM_EPOCHS):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for batch in train_loader:
            if len(batch) == 3:  # With weights
                inputs, targets, weights = batch
                inputs, targets, weights = inputs.to(DEVICE), targets.to(DEVICE), weights.to(DEVICE)
            else:  # Without weights
                inputs, targets = batch
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                weights = None
            
            optimizer.zero_grad()
            outputs, _ = model(inputs)
            
            # Apply weights if available
            if weights is not None:
                loss = weighted_mse_loss(outputs, targets, weights)
            else:
                loss = criterion(outputs, targets)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        attention_weights_list = []
        
        with torch.no_grad():
            for batch in val_loader:
                if len(batch) == 3:  # With weights
                    inputs, targets, weights = batch
                    inputs, targets, weights = inputs.to(DEVICE), targets.to(DEVICE), weights.to(DEVICE)
                else:  # Without weights
                    inputs, targets = batch
                    inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                    weights = None
                
                outputs, attention_weights = model(inputs)
                
                if weights is not None:
                    loss = weighted_mse_loss(outputs, targets, weights)
                else:
                    loss = criterion(outputs, targets)
                
                val_loss += loss.item()
                attention_weights_list.append(attention_weights.cpu().numpy())
        
        avg_val_loss = val_loss / len(val_loader)
        history['val_loss'].append(avg_val_loss)
        
        # Update scheduler
        scheduler.step(avg_val_loss)
        
        print(f'Epoch {epoch+1}/{NUM_EPOCHS}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}')
        
        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            
            # Save best model
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'lstm_model.pth'))
            
            # Save sample of attention weights
            attention_sample = np.concatenate(attention_weights_list[:5], axis=0)
            np.save(os.path.join(OUTPUT_DIR, 'attention_weights_sample.npy'), attention_sample)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f'Early stopping triggered after {epoch+1} epochs')
                break
    
    # Load best model
    model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, 'lstm_model.pth')))
    
    # Plot training history
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('LSTM Model Training History')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, 'lstm_training_history.png'))
    
    return model

def train_regime_identification_model(train_loader, val_loader, input_size, num_regimes=3):
    """Train a LSTM model for climate regime identification."""
    print("\n===== Training Regime Identification Model =====")
    
    # Define model architecture
    hidden_size = 128
    num_layers = 2
    model = RegimeIdentificationLSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_regimes=num_regimes,
        dropout=0.3
    ).to(DEVICE)
    
    print(f"Regime Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # For the regime model, we would typically use labeled data for different regimes
    # For this example, we'll adapt to use our existing data by creating synthetic regime labels
    # In a real application, you would want properly labeled regime data
    
    # Train model
    # Note: This would need customization for your specific regime classification task
    # For example only - not actually useful without labeled data
    # trained_model, history = train_with_early_stopping(...)
    
    # For now, we'll return the untrained model as a placeholder
    return model

def evaluate_model(model, test_loader, model_type='tcn'):
    """Evaluate model performance on test data."""
    print(f"\n===== Evaluating {model_type.upper()} Model =====")
    
    model.eval()
    all_outputs = []
    all_targets = []
    
    with torch.no_grad():
        for batch in test_loader:
            if len(batch) == 3:  # With weights
                inputs, targets, _ = batch
            else:  # Without weights
                inputs, targets = batch
            
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            
            if model_type == 'lstm':
                outputs, _ = model(inputs)
            else:
                outputs = model(inputs)
            
            all_outputs.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
    
    # Concatenate batches
    y_pred = np.concatenate(all_outputs, axis=0)
    y_true = np.concatenate(all_targets, axis=0)
    
    # Calculate metrics
    metrics = regression_metrics(y_true, y_pred)
    for name, value in metrics.items():
        print(f"{name}: {value}")
    
    # Calculate climate-specific metrics
    climate_metrics = climate_specific_metrics(y_true, y_pred)
    for name, value in climate_metrics.items():
        print(f"{name}: {value}")
    
    # Plot predictions vs actual
    plt.figure(figsize=(12, 6))
    plt.plot(y_true[:100, 0], label='Actual')
    plt.plot(y_pred[:100, 0], label='Predicted')
    plt.title(f'{model_type.upper()} Model: Predictions vs Actual')
    plt.xlabel('Time Step')
    plt.ylabel('Temperature')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, f'{model_type}_predictions.png'))
    
    # Plot error
    error = y_true - y_pred
    plt.figure(figsize=(12, 6))
    plt.plot(error[:100, 0])
    plt.title(f'{model_type.upper()} Model: Prediction Error')
    plt.xlabel('Time Step')
    plt.ylabel('Error')
    plt.axhline(y=0, color='r', linestyle='-')
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, f'{model_type}_error.png'))
    
    return y_true, y_pred, metrics

def visualize_attention_weights(model, test_loader):
    """Visualize attention weights from LSTM model."""
    print("\n===== Visualizing Attention Weights =====")
    
    model.eval()
    all_attention_weights = []
    
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if i >= 5:  # Only process a few batches for visualization
                break
                
            if len(batch) == 3:  # With weights
                inputs, targets, _ = batch
            else:  # Without weights
                inputs, targets = batch
            
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            _, attention_weights = model(inputs)
            
            all_attention_weights.append(attention_weights.cpu().numpy())
    
    # Concatenate batches
    attention_weights = np.concatenate(all_attention_weights, axis=0)
    
    # Plot attention weights heatmap
    plt.figure(figsize=(12, 8))
    sample_idx = 0  # First sample in batch
    plt.imshow(attention_weights[sample_idx].reshape(-1, 1), cmap='viridis', aspect='auto')
    plt.colorbar(label='Attention Weight')
    plt.title('Attention Weights Over Input Sequence')
    plt.xlabel('Weight Magnitude')
    plt.ylabel('Time Step')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'attention_weights.png'))
    
    # Plot attention weights line plot for a few samples
    plt.figure(figsize=(12, 6))
    for i in range(min(5, attention_weights.shape[0])):
        plt.plot(attention_weights[i].flatten(), label=f'Sample {i+1}')
    plt.title('Attention Weight Distribution Over Sequence')
    plt.xlabel('Time Step')
    plt.ylabel('Attention Weight')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, 'attention_weights_comparison.png'))

def compare_models(tcn_metrics, lstm_metrics):
    """Compare performance of different models."""
    print("\n===== Model Comparison =====")
    
    # Create comparison table
    metrics = ['MSE', 'RMSE', 'MAE', 'R²', 'NSE']
    model_names = ['TCN', 'LSTM']
    
    results = {
        'TCN': [tcn_metrics['MSE'], tcn_metrics['RMSE'], tcn_metrics['MAE'], 
               tcn_metrics['R²'], tcn_metrics.get('NSE', 0)],
        'LSTM': [lstm_metrics['MSE'], lstm_metrics['RMSE'], lstm_metrics['MAE'], 
                lstm_metrics['R²'], lstm_metrics.get('NSE', 0)]
    }
    
    comparison_df = pd.DataFrame(results, index=metrics)
    print(comparison_df)
    
    # Plot comparison
    plt.figure(figsize=(10, 6))
    x = np.arange(len(metrics))
    width = 0.35
    
    # Normalize metrics for better visualization
    max_values = comparison_df.max(axis=1)
    normalized_df = comparison_df.div(max_values, axis=0)
    
    plt.bar(x - width/2, normalized_df['TCN'], width, label='TCN')
    plt.bar(x + width/2, normalized_df['LSTM'], width, label='LSTM')
    
    plt.title('Model Performance Comparison (Normalized)')
    plt.xticks(x, metrics)
    plt.ylabel('Normalized Metric Value')
    plt.legend()
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'model_comparison.png'))
    
    return comparison_df

def main():
    """Main function to run the neural network analysis."""
    print("Starting Berkeley Earth Temperature Neural Network Analysis...")
    
    # 1. Load and preprocess data
    train_data, val_data, test_data, all_columns = load_and_preprocess_data(
        GLOBAL_TEMP_FILE, target_col='LandAverageTemperature'
    )
    
    # 2. Prepare sequence data
    sequence_data = prepare_sequence_data(
        train_data, val_data, test_data, 
        target_col='temperature',
        feature_cols=None  # Auto-select numeric features
    )
    (X_train, y_train, X_val, y_val, X_test, y_test,
     weights_train, weights_val, weights_test, feature_cols) = sequence_data
    
    # 3. Create DataLoaders
    train_loader, val_loader, test_loader = create_dataloaders(
        X_train, y_train, X_val, y_val, X_test, y_test,
        weights_train, weights_val, weights_test
    )
    
    # 4. Train TCN model
    input_size = X_train.shape[2]  # Number of features
    tcn_model = train_tcn_model(train_loader, val_loader, input_size)
    
    # 5. Train LSTM model
    lstm_model = train_lstm_model(train_loader, val_loader, input_size)
    
    # 6. Evaluate models
    y_true_tcn, y_pred_tcn, tcn_metrics = evaluate_model(tcn_model, test_loader, 'tcn')
    y_true_lstm, y_pred_lstm, lstm_metrics = evaluate_model(lstm_model, test_loader, 'lstm')
    
    # 7. Visualize attention weights
    visualize_attention_weights(lstm_model, test_loader)
    
    # 8. Compare models
    comparison_df = compare_models(tcn_metrics, lstm_metrics)
    
    print("\nAnalysis completed! Results saved to:", OUTPUT_DIR)
    
    return (tcn_model, lstm_model, comparison_df)

if __name__ == "__main__":
    main()
