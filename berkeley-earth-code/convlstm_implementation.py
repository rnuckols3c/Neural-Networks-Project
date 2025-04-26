"""
Berkeley Earth Temperature Dataset - ConvLSTM Analysis
=====================================================
Implementation of the ConvLSTM model for spatial-temporal temperature analysis,
focusing on urban heat island effects.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings('ignore')

# Import necessary utilities
import sys
sys.path.append('./data-preprocessing')
from data_cleaning import fill_spatial_gaps
from spatial_processing import create_temperature_grid, add_spatial_features, create_spatiotemporal_sequences
from data_splitting import temporal_split

# Import model implementation
sys.path.append('./model-implementation')
from convlstm_model import UHIAnalysisModel

# Import training utilities
sys.path.append('./training-utility')
from training import train_with_early_stopping
from evaluation import regression_metrics
from visualization import plot_spatial_predictions

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Set paths
DATA_DIR = r"C:\Users\Richard Nuckols\Desktop\Desktop\Personal\JHU\Neural Networks\Module7\Data"
CITIES_FILE = os.path.join(DATA_DIR, "GlobalLandTemperaturesByCity.csv")
MAJOR_CITIES_FILE = os.path.join(DATA_DIR, "GlobalLandTemperaturesByMajorCity.csv")
OUTPUT_DIR = os.path.join(DATA_DIR, "Spatial_Model_Results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Configuration parameters
SEQUENCE_LENGTH = 12  # 1 year of monthly data
HORIZON = 1  # Predict next month
LAT_RES = 5  # Latitude resolution in degrees
LON_RES = 5  # Longitude resolution in degrees
BATCH_SIZE = 8
NUM_EPOCHS = 50
LEARNING_RATE = 0.0005
PATIENCE = 10
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

def create_dataloaders(X_train, y_train, X_val, y_val, X_test, y_test,
                     urban_mask_train=None, urban_mask_val=None, urban_mask_test=None):
    """Create PyTorch DataLoaders for spatial-temporal data."""
    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
    
    # Create datasets
    if urban_mask_train is not None:
        urban_mask_train_tensor = torch.tensor(urban_mask_train, dtype=torch.float32)
        urban_mask_val_tensor = torch.tensor(urban_mask_val, dtype=torch.float32)
        urban_mask_test_tensor = torch.tensor(urban_mask_test, dtype=torch.float32)
        
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor, urban_mask_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor, urban_mask_val_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor, urban_mask_test_tensor)
    else:
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    return train_loader, val_loader, test_loader

def load_and_process_spatial_data():
    """
    Load and process spatial temperature data.
    
    Returns:
    --------
    tuple
        Processed gridded data and metadata
    """
    print(f"Loading data from {MAJOR_CITIES_FILE}...")
    
    # Load city temperature data
    df = pd.read_csv(MAJOR_CITIES_FILE, parse_dates=['dt'])
    df.set_index('dt', inplace=True)
    print(f"Data loaded, shape: {df.shape}")
    
    # Extract numeric latitude and longitude
    df['latitude'] = df['Latitude'].str.replace('[NS]', '', regex=True).astype(float)
    df['latitude'] = df['latitude'] * df['Latitude'].str.contains('S').map({True: -1, False: 1})
    
    df['longitude'] = df['Longitude'].str.replace('[EW]', '', regex=True).astype(float)
    df['longitude'] = df['longitude'] * df['Longitude'].str.contains('W').map({True: -1, False: 1})
    
    # Rename temperature column
    df['temperature'] = df['AverageTemperature']
    if 'AverageTemperatureUncertainty' in df.columns:
        df['uncertainty'] = df['AverageTemperatureUncertainty']
    
    print("Coordinates extracted and normalized")
    
    # Focus on recent years for better data quality
    recent_years = 50
    cutoff_year = df.index.year.max() - recent_years
    df = df[df.index.year >= cutoff_year]
    
    # Create gridded temperature data
    print("Creating temperature grid...")
    grid_data, grid_metadata = create_temperature_grid(
        df, lat_res=LAT_RES, lon_res=LON_RES, method='weighted_mean'
    )
    print(f"Grid created with shape: {grid_data.shape}")
    
    # Fill gaps in the grid
    print("Filling spatial gaps...")
    filled_grid = fill_spatial_gaps(grid_data, method='interpolate')
    
    # Add spatial features
    print("Adding spatial features...")
    enhanced_grid = add_spatial_features(filled_grid, grid_metadata)
    print(f"Enhanced grid created with shape: {enhanced_grid.shape}")
    
    # Split data temporally
    n_timesteps = enhanced_grid.shape[0]
    train_idx = int(n_timesteps * 0.7)
    val_idx = int(n_timesteps * 0.85)
    
    train_grid = enhanced_grid[:train_idx]
    val_grid = enhanced_grid[train_idx:val_idx]
    test_grid = enhanced_grid[val_idx:]
    
    print(f"Data split: Train={train_grid.shape}, Val={val_grid.shape}, Test={test_grid.shape}")
    
    # Create sequences for spatiotemporal modeling
    print("Creating spatiotemporal sequences...")
    X_train, y_train = create_spatiotemporal_sequences(train_grid, SEQUENCE_LENGTH, HORIZON)
    X_val, y_val = create_spatiotemporal_sequences(val_grid, SEQUENCE_LENGTH, HORIZON)
    X_test, y_test = create_spatiotemporal_sequences(test_grid, SEQUENCE_LENGTH, HORIZON)
    
    print(f"Sequences created: X_train={X_train.shape}, y_train={y_train.shape}")
    
    # Create urban mask (simplified - in reality would need urban area data)
    # For this example, we'll use a simple threshold approach to identify urban areas
    # Cities tend to be warmer than surrounding areas
    urban_threshold = np.percentile(y_train[:, :, :, 0].reshape(-1), 75)  # Top 25% warmest areas
    urban_mask_train = (y_train[:, :, :, 0] > urban_threshold).astype(np.float32)
    urban_mask_val = (y_val[:, :, :, 0] > urban_threshold).astype(np.float32)
    urban_mask_test = (y_test[:, :, :, 0] > urban_threshold).astype(np.float32)
    
    return (X_train, y_train, X_val, y_val, X_test, y_test,
            urban_mask_train, urban_mask_val, urban_mask_test, 
            grid_metadata)

def train_convlstm_model(train_loader, val_loader, input_shape):
    """Train a ConvLSTM model for spatial-temporal temperature prediction."""
    print("\n===== Training ConvLSTM Model =====")
    
    # Extract input dimensions
    _, seq_len, height, width, channels = input_shape
    
    # Define model architecture
    hidden_channels = [32, 64, 32]  # Number of hidden channels in each layer
    kernel_size = 3
    num_layers = len(hidden_channels)
    output_channels = 1  # Temperature prediction
    
    model = UHIAnalysisModel(
        input_channels=channels,
        hidden_channels=hidden_channels,
        kernel_size=kernel_size,
        num_layers=num_layers,
        output_channels=output_channels
    ).to(DEVICE)
    
    print(f"ConvLSTM Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Define loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Custom training loop for ConvLSTM model
    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(NUM_EPOCHS):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for batch in train_loader:
            if len(batch) == 3:  # With urban mask
                inputs, targets, urban_masks = batch
                inputs = inputs.to(DEVICE)
                targets = targets.to(DEVICE)
                urban_masks = urban_masks.to(DEVICE)
            else:  # Without urban mask
                inputs, targets = batch
                inputs = inputs.to(DEVICE)
                targets = targets.to(DEVICE)
                urban_masks = None
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs, urban_probs, _ = model(inputs, urban_masks)
            
            # Get predictions for the last time step
            pred_temps = outputs[:, -1, :, :, 0]
            
            # Calculate loss
            loss = criterion(pred_temps, targets)
            
            # Add urban heat island detection loss if mask is available
            if urban_masks is not None:
                urban_loss = criterion(urban_probs.squeeze(1), urban_masks)
                loss = loss + 0.2 * urban_loss  # Weight for UHI detection task
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                if len(batch) == 3:  # With urban mask
                    inputs, targets, urban_masks = batch
                    inputs = inputs.to(DEVICE)
                    targets = targets.to(DEVICE)
                    urban_masks = urban_masks.to(DEVICE)
                else:  # Without urban mask
                    inputs, targets = batch
                    inputs = inputs.to(DEVICE)
                    targets = targets.to(DEVICE)
                    urban_masks = None
                
                # Forward pass
                outputs, urban_probs, _ = model(inputs, urban_masks)
                
                # Get predictions for the last time step
                pred_temps = outputs[:, -1, :, :, 0]
                
                # Calculate loss
                loss = criterion(pred_temps, targets)
                
                # Add urban heat island detection loss if mask is available
                if urban_masks is not None:
                    urban_loss = criterion(urban_probs.squeeze(1), urban_masks)
                    loss = loss + 0.2 * urban_loss
                
                val_loss += loss.item()
        
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
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'convlstm_model.pth'))
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f'Early stopping triggered after {epoch+1} epochs')
                break
    
    # Load best model
    model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, 'convlstm_model.pth')))
    
    # Plot training history
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('ConvLSTM Model Training History')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, 'convlstm_training_history.png'))
    
    return model

def evaluate_convlstm_model(model, test_loader, grid_metadata):
    """Evaluate ConvLSTM model performance on test data."""
    print("\n===== Evaluating ConvLSTM Model =====")
    
    model.eval()
    all_preds = []
    all_targets = []
    all_urban_probs = []
    all_attention_weights = []
    
    with torch.no_grad():
        for batch in test_loader:
            if len(batch) == 3:  # With urban mask
                inputs, targets, urban_masks = batch
                inputs = inputs.to(DEVICE)
                targets = targets.to(DEVICE)
                urban_masks = urban_masks.to(DEVICE)
            else:  # Without urban mask
                inputs, targets = batch
                inputs = inputs.to(DEVICE)
                targets = targets.to(DEVICE)
                urban_masks = None
            
            # Forward pass
            outputs, urban_probs, attention_weights = model(inputs, urban_masks)
            
            # Get predictions for the last time step
            pred_temps = outputs[:, -1, :, :, 0]
            
            all_preds.append(pred_temps.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            all_urban_probs.append(urban_probs.squeeze(1).cpu().numpy())
            all_attention_weights.append(attention_weights.cpu().numpy())
    
    # Concatenate batches
    y_pred = np.concatenate(all_preds, axis=0)
    y_true = np.concatenate(all_targets, axis=0)
    urban_probs = np.concatenate(all_urban_probs, axis=0)
    
    # Calculate metrics
    # Reshape to 2D for sklearn metrics
    y_pred_flat = y_pred.reshape(-1)
    y_true_flat = y_true.reshape(-1)
    # Remove NaN values if any
    mask = ~np.isnan(y_true_flat) & ~np.isnan(y_pred_flat)
    y_pred_flat = y_pred_flat[mask]
    y_true_flat = y_true_flat[mask]
    
    metrics = regression_metrics(y_true_flat, y_pred_flat)
    for name, value in metrics.items():
        print(f"{name}: {value}")
    
    # Visualize predictions
    sample_idx = 0  # First sample in test set
    
    # Plot spatial predictions
    fig = plot_spatial_predictions(
        y_true, y_pred, 
        grid_metadata['lat_bins'], 
        grid_metadata['lon_bins'],
        time_idx=sample_idx, 
        title='ConvLSTM: Spatial Temperature Predictions'
    )
    fig.savefig(os.path.join(OUTPUT_DIR, 'convlstm_spatial_predictions.png'))
    
    # Plot urban heat island probabilities
    plt.figure(figsize=(10, 8))
    plt.imshow(urban_probs[sample_idx], cmap='hot', vmin=0, vmax=1,
              extent=[
                  grid_metadata['lon_bins'][0], 
                  grid_metadata['lon_bins'][-1], 
                  grid_metadata['lat_bins'][0], 
                  grid_metadata['lat_bins'][-1]
              ])
    plt.colorbar(label='Urban Heat Island Probability')
    plt.title('Urban Heat Island Detection')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.savefig(os.path.join(OUTPUT_DIR, 'urban_heat_island_probs.png'))
    
    # Plot attention weights
    plt.figure(figsize=(10, 8))
    plt.imshow(all_attention_weights[0][sample_idx], cmap='viridis',
              extent=[
                  grid_metadata['lon_bins'][0], 
                  grid_metadata['lon_bins'][-1], 
                  grid_metadata['lat_bins'][0], 
                  grid_metadata['lat_bins'][-1]
              ])
    plt.colorbar(label='Attention Weight')
    plt.title('Spatial Attention Weights')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.savefig(os.path.join(OUTPUT_DIR, 'spatial_attention_weights.png'))
    
    return y_true, y_pred, metrics

def main():
    """Main function to run the ConvLSTM analysis."""
    print("Starting Berkeley Earth Temperature ConvLSTM Analysis...")
    
    # 1. Load and process spatial data
    spatial_data = load_and_process_spatial_data()
    (X_train, y_train, X_val, y_val, X_test, y_test,
     urban_mask_train, urban_mask_val, urban_mask_test,
     grid_metadata) = spatial_data
    
    # 2. Create DataLoaders
    train_loader, val_loader, test_loader = create_dataloaders(
        X_train, y_train, X_val, y_val, X_test, y_test,
        urban_mask_train, urban_mask_val, urban_mask_test
    )
    
    # 3. Train ConvLSTM model
    convlstm_model = train_convlstm_model(train_loader, val_loader, X_train.shape)
    
    # 4. Evaluate model
    y_true, y_pred, metrics = evaluate_convlstm_model(convlstm_model, test_loader, grid_metadata)
    
    print("\nConvLSTM Analysis completed! Results saved to:", OUTPUT_DIR)
    
    return convlstm_model, metrics

if __name__ == "__main__":
    main()
