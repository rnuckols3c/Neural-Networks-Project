# Berkeley Earth Neural Network Analysis

This project implements neural network models for analyzing temperature patterns in the Berkeley Earth dataset. It includes three architectures:

1. **Temporal Convolutional Network (TCN)** - For capturing long-range dependencies in time series data
2. **LSTM/GRU Network** - For identifying regime shifts and extreme event patterns
3. **Convolutional LSTM** - For analyzing spatial-temporal patterns including urban heat island effects

## Project Structure

```
berkeley-earth-code/
├── data-preprocessing/      # Data preparation utilities
│   ├── data_cleaning.py
│   ├── data_normalization.py
│   ├── data_splitting.py
│   ├── spatial_processing.py
│   └── time_series_processing.py
├── model-implementation/    # Neural network model implementations
│   ├── convlstm_model.py
│   ├── lstm_model.py
│   └── tcn_model.py
├── training-utility/        # Training and evaluation utilities
│   ├── dataloaders.py
│   ├── evaluation.py
│   ├── losses.py
│   ├── schedulers.py
│   ├── training.py
│   └── visualization.py
├── main_implementation.py   # Main script for temporal models (TCN, LSTM)
├── convlstm_implementation.py # Script for spatial-temporal model (ConvLSTM)
├── run_all_models.py        # Unified script to run all models
└── README.md                # This file
```

## Requirements

- Python 3.8+
- PyTorch 1.9+
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- SciPy

## Dataset

The Berkeley Earth temperature dataset includes several CSV files:

- **GlobalTemperatures.csv** - Global land and ocean temperature averages
- **GlobalLandTemperaturesByCountry.csv** - Temperature data by country
- **GlobalLandTemperaturesByState.csv** - Temperature data by state/region
- **GlobalLandTemperaturesByMajorCity.csv** - Temperature data for major cities
- **GlobalLandTemperaturesByCity.csv** - Temperature data for all cities

The dataset path should be set to: `C:\Users\Richard Nuckols\Desktop\Desktop\Personal\JHU\Neural Networks\Module7\Data`

## Running the Models

### Option 1: Run All Models

```bash
python run_all_models.py --model all
```

### Option 2: Run Specific Models

```bash
# Run only TCN model
python run_all_models.py --model tcn

# Run only LSTM model
python run_all_models.py --model lstm

# Run only ConvLSTM model
python run_all_models.py --model convlstm

# Run both temporal models (TCN and LSTM)
python run_all_models.py --model all_temporal
```

### Option 3: Run Individual Scripts

```bash
# Run temporal models (TCN and LSTM)
python main_implementation.py

# Run spatial-temporal model (ConvLSTM)
python convlstm_implementation.py
```

## Model Architectures

### Temporal Convolutional Network (TCN)

The TCN model is designed for capturing long-range dependencies in time series data. Key features:
- Dilated causal convolutions to handle long sequences
- Residual connections for better gradient flow
- Increasing channel depth for hierarchical feature extraction

**Use Case**: Predicting future temperature trends based on historical patterns

### Bidirectional LSTM with Attention

The LSTM model is implemented with bidirectional layers and an attention mechanism. Key features:
- Bidirectional processing captures both past and future context
- Attention mechanism focuses on relevant time steps
- Multiple stacked layers for complex pattern recognition

**Use Case**: Identifying regime shifts and precursors to extreme temperature events

### Convolutional LSTM

The ConvLSTM model combines convolutional and recurrent layers for spatial-temporal analysis. Key features:
- Convolutional operations for spatial feature extraction
- LSTM cells for temporal dependencies
- Spatial attention mechanism
- Urban heat island detection capabilities

**Use Case**: Analyzing urban heat island effects and spatial temperature propagation

## Results and Output

Each model generates output in its respective directory:

- Temporal models (TCN, LSTM): `Data/Model_Results/`
- Spatial model (ConvLSTM): `Data/Spatial_Model_Results/`

The output includes:
- Trained model weights
- Training history plots
- Prediction visualizations
- Performance metrics
- Attention weights visualizations (for attention-based models)
- Urban heat island probability maps (for ConvLSTM)

## Model Performance Evaluation

Models are evaluated using multiple metrics:
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- R² Score
- Nash-Sutcliffe Efficiency (NSE)
- Anomaly Correlation Coefficient (ACC)

## Extending the Project

To extend this project:
1. Add new data sources in the data preprocessing modules
2. Implement additional neural network architectures
3. Customize the training process with different loss functions
4. Add more evaluation metrics and visualizations

## Troubleshooting

- **CUDA out of memory**: Reduce batch size or model complexity
- **Slow training**: Use a smaller subset of data for faster iteration
- **Overfitting**: Increase dropout rate or add more regularization
- **Poor performance**: Try different hyperparameters or model architectures
