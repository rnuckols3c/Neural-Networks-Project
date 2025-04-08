# Berkeley Earth Neural Network Analysis

This directory contains the code and plan for applying neural networks to the Berkeley Earth Temperature Dataset.

## Files Included

### Project Plan
- `berkeley-earth-plan.md` - Detailed project plan for neural network modeling

### Python Code Files
- `berkeley-earth-code/tcn_model.py` - Temporal Convolutional Network implementation
- `berkeley-earth-code/lstm_model.py` - LSTM/GRU Network implementation
- `berkeley-earth-code/convlstm_model.py` - Convolutional LSTM implementation
- `berkeley-earth-code/data-preprocessing` - Data preprocessing utilities
- `berkeley-earth-code/training-utils` - Training and evaluation utilities

## Using the Code in Spyder

1. Download or clone this repository to your local machine
2. Open Spyder
3. Create a new script that imports the necessary modules:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

# Add the path to your code files (if needed)
import sys
sys.path.append('/path/to/repository')

# Import the modules
from berkeley-earth-code.data_preprocessing import create_time_features, handle_missing_and_uncertainty
from berkeley-earth-code.tcn_model import TCNForecaster
from berkeley-earth-code.lstm_model import BidirectionalAttentionLSTM
from berkeley-earth-code.training_utils import train_with_early_stopping, regression_metrics# Neural-Networks-Project
