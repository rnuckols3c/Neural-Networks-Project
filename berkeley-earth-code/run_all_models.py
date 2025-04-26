"""
Berkeley Earth Temperature Dataset - Comprehensive Neural Network Analysis
=========================================================================
Master script to run all neural network models for temperature analysis.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import warnings
warnings.filterwarnings('ignore')

# Import main scripts
# Add current directory to path for relative imports
sys.path.append('.')

def print_section_header(title):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)

def run_model(model_name, data_path):
    """Run a specific model."""
    if model_name == 'tcn' or model_name == 'lstm' or model_name == 'all_temporal':
        print_section_header(f"Running {'Temporal Models' if model_name == 'all_temporal' else model_name.upper() + ' Model'}")
        
        # Import the main implementation script
        sys.path.append('./berkeley-earth-code')
        from main_implementation import main as run_temporal_models
        
        # Run the model
        if model_name == 'all_temporal':
            # Run both TCN and LSTM models
            run_temporal_models()
        else:
            # For future: Add selective model running
            run_temporal_models()
            
    elif model_name == 'convlstm':
        print_section_header("Running ConvLSTM Model")
        
        # Import the ConvLSTM implementation script
        sys.path.append('./berkeley-earth-code')
        from convlstm_implementation import main as run_convlstm
        
        # Run the ConvLSTM model
        run_convlstm()
        
    elif model_name == 'all':
        print_section_header("Running All Models")
        
        # Run temporal models
        sys.path.append('./berkeley-earth-code')
        from main_implementation import main as run_temporal_models
        
        # Run ConvLSTM model
        from convlstm_implementation import main as run_convlstm
        
        # Run all models
        print_section_header("Running Temporal Models (TCN and LSTM)")
        run_temporal_models()
        
        print_section_header("Running Spatial-Temporal Model (ConvLSTM)")
        run_convlstm()
        
        print_section_header("All Models Completed Successfully")
    else:
        print(f"Error: Unknown model '{model_name}'")
        return

def main():
    """Main function to parse arguments and run models."""
    parser = argparse.ArgumentParser(description='Run Berkeley Earth temperature neural network models.')
    
    parser.add_argument('--model', type=str, default='all',
                        choices=['tcn', 'lstm', 'convlstm', 'all_temporal', 'all'],
                        help='Which model to run (default: all)')
    
    parser.add_argument('--data_path', type=str, 
                        default=r"C:\Users\Richard Nuckols\Desktop\Desktop\Personal\JHU\Neural Networks\Module7\Data",
                        help='Path to the Berkeley Earth data directory')
    
    args = parser.parse_args()
    
    print("Berkeley Earth Temperature Neural Network Analysis")
    print(f"Selected model: {args.model}")
    print(f"Data path: {args.data_path}")
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    # Run the selected model
    run_model(args.model, args.data_path)

if __name__ == "__main__":
    main()
