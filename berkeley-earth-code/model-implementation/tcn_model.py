import torch
import torch.nn as nn
import torch.nn.functional as F

class TCNBlock(nn.Module):
    def __init__(self, input_size, output_size, kernel_size, dilation, dropout=0.2):
        super(TCNBlock, self).__init__()
        self.conv1 = nn.Conv1d(input_size, output_size, kernel_size, 
                              padding=(kernel_size-1) * dilation, 
                              dilation=dilation)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = nn.Conv1d(output_size, output_size, kernel_size, 
                              padding=(kernel_size-1) * dilation, 
                              dilation=dilation)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        self.downsample = nn.Conv1d(input_size, output_size, 1) if input_size != output_size else None
        self.relu = nn.ReLU()
    
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.dropout1(out)
        
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.dropout2(out)
        
        if self.downsample is not None:
            residual = self.downsample(residual)
        
        return self.relu(out + residual)

class TemporalConvNet(nn.Module):
    def __init__(self, input_channels, num_channels, kernel_size=3, dropout=0.2):
        """
        Temporal Convolutional Network for temperature prediction.
        
        Parameters:
        -----------
        input_channels : int
            Number of input features
        num_channels : list
            Number of channels in each layer
        kernel_size : int, optional (default=3)
            Kernel size for all convolutions
        dropout : float, optional (default=0.2)
            Dropout probability
        """
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_channels if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            
            layers.append(TCNBlock(in_channels, out_channels, kernel_size, 
                                  dilation=dilation_size, dropout=dropout))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Forward pass through the TCN.
        
        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape (batch_size, input_channels, sequence_length)
            
        Returns:
        --------
        torch.Tensor
            Output tensor of shape (batch_size, num_channels[-1], sequence_length)
        """
        return self.network(x)

class TCNForecaster(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size=3, dropout=0.2):
        """
        TCN model for temperature forecasting.
        
        Parameters:
        -----------
        input_size : int
            Number of input features
        output_size : int
            Number of output features (usually 1 for temperature prediction)
        num_channels : list
            Number of channels in each TCN layer
        kernel_size : int, optional (default=3)
            Kernel size for all convolutions
        dropout : float, optional (default=0.2)
            Dropout probability
        """
        super(TCNForecaster, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size, dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)
    
    def forward(self, x):
        """
        Forward pass through the TCN forecaster.
        
        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape (batch_size, sequence_length, input_size)
            
        Returns:
        --------
        torch.Tensor
            Output tensor of shape (batch_size, output_size)
        """
        # TCN expects inputs of shape (batch_size, channels, seq_len)
        x = x.transpose(1, 2)
        
        # Apply TCN
        output = self.tcn(x)
        
        # Global average pooling over the sequence dimension
        output = output.mean(dim=2)
        
        # Apply final linear layer
        output = self.linear(output)
        
        return output
