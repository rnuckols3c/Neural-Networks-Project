"""
Custom loss functions for neural network models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class UncertaintyWeightedLoss(nn.Module):
    """
    Loss function that weights samples by inverse uncertainty.
    
    This loss function gives more weight to samples with lower uncertainty,
    effectively making the model focus more on more reliable data points.
    """
    def __init__(self, base_loss='mse'):
        """
        Initialize the loss function.
        
        Parameters:
        -----------
        base_loss : str, optional (default='mse')
            Base loss function type: 'mse' or 'mae'.
        """
        super(UncertaintyWeightedLoss, self).__init__()
        self.base_loss = base_loss
    
    def forward(self, predictions, targets, uncertainties):
        """
        Calculate the weighted loss.
        
        Parameters:
        -----------
        predictions : torch.Tensor
            Model predictions.
        targets : torch.Tensor
            Target values.
        uncertainties : torch.Tensor
            Uncertainty estimates for each sample.
            
        Returns:
        --------
        torch.Tensor
            Weighted loss value.
        """
        # Clip uncertainties to avoid division by zero
        eps = 1e-8
        weights = 1.0 / (uncertainties + eps)
        
        # Normalize weights
        weights = weights / weights.mean()
        
        # Calculate base loss
        if self.base_loss == 'mse':
            losses = F.mse_loss(predictions, targets, reduction='none')
        elif self.base_loss == 'mae':
            losses = F.l1_loss(predictions, targets, reduction='none')
        
        # Apply weights and take mean
        weighted_loss = (losses * weights).mean()
        
        return weighted_loss

class TemporalAttentionLoss(nn.Module):
    """
    Loss function that pays more attention to recent time steps.
    
    This loss function applies increasing weights to more recent time steps,
    making the model focus more on recent data for its predictions.
    """
    def __init__(self, base_loss='mse', alpha=0.5):
        """
        Initialize the loss function.
        
        Parameters:
        -----------
        base_loss : str, optional (default='mse')
            Base loss function type: 'mse' or 'mae'.
        alpha : float, optional (default=0.5)
            Weight decay factor. Higher values give more weight to recent time steps.
        """
        super(TemporalAttentionLoss, self).__init__()
        self.base_loss = base_loss
        self.alpha = alpha
    
    def forward(self, predictions, targets, sequence_lens=None):
        """
        Calculate the temporally weighted loss.
        
        Parameters:
        -----------
        predictions : torch.Tensor
            Model predictions.
        targets : torch.Tensor
            Target values.
        sequence_lens : torch.Tensor, optional (default=None)
            Length of each sequence in the batch.
            
        Returns:
        --------
        torch.Tensor
            Temporally weighted loss value.
        """
        batch_size, seq_len = predictions.shape[0], predictions.shape[1]
        
        # Generate temporal weights
        time_steps = torch.arange(seq_len, device=predictions.device, dtype=torch.float32)
        weights = torch.exp(self.alpha * time_steps / seq_len)
        
        # Adjust weights based on sequence lengths if provided
        if sequence_lens is not None:
            mask = torch.zeros((batch_size, seq_len), device=predictions.device)
            for i, length in enumerate(sequence_lens):
                mask[i, :length] = 1.0
            weights = weights * mask
        
        # Normalize weights
        weights = weights / weights.sum(dim=1, keepdim=True)
        
        # Calculate base loss
        if self.base_loss == 'mse':
            losses = F.mse_loss(predictions, targets, reduction='none')
        elif self.base_loss == 'mae':
            losses = F.l1_loss(predictions, targets, reduction='none')
        
        # Apply weights and take mean
        weighted_loss = (losses * weights).sum() / batch_size
        
        return weighted_loss
