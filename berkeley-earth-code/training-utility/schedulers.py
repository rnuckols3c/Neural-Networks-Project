"""
Learning rate scheduling utilities.
"""

import numpy as np
import torch
from torch.optim.lr_scheduler import _LRScheduler

class CyclicalLR(_LRScheduler):
    """
    Cyclical learning rate scheduler.
    
    This scheduler adjusts the learning rate cyclically between a minimum and maximum value.
    Based on the approach from the paper "Cyclical Learning Rates for Training Neural Networks."
    """
    def __init__(self, optimizer, base_lr, max_lr, step_size, mode='triangular', gamma=1.0, scale_fn=None, last_epoch=-1):
        """
        Initialize the scheduler.
        
        Parameters:
        -----------
        optimizer : torch.optim.Optimizer
            The optimizer for which to adjust the learning rate.
        base_lr : float
            Lower learning rate boundary in the cycle.
        max_lr : float
            Upper learning rate boundary in the cycle.
        step_size : int
            Number of iterations in half a cycle. 
            The full cycle length is 2*step_size.
        mode : str, optional (default='triangular')
            One of {triangular, triangular2, exp_range}.
            If scale_fn is provided, this is ignored.
        gamma : float, optional (default=1.0)
            Constant in 'exp_range' scaling function: gamma^global_step
        scale_fn : function, optional (default=None)
            Custom scaling function.
        last_epoch : int, optional (default=-1)
            The index of the last epoch.
        """
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.gamma = gamma
        
        if scale_fn is None:
            if self.mode == 'triangular':
                self.scale_fn = lambda x: 1.0
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = lambda x: 1 / (2 ** (x - 1))
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = lambda x: gamma ** x
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = 'iterations'
        
        super(CyclicalLR, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        """Calculate the learning rate based on the current epoch."""
        cycle = np.floor(1 + self.last_epoch / (2 * self.step_size))
        x = np.abs(self.last_epoch / self.step_size - 2 * cycle + 1)
        
        if self.scale_mode == 'cycle':
            lr = self.base_lr + (self.max_lr - self.base_lr) * max(0, (1 - x)) * self.scale_fn(cycle)
        else:
            lr = self.base_lr + (self.max_lr - self.base_lr) * max(0, (1 - x)) * self.scale_fn(self.last_epoch)
        
        return [lr for _ in self.base_lrs]

class WarmupScheduler(_LRScheduler):
    """
    Warmup learning rate scheduler.
    
    Gradually increases the learning rate from a small value to the initial learning rate 
    over a number of iterations, then uses another scheduler for the rest of training.
    """
    def __init__(self, optimizer, warmup_steps, base_scheduler=None, last_epoch=-1):
        """
        Initialize the scheduler.
        
        Parameters:
        -----------
        optimizer : torch.optim.Optimizer
            The optimizer for which to adjust the learning rate.
        warmup_steps : int
            Number of warmup steps.
        base_scheduler : torch.optim.lr_scheduler._LRScheduler, optional (default=None)
            Scheduler to use after warmup.
        last_epoch : int, optional (default=-1)
            The index of the last epoch.
        """
        self.warmup_steps = warmup_steps
        self.base_scheduler = base_scheduler
        self.last_step = 0
        super(WarmupScheduler, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        """Calculate the learning rate based on the current epoch."""
        if self.last_step < self.warmup_steps:
            # Linear warmup
            alpha = self.last_step / self.warmup_steps
            lr = [base_lr * alpha for base_lr in self.base_lrs]
        else:
            if self.base_scheduler is not None:
                # Forward to base scheduler
                self.base_scheduler.step(self.last_step - self.warmup_steps)
                lr = self.base_scheduler.get_lr()
            else:
                # Keep the initial learning rate
                lr = self.base_lrs
        
        self.last_step += 1
        return lr
