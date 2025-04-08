"""
Training utilities for neural network models.
"""

import torch
import torch.nn as nn
import numpy as np

def train_with_early_stopping(model, train_loader, val_loader, criterion, optimizer, 
                             scheduler, device, num_epochs, patience=10):
    """
    Train a model with early stopping based on validation loss.
    """
    best_val_loss = float('inf')
    patience_counter = 0
    training_history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            
            # Optional gradient clipping
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    pass  # Will be updated after validation
                else:
                    scheduler.step()
                
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        training_history['train_loss'].append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                inputs, targets = batch
                inputs, targets = inputs.to(device), targets.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        training_history['val_loss'].append(avg_val_loss)
        
        # Update ReduceLROnPlateau scheduler if used
        if scheduler is not None and isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(avg_val_loss)
        
        print(f'Epoch {epoch+1}/{num_epochs}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}')
        
        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping triggered after {epoch+1} epochs')
                break
    
    # Load best model
    model.load_state_dict(torch.load('best_model.pth'))
    return model, training_history

def multi_stage_training(model, train_loader, val_loader, device, feature_layers, prediction_layers):
    """Train complex models in stages to improve convergence."""
    # Stage 1: Train only the feature extraction layers
    print("Stage 1: Training feature extraction layers...")
    
    # Freeze prediction layers
    for param in prediction_layers.parameters():
        param.requires_grad = False
    
    # Create optimizer and loss for stage 1
    optimizer_stage1 = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=1e-3
    )
    criterion_stage1 = nn.MSELoss()
    
    # Train for fewer epochs in stage 1
    model, _ = train_with_early_stopping(
        model, train_loader, val_loader, 
        criterion_stage1, optimizer_stage1, None, 
        device, num_epochs=30, patience=5
    )
    
    # Stage 2: Fine-tune all layers
    print("Stage 2: Fine-tuning all layers...")
    
    # Unfreeze all layers
    for param in prediction_layers.parameters():
        param.requires_grad = True
    
    # Create optimizer with lower learning rate
    optimizer_stage2 = torch.optim.Adam(model.parameters(), lr=5e-5)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_stage2, mode='min', factor=0.5, patience=3
    )
    
    # Train the entire model
    model, history = train_with_early_stopping(
        model, train_loader, val_loader, 
        criterion_stage1, optimizer_stage2, scheduler, 
        device, num_epochs=100, patience=10
    )
    
    return model, history

def transfer_learning_approach(source_model, target_dataset, hidden_size, new_output_size, lr=1e-4):
    """Apply transfer learning from one model to another task."""
    # Freeze early layers (feature extractors)
    for param in source_model.feature_layers.parameters():
        param.requires_grad = False
    
    # Replace and initialize output layers for the new task
    source_model.output_layer = nn.Linear(hidden_size, new_output_size)
    
    # Create optimizer that only updates unfrozen parameters
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, source_model.parameters()), lr=lr)
    
    return source_model, optimizer
