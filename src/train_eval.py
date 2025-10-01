import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

import config
from model import CrowdCCT

def train_one_epoch(model, train_loader, optimizer, criterion, device):
    """Runs the training loop for one epoch."""
    model.train()
    total_loss = 0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        
        loss = criterion(outputs, targets) 
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    return total_loss / len(train_loader)

def evaluate_model(model, test_loader, device):
    """Evaluates the model on the test set and calculates MAE/MSE."""
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            
            all_preds.extend(outputs.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    # MAE (Mean Absolute Error) is the L1 metric
    mae = np.mean(np.abs(all_preds - all_targets)) 
    # MSE (Mean Squared Error)
    mse = np.mean((all_preds - all_targets)**2)
    
    return mae, mse

def plot_history(history):
    """Plots and saves the training loss and test MAE history."""
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.title('Training Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (L1)')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['test_mae'], label='Test MAE')
    plt.title('Test MAE over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    
    plt.savefig(config.HISTORY_PLOT_PATH)
    print(f"Training history saved as {config.HISTORY_PLOT_PATH}")