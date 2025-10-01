import torch.optim as optim
import torch.nn as nn
import torch

import config
from data_loader import get_data_loaders
from model import CrowdCCT
from train_eval import train_one_epoch, evaluate_model, plot_history

def main():
    """Main function to setup and run the CrowdCCT training pipeline."""
    
    print("--- Starting CrowdCCT Project ---")
    print(f"Using device: {config.DEVICE}")

    # 1. Setup Data Loaders
    train_loader, test_loader = get_data_loaders()
    print(f"Data Loaded: Train Samples={len(train_loader.dataset)}, Test Samples={len(test_loader.dataset)}")

    # 2. Setup Model, Optimizer, and Loss
    model = CrowdCCT().to(config.DEVICE)
    
    # Adam Optimizer (as per paper)
    optimizer = optim.Adam(model.parameters(), 
                           lr=config.LEARNING_RATE, 
                           weight_decay=config.WEIGHT_DECAY) 
                           
    # L1 Loss (MAE) is the objective function
    criterion = nn.L1Loss() 
    
    print("Model and Optimizer Setup Complete.")

    # 3. Training Loop
    best_mae = float('inf')
    history = {'train_loss': [], 'test_mae': [], 'test_mse': []}
    
    for epoch in range(1, config.EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, config.DEVICE)
        mae, mse = evaluate_model(model, test_loader, config.DEVICE)
        
        print(f"Epoch {epoch}/{config.EPOCHS} | Train Loss: {train_loss:.2f} | Test MAE: {mae:.2f} | Test MSE: {mse:.2f}")

        # Record history
        history['train_loss'].append(train_loss)
        history['test_mae'].append(mae)
        history['test_mse'].append(mse)

        # Save the best model based on MAE
        if mae < best_mae:
            best_mae = mae
            torch.save(model.state_dict(), config.MODEL_OUTPUT_PATH)
            print(f"--- Model saved to {config.MODEL_OUTPUT_PATH}! ---")
            
    print(f"\nFinal Best MAE on ShanghaiTech Part-A Test Set: {best_mae:.2f}")
    
    # 4. Final Visualization
    plot_history(history)
    print("--- Project Execution Finished ---")

if __name__ == '__main__':
    main()