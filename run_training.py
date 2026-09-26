import os
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

# Import the model architecture defined previously
from train_model import IMUDeadReckoningLSTM
from sih_code import load_and_sync_nclt_session

def train_and_evaluate():
    # 1. Load your synchronized sliding windows from the NCLT session folder
    session_dir = "data/2013-01-10"
    X, y = load_and_sync_nclt_session(session_dir)
    
    # Convert data to PyTorch tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    
    # Create a DataLoader for batching (e.g., batch size of 64)
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    # 2. Initialize Model, Loss Function, and Optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = IMUDeadReckoningLSTM(input_size=9, hidden_size=64, output_size=2).to(device)
    
    # Planar Root Mean Square Error loss component objective function
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 3. Training Loop
    model.train()
    epochs = 3 # Increase epochs for full training runs
    print("Starting training loop...")
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_X, batch_y in dataloader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            predictions = model(batch_X)
            
            # Compute Euclidean distance / RMSE loss over planar coordinates
            loss = torch.mean(torch.sqrt(torch.sum((predictions - batch_y) ** 2, dim=1) + 1e-8))
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * batch_X.size(0)
            
        total_epoch_loss = epoch_loss / len(dataset)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_epoch_loss:.4f}")
        
    print("Training complete!")
    
    # 4. Save trained model for ONNX export later
    torch.save(model.state_dict(), "imu_lstm_model.pth")
    print("Model weights saved to imu_lstm_model.pth")

if __name__ == "__main__":
    train_and_evaluate()