import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from .dataset import RealFakeDataset
from .model import EnsembleDetector
import os
import copy
import argparse

def train_model(data_dir, num_epochs=10, batch_size=32, learning_rate=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Datasets
    # Ensure directories exist
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    
    if not os.path.exists(train_dir) or not os.path.exists(val_dir):
        print(f"Error: Data directories not found at {train_dir} or {val_dir}")
        return

    train_dataset = RealFakeDataset(train_dir, phase='train')
    val_dataset = RealFakeDataset(val_dir, phase='val')
    
    # Check if datasets are empty
    if len(train_dataset) == 0:
        print("Train dataset is empty.")
        return
    if len(val_dataset) == 0:
        print("Val dataset is empty.")
        val_dataset = train_dataset # Fallback for testing if val is empty
            
    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0),
        'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    }
    
    # Model
    model = EnsembleDetector(pretrained=True)
    model.to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
                
            running_loss = 0.0
            running_corrects = 0
            
            for batch in dataloaders[phase]:
                rgb = batch['rgb'].to(device)
                fft = batch['fft'].to(device)
                noise = batch['noise'].to(device)
                labels = batch['label'].to(device).unsqueeze(1)
                
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(rgb, fft, noise)
                    loss = criterion(outputs, labels)
                    
                    preds = (torch.sigmoid(outputs) > 0.5).float()
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        
                running_loss += loss.item() * rgb.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), 'best_model.pth')
                
    print(f'Best val Acc: {best_acc:4f}')
    model.load_state_dict(best_model_wts)
    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=16)
    args = parser.parse_args()
    
    train_model(args.data_dir, num_epochs=args.epochs, batch_size=args.batch_size)
