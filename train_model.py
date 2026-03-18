"""
Plant Disease Detection Model Training Script
Dataset: New Plant Diseases Dataset from Kaggle
Author: Enhanced Training Pipeline
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import time
import copy
import os
from pathlib import Path
import sys

# Add Flask Deployed App to path to import CNN
sys.path.append(str(Path(__file__).parent / "Flask Deployed App"))
from CNN import CNN

# Configuration
class Config:
    # Paths - UPDATE THESE PATHS ACCORDING TO YOUR DATASET LOCATION
    DATASET_PATH = "New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)"
    TRAIN_DIR = os.path.join(DATASET_PATH, "train")
    VALID_DIR = os.path.join(DATASET_PATH, "valid")
    
    # Model save path
    MODEL_SAVE_PATH = "Flask Deployed App/plant_disease_model_improved.pt"
    CHECKPOINT_DIR = "checkpoints"
    
    # Hyperparameters
    BATCH_SIZE = 32
    NUM_EPOCHS = 25
    LEARNING_RATE = 0.001
    NUM_CLASSES = 39
    IMG_SIZE = 224
    
    # Device configuration
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Training settings
    NUM_WORKERS = 4
    PATIENCE = 5  # Early stopping patience
    
def get_data_transforms():
    """
    Define data augmentation and normalization for training and validation
    """
    # Training data augmentation
    train_transforms = transforms.Compose([
        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(degrees=20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Validation data - only resize and normalize
    valid_transforms = transforms.Compose([
        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transforms, valid_transforms

def load_datasets():
    """
    Load training and validation datasets
    """
    train_transforms, valid_transforms = get_data_transforms()
    
    print("Loading datasets...")
    print(f"Train directory: {Config.TRAIN_DIR}")
    print(f"Valid directory: {Config.VALID_DIR}")
    
    # Load datasets
    train_dataset = datasets.ImageFolder(Config.TRAIN_DIR, transform=train_transforms)
    valid_dataset = datasets.ImageFolder(Config.VALID_DIR, transform=valid_transforms)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if Config.DEVICE.type == 'cuda' else False
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if Config.DEVICE.type == 'cuda' else False
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(valid_dataset)}")
    print(f"Number of classes: {len(train_dataset.classes)}")
    print(f"Classes: {train_dataset.classes}")
    
    return train_loader, valid_loader, train_dataset.classes

def train_epoch(model, train_loader, criterion, optimizer, device):
    """
    Train for one epoch
    """
    model.train()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0
    
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        total_samples += inputs.size(0)
        
        # Print progress every 50 batches
        if (batch_idx + 1) % 50 == 0:
            batch_acc = torch.sum(preds == labels.data).double() / inputs.size(0)
            print(f'  Batch [{batch_idx + 1}/{len(train_loader)}] '
                  f'Loss: {loss.item():.4f} Acc: {batch_acc:.4f}')
    
    epoch_loss = running_loss / total_samples
    epoch_acc = running_corrects.double() / total_samples
    
    return epoch_loss, epoch_acc

def validate_epoch(model, valid_loader, criterion, device):
    """
    Validate the model
    """
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0
    
    with torch.no_grad():
        for inputs, labels in valid_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            total_samples += inputs.size(0)
    
    epoch_loss = running_loss / total_samples
    epoch_acc = running_corrects.double() / total_samples
    
    return epoch_loss, epoch_acc

def train_model(model, train_loader, valid_loader, criterion, optimizer, scheduler, num_epochs):
    """
    Complete training loop with early stopping and checkpointing
    """
    since = time.time()
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_epoch = 0
    
    # Create checkpoint directory
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'valid_loss': [],
        'valid_acc': []
    }
    
    # Early stopping
    patience_counter = 0
    
    print(f"\nTraining on device: {Config.DEVICE}")
    print(f"{'='*60}")
    
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch + 1}/{num_epochs}')
        print('-' * 60)
        
        # Training phase
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, Config.DEVICE)
        
        # Validation phase
        valid_loss, valid_acc = validate_epoch(model, valid_loader, criterion, Config.DEVICE)
        
        # Learning rate scheduling
        scheduler.step(valid_loss)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc.item())
        history['valid_loss'].append(valid_loss)
        history['valid_acc'].append(valid_acc.item())
        
        # Print epoch results
        print(f'\nEpoch {epoch + 1} Results:')
        print(f'  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}')
        print(f'  Valid Loss: {valid_loss:.4f} | Valid Acc: {valid_acc:.4f}')
        print(f'  Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
        
        # Save checkpoint
        checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, f'checkpoint_epoch_{epoch + 1}.pt')
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'valid_loss': valid_loss,
            'train_acc': train_acc,
            'valid_acc': valid_acc,
        }, checkpoint_path)
        
        # Deep copy the model if it's the best so far
        if valid_acc > best_acc:
            best_acc = valid_acc
            best_epoch = epoch + 1
            best_model_wts = copy.deepcopy(model.state_dict())
            patience_counter = 0
            print(f'  ✓ New best model! Validation Acc: {valid_acc:.4f}')
        else:
            patience_counter += 1
            print(f'  No improvement. Patience: {patience_counter}/{Config.PATIENCE}')
        
        # Early stopping
        if patience_counter >= Config.PATIENCE:
            print(f'\nEarly stopping triggered after {epoch + 1} epochs')
            break
    
    time_elapsed = time.time() - since
    print(f'\n{"="*60}')
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best Validation Acc: {best_acc:.4f} at epoch {best_epoch}')
    
    # Load best model weights
    model.load_state_dict(best_model_wts)
    
    return model, history

def plot_training_history(history):
    """
    Plot training and validation metrics
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot loss
    axes[0].plot(history['train_loss'], label='Train Loss', marker='o')
    axes[0].plot(history['valid_loss'], label='Valid Loss', marker='s')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot accuracy
    axes[1].plot(history['train_acc'], label='Train Accuracy', marker='o')
    axes[1].plot(history['valid_acc'], label='Valid Accuracy', marker='s')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    print("\nTraining history plot saved as 'training_history.png'")
    plt.show()

def main():
    """
    Main training function
    """
    print("="*60)
    print("Plant Disease Detection - Model Training")
    print("="*60)
    
    # Check if dataset exists
    if not os.path.exists(Config.TRAIN_DIR):
        print(f"\n❌ Error: Training directory not found at {Config.TRAIN_DIR}")
        print("\nPlease download the dataset from:")
        print("https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset")
        print("\nInstructions:")
        print("1. Download the dataset from Kaggle")
        print("2. Extract it to the project directory")
        print("3. Update the DATASET_PATH in Config class if needed")
        return
    
    # Load datasets
    train_loader, valid_loader, class_names = load_datasets()
    
    # Initialize model
    print(f"\nInitializing CNN model with {Config.NUM_CLASSES} classes...")
    model = CNN(Config.NUM_CLASSES)
    model = model.to(Config.DEVICE)
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )
    
    # Train the model
    print(f"\nStarting training for {Config.NUM_EPOCHS} epochs...")
    model, history = train_model(
        model, train_loader, valid_loader, criterion, optimizer, scheduler, Config.NUM_EPOCHS
    )
    
    # Save the final model
    print(f"\nSaving model to {Config.MODEL_SAVE_PATH}...")
    torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
    print("✓ Model saved successfully!")
    
    # Plot training history
    plot_training_history(history)
    
    print("\n" + "="*60)
    print("Training completed successfully!")
    print("="*60)

if __name__ == "__main__":
    main()
