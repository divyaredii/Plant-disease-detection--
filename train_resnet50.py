"""
PHASE 1 IMPROVEMENT: Transfer Learning with ResNet50
This script implements the highest-priority improvement for better accuracy
Expected improvement: 92-94% → 97-99% accuracy
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import numpy as np
import matplotlib.pyplot as plt
import time
import copy
import os
from pathlib import Path
import torch.nn.functional as F

# Configuration
class Config:
    # Paths
    DATASET_PATH = "New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)"
    TRAIN_DIR = os.path.join(DATASET_PATH, "train")
    VALID_DIR = os.path.join(DATASET_PATH, "valid")
    
    # Model save path
    MODEL_SAVE_PATH = "Flask Deployed App/plant_disease_model_resnet50.pt"
    CHECKPOINT_DIR = "checkpoints_resnet"
    
    # Hyperparameters
    BATCH_SIZE = 32
    NUM_EPOCHS = 20
    LEARNING_RATE = 0.001
    NUM_CLASSES = 39
    IMG_SIZE = 224
    
    # Device configuration
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Training settings
    NUM_WORKERS = 4
    PATIENCE = 5

def get_data_transforms():
    """Enhanced data augmentation"""
    train_transforms = transforms.Compose([
        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(degrees=30),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.8, 1.2)),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    valid_transforms = transforms.Compose([
        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transforms, valid_transforms

def create_resnet_model(num_classes=39, pretrained=True):
    """
    Create ResNet50 model with transfer learning
    """
    print(f"\nCreating ResNet50 model (pretrained={pretrained})...")
    
    # Load pre-trained ResNet50
    model = models.resnet50(pretrained=pretrained)
    
    # Freeze early layers (optional - comment out for full fine-tuning)
    for param in model.parameters():
        param.requires_grad = False
    
    # Replace the final fully connected layer
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes)
    )
    
    # Unfreeze the final layers for training
    for param in model.fc.parameters():
        param.requires_grad = True
    
    # Optionally unfreeze layer4 for better fine-tuning
    for param in model.layer4.parameters():
        param.requires_grad = True
    
    return model

def load_datasets():
    """Load training and validation datasets"""
    train_transforms, valid_transforms = get_data_transforms()
    
    print("Loading datasets...")
    print(f"Train directory: {Config.TRAIN_DIR}")
    print(f"Valid directory: {Config.VALID_DIR}")
    
    train_dataset = datasets.ImageFolder(Config.TRAIN_DIR, transform=train_transforms)
    valid_dataset = datasets.ImageFolder(Config.VALID_DIR, transform=valid_transforms)
    
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
    
    return train_loader, valid_loader, train_dataset.classes

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0
    
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        total_samples += inputs.size(0)
        
        if (batch_idx + 1) % 50 == 0:
            batch_acc = torch.sum(preds == labels.data).double() / inputs.size(0)
            print(f'  Batch [{batch_idx + 1}/{len(train_loader)}] '
                  f'Loss: {loss.item():.4f} Acc: {batch_acc:.4f}')
    
    epoch_loss = running_loss / total_samples
    epoch_acc = running_corrects.double() / total_samples
    
    return epoch_loss, epoch_acc

def validate_epoch(model, valid_loader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0
    
    with torch.no_grad():
        for inputs, labels in valid_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            total_samples += inputs.size(0)
    
    epoch_loss = running_loss / total_samples
    epoch_acc = running_corrects.double() / total_samples
    
    return epoch_loss, epoch_acc

def train_model(model, train_loader, valid_loader, criterion, optimizer, scheduler, num_epochs):
    """Complete training loop"""
    since = time.time()
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_epoch = 0
    
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'valid_loss': [],
        'valid_acc': []
    }
    
    patience_counter = 0
    
    print(f"\nTraining ResNet50 on device: {Config.DEVICE}")
    print(f"{'='*60}")
    
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch + 1}/{num_epochs}')
        print('-' * 60)
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, Config.DEVICE)
        valid_loss, valid_acc = validate_epoch(model, valid_loader, criterion, Config.DEVICE)
        
        scheduler.step(valid_loss)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc.item())
        history['valid_loss'].append(valid_loss)
        history['valid_acc'].append(valid_acc.item())
        
        print(f'\nEpoch {epoch + 1} Results:')
        print(f'  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}')
        print(f'  Valid Loss: {valid_loss:.4f} | Valid Acc: {valid_acc:.4f}')
        print(f'  Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
        
        checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, f'resnet_epoch_{epoch + 1}.pt')
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'valid_loss': valid_loss,
            'train_acc': train_acc,
            'valid_acc': valid_acc,
        }, checkpoint_path)
        
        if valid_acc > best_acc:
            best_acc = valid_acc
            best_epoch = epoch + 1
            best_model_wts = copy.deepcopy(model.state_dict())
            patience_counter = 0
            print(f'  ✓ New best model! Validation Acc: {valid_acc:.4f}')
        else:
            patience_counter += 1
            print(f'  No improvement. Patience: {patience_counter}/{Config.PATIENCE}')
        
        if patience_counter >= Config.PATIENCE:
            print(f'\nEarly stopping triggered after {epoch + 1} epochs')
            break
    
    time_elapsed = time.time() - since
    print(f'\n{"="*60}')
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best Validation Acc: {best_acc:.4f} at epoch {best_epoch}')
    
    model.load_state_dict(best_model_wts)
    
    return model, history

def plot_training_history(history):
    """Plot training and validation metrics"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    axes[0].plot(history['train_loss'], label='Train Loss', marker='o')
    axes[0].plot(history['valid_loss'], label='Valid Loss', marker='s')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('ResNet50 - Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    axes[1].plot(history['train_acc'], label='Train Accuracy', marker='o')
    axes[1].plot(history['valid_acc'], label='Valid Accuracy', marker='s')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('ResNet50 - Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history_resnet50.png', dpi=300, bbox_inches='tight')
    print("\nTraining history plot saved as 'training_history_resnet50.png'")
    plt.show()

def main():
    """Main training function"""
    print("="*60)
    print("Plant Disease Detection - ResNet50 Transfer Learning")
    print("PHASE 1 IMPROVEMENT - Expected Accuracy: 97-99%")
    print("="*60)
    
    if not os.path.exists(Config.TRAIN_DIR):
        print(f"\n❌ Error: Training directory not found at {Config.TRAIN_DIR}")
        print("\nPlease run: python download_dataset.py")
        return
    
    # Load datasets
    train_loader, valid_loader, class_names = load_datasets()
    
    # Initialize ResNet50 model
    model = create_resnet_model(Config.NUM_CLASSES, pretrained=True)
    model = model.to(Config.DEVICE)
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: ResNet50")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Frozen parameters: {total_params - trainable_params:,}")
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    
    # Only optimize trainable parameters
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=Config.LEARNING_RATE
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )
    
    # Train the model
    print(f"\nStarting ResNet50 training for {Config.NUM_EPOCHS} epochs...")
    model, history = train_model(
        model, train_loader, valid_loader, criterion, optimizer, scheduler, Config.NUM_EPOCHS
    )
    
    # Save the final model
    print(f"\nSaving ResNet50 model to {Config.MODEL_SAVE_PATH}...")
    torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
    print("✓ Model saved successfully!")
    
    # Plot training history
    plot_training_history(history)
    
    print("\n" + "="*60)
    print("✓ ResNet50 Training completed successfully!")
    print("Expected accuracy improvement: 92-94% → 97-99%")
    print("="*60)
    
    print("\n📝 Next Steps:")
    print("1. Compare this model with the original CNN")
    print("2. Update Flask app to use the new model")
    print("3. Test with real plant images")
    print("4. Implement Phase 2 improvements (see IMPROVEMENTS_ROADMAP.md)")

if __name__ == "__main__":
    main()
