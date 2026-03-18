"""
Vision Transformer (ViT) Training Script for Plant Disease Detection
=====================================================================
Uses pretrained ViT-Base model from timm with fine-tuning
Expected accuracy: 97-99% (state-of-the-art performance)

Key Features:
  - Pretrained ViT-Base-Patch16-224 backbone
  - Mixed precision training (FP16) for faster training
  - Cosine annealing with warmup learning rate schedule
  - Label smoothing for better generalization
  - Gradient accumulation for effective larger batch sizes
  - Progressive unfreezing strategy
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
import math
import json

# Try to import timm
try:
    import timm
    print("✓ timm library available")
except ImportError:
    print("❌ timm not found. Install with: pip install timm")
    exit(1)

# Try to import mixed precision training
try:
    from torch.cuda.amp import GradScaler, autocast
    AMP_AVAILABLE = torch.cuda.is_available()
except ImportError:
    AMP_AVAILABLE = False


# ============================================================
# Configuration
# ============================================================
class Config:
    # Paths
    DATASET_PATH = "New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)"
    TRAIN_DIR = os.path.join(DATASET_PATH, "train")
    VALID_DIR = os.path.join(DATASET_PATH, "valid")

    # Model save path
    MODEL_SAVE_PATH = "Flask Deployed App/plant_disease_model_vit.pt"
    CHECKPOINT_DIR = "checkpoints_vit"

    # Model configuration
    MODEL_NAME = "vit_base_patch16_224"  # ViT-Base with 16x16 patches
    NUM_CLASSES = 39
    IMG_SIZE = 224

    # Hyperparameters
    BATCH_SIZE = 16  # ViT needs more memory, smaller batch size
    NUM_EPOCHS = 15
    LEARNING_RATE = 3e-4       # Initial LR for new head
    BACKBONE_LR = 1e-5         # Lower LR for pretrained backbone
    WEIGHT_DECAY = 0.05        # AdamW weight decay
    LABEL_SMOOTHING = 0.1      # Label smoothing for better generalization
    WARMUP_EPOCHS = 2          # Linear warmup epochs
    GRAD_ACCUM_STEPS = 2       # Gradient accumulation steps (effective batch = 32)

    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Training settings
    NUM_WORKERS = 4
    PATIENCE = 5


# ============================================================
# Data Transforms (ViT-specific augmentation)
# ============================================================
def get_data_transforms():
    """
    ViT-specific data augmentation pipeline.
    Uses RandAugment-like transforms for best performance.
    """
    train_transforms = transforms.Compose([
        transforms.Resize((Config.IMG_SIZE + 32, Config.IMG_SIZE + 32)),
        transforms.RandomCrop(Config.IMG_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(degrees=25),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.85, 1.15)),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
        transforms.RandomGrayscale(p=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),  # Cutout-like augmentation
    ])

    valid_transforms = transforms.Compose([
        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return train_transforms, valid_transforms


# ============================================================
# Vision Transformer Model
# ============================================================
def create_vit_model(num_classes=39, pretrained=True):
    """
    Create Vision Transformer model with custom classification head.
    
    Architecture:
      - ViT-Base backbone (86M params) with pretrained ImageNet weights
      - Custom classification head with LayerNorm, Dropout, GELU
      - Progressive unfreezing: only head + last 2 transformer blocks trainable initially
    """
    print(f"\n{'='*60}")
    print(f"Creating Vision Transformer: {Config.MODEL_NAME}")
    print(f"{'='*60}")

    # Load pretrained ViT
    model = timm.create_model(
        Config.MODEL_NAME,
        pretrained=pretrained,
        num_classes=num_classes
    )

    # Freeze all parameters first
    for param in model.parameters():
        param.requires_grad = False

    # Replace classification head with custom one
    in_features = model.head.in_features
    model.head = nn.Sequential(
        nn.LayerNorm(in_features),
        nn.Dropout(0.3),
        nn.Linear(in_features, 512),
        nn.GELU(),
        nn.Dropout(0.2),
        nn.Linear(512, num_classes)
    )

    # Unfreeze the classification head (always trainable)
    for param in model.head.parameters():
        param.requires_grad = True

    # Unfreeze the last 2 transformer blocks for fine-tuning
    for block in model.blocks[-2:]:
        for param in block.parameters():
            param.requires_grad = True

    # Unfreeze the norm layer before the head
    if hasattr(model, 'norm'):
        for param in model.norm.parameters():
            param.requires_grad = True

    return model


def unfreeze_all_layers(model):
    """Unfreeze all layers for full fine-tuning (used after initial training)"""
    for param in model.parameters():
        param.requires_grad = True
    print("✓ All layers unfrozen for full fine-tuning")


# ============================================================
# Learning Rate Scheduler with Warmup
# ============================================================
class CosineWarmupScheduler:
    """Cosine annealing LR schedule with linear warmup."""

    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]

    def step(self, epoch):
        if epoch < self.warmup_epochs:
            # Linear warmup
            alpha = epoch / max(1, self.warmup_epochs)
            for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                param_group['lr'] = base_lr * alpha
        else:
            # Cosine annealing
            progress = (epoch - self.warmup_epochs) / max(1, self.total_epochs - self.warmup_epochs)
            for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                param_group['lr'] = self.min_lr + (base_lr - self.min_lr) * 0.5 * (
                    1 + math.cos(math.pi * progress)
                )

    def get_last_lr(self):
        return [group['lr'] for group in self.optimizer.param_groups]


# ============================================================
# Dataset Loading
# ============================================================
def load_datasets():
    """Load training and validation datasets"""
    train_transforms, valid_transforms = get_data_transforms()

    print("\nLoading datasets...")
    print(f"  Train directory: {Config.TRAIN_DIR}")
    print(f"  Valid directory: {Config.VALID_DIR}")

    train_dataset = datasets.ImageFolder(Config.TRAIN_DIR, transform=train_transforms)
    valid_dataset = datasets.ImageFolder(Config.VALID_DIR, transform=valid_transforms)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if Config.DEVICE.type == 'cuda' else False,
        drop_last=True
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if Config.DEVICE.type == 'cuda' else False
    )

    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(valid_dataset)}")
    print(f"  Number of classes: {len(train_dataset.classes)}")
    print(f"  Batch size: {Config.BATCH_SIZE} (effective: {Config.BATCH_SIZE * Config.GRAD_ACCUM_STEPS})")

    return train_loader, valid_loader, train_dataset.classes


# ============================================================
# Training Loop
# ============================================================
def train_epoch(model, train_loader, criterion, optimizer, device, scaler=None):
    """Train for one epoch with gradient accumulation and optional AMP."""
    model.train()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0

    optimizer.zero_grad()

    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Mixed precision training
        if scaler is not None and AMP_AVAILABLE:
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss = loss / Config.GRAD_ACCUM_STEPS

            scaler.scale(loss).backward()

            if (batch_idx + 1) % Config.GRAD_ACCUM_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss = loss / Config.GRAD_ACCUM_STEPS
            loss.backward()

            if (batch_idx + 1) % Config.GRAD_ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * Config.GRAD_ACCUM_STEPS * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        total_samples += inputs.size(0)

        # Print progress every 100 batches
        if (batch_idx + 1) % 100 == 0:
            batch_acc = torch.sum(preds == labels.data).double() / inputs.size(0)
            current_lr = optimizer.param_groups[0]['lr']
            print(f'  Batch [{batch_idx + 1}/{len(train_loader)}] '
                  f'Loss: {loss.item() * Config.GRAD_ACCUM_STEPS:.4f} '
                  f'Acc: {batch_acc:.4f} '
                  f'LR: {current_lr:.2e}')

    epoch_loss = running_loss / total_samples
    epoch_acc = running_corrects.double() / total_samples

    return epoch_loss, epoch_acc


def validate_epoch(model, valid_loader, criterion, device):
    """Validate the model."""
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


def train_model(model, train_loader, valid_loader, criterion, optimizer, scheduler, num_epochs, scaler=None):
    """Complete training loop with early stopping and checkpointing."""
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_epoch = 0

    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)

    history = {
        'train_loss': [],
        'train_acc': [],
        'valid_loss': [],
        'valid_acc': [],
        'learning_rates': []
    }

    patience_counter = 0

    print(f"\n{'='*60}")
    print(f"Training ViT on device: {Config.DEVICE}")
    print(f"Mixed precision: {'Enabled' if scaler else 'Disabled'}")
    print(f"Gradient accumulation: {Config.GRAD_ACCUM_STEPS} steps")
    print(f"{'='*60}")

    for epoch in range(num_epochs):
        # Update learning rate
        scheduler.step(epoch)
        current_lrs = scheduler.get_last_lr()

        print(f'\nEpoch {epoch + 1}/{num_epochs}')
        print('-' * 60)
        print(f'  Learning rates: Head={current_lrs[0]:.2e}, Backbone={current_lrs[-1]:.2e}')

        # Progressive unfreezing: unfreeze all layers after epoch 5
        if epoch == 5:
            unfreeze_all_layers(model)
            # Update optimizer to include all parameters
            optimizer = optim.AdamW([
                {'params': model.head.parameters(), 'lr': Config.LEARNING_RATE * 0.5},
                {'params': [p for n, p in model.named_parameters()
                           if 'head' not in n and p.requires_grad], 'lr': Config.BACKBONE_LR}
            ], weight_decay=Config.WEIGHT_DECAY)
            scheduler = CosineWarmupScheduler(optimizer, 0, num_epochs - epoch)
            print("  → Switched to full fine-tuning mode")

        # Training phase
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, Config.DEVICE, scaler
        )

        # Validation phase
        valid_loss, valid_acc = validate_epoch(model, valid_loader, criterion, Config.DEVICE)

        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc.item())
        history['valid_loss'].append(valid_loss)
        history['valid_acc'].append(valid_acc.item())
        history['learning_rates'].append(current_lrs[0])

        # Print results
        print(f'\n  Epoch {epoch + 1} Results:')
        print(f'    Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}')
        print(f'    Valid Loss: {valid_loss:.4f} | Valid Acc: {valid_acc:.4f}')

        # Save checkpoint
        checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, f'vit_epoch_{epoch + 1}.pt')
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'valid_loss': valid_loss,
            'train_acc': train_acc.item(),
            'valid_acc': valid_acc.item(),
        }, checkpoint_path)

        # Best model tracking
        if valid_acc > best_acc:
            best_acc = valid_acc
            best_epoch = epoch + 1
            best_model_wts = copy.deepcopy(model.state_dict())
            patience_counter = 0
            print(f'    ✓ New best model! Validation Acc: {valid_acc:.4f}')
        else:
            patience_counter += 1
            print(f'    No improvement. Patience: {patience_counter}/{Config.PATIENCE}')

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


# ============================================================
# Visualization
# ============================================================
def plot_training_history(history):
    """Plot training and validation metrics."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    # Loss plot
    axes[0].plot(history['train_loss'], label='Train Loss', marker='o', linewidth=2)
    axes[0].plot(history['valid_loss'], label='Valid Loss', marker='s', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('ViT - Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)

    # Accuracy plot
    axes[1].plot(history['train_acc'], label='Train Accuracy', marker='o', linewidth=2, color='green')
    axes[1].plot(history['valid_acc'], label='Valid Accuracy', marker='s', linewidth=2, color='orange')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_title('ViT - Training and Validation Accuracy', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)

    # Learning rate plot
    axes[2].plot(history['learning_rates'], marker='o', linewidth=2, color='red')
    axes[2].set_xlabel('Epoch', fontsize=12)
    axes[2].set_ylabel('Learning Rate', fontsize=12)
    axes[2].set_title('ViT - Learning Rate Schedule', fontsize=14, fontweight='bold')
    axes[2].set_yscale('log')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('training_history_vit.png', dpi=300, bbox_inches='tight')
    print("\n✓ Training history plot saved as 'training_history_vit.png'")
    plt.show()


# ============================================================
# Main
# ============================================================
def main():
    """Main training function."""
    print("=" * 60)
    print("🌿 Plant Disease Detection - Vision Transformer (ViT)")
    print("   Model: ViT-Base-Patch16-224 (86M parameters)")
    print("   Expected Accuracy: 97-99%")
    print("=" * 60)

    # Check dataset
    if not os.path.exists(Config.TRAIN_DIR):
        print(f"\n❌ Error: Training directory not found at {Config.TRAIN_DIR}")
        print("\nPlease run: python download_dataset.py")
        return

    # Device info
    print(f"\n📱 Device: {Config.DEVICE}")
    if Config.DEVICE.type == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
    else:
        print("   ⚠️  Training on CPU - this will be SLOW!")
        print("   💡 Tip: Use Google Colab or a GPU machine for faster training.")

    # Load datasets
    train_loader, valid_loader, class_names = load_datasets()

    # Save class names for inference
    class_names_path = os.path.join(os.path.dirname(Config.MODEL_SAVE_PATH), 'vit_class_names.json')
    with open(class_names_path, 'w') as f:
        json.dump(class_names, f)
    print(f"\n✓ Class names saved to: {class_names_path}")

    # Create model
    model = create_vit_model(Config.NUM_CLASSES, pretrained=True)
    model = model.to(Config.DEVICE)

    # Model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    print(f"\n📊 Model Summary:")
    print(f"   Architecture: {Config.MODEL_NAME}")
    print(f"   Total parameters:     {total_params:>12,}")
    print(f"   Trainable parameters: {trainable_params:>12,}")
    print(f"   Frozen parameters:    {frozen_params:>12,}")
    print(f"   Training strategy:    Progressive Unfreezing")

    # Loss function with label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=Config.LABEL_SMOOTHING)

    # Optimizer: AdamW with different LRs for head and backbone
    optimizer = optim.AdamW([
        {'params': model.head.parameters(), 'lr': Config.LEARNING_RATE},
        {'params': [p for n, p in model.named_parameters()
                   if 'head' not in n and p.requires_grad], 'lr': Config.BACKBONE_LR}
    ], weight_decay=Config.WEIGHT_DECAY)

    # Learning rate scheduler
    scheduler = CosineWarmupScheduler(
        optimizer,
        warmup_epochs=Config.WARMUP_EPOCHS,
        total_epochs=Config.NUM_EPOCHS
    )

    # Mixed precision scaler
    scaler = GradScaler() if AMP_AVAILABLE else None

    # Train
    print(f"\n🚀 Starting ViT training for {Config.NUM_EPOCHS} epochs...")
    print(f"   Label smoothing: {Config.LABEL_SMOOTHING}")
    print(f"   Weight decay: {Config.WEIGHT_DECAY}")
    print(f"   Warmup epochs: {Config.WARMUP_EPOCHS}")

    model, history = train_model(
        model, train_loader, valid_loader, criterion, optimizer, scheduler,
        Config.NUM_EPOCHS, scaler
    )

    # Save final model
    print(f"\n💾 Saving ViT model to {Config.MODEL_SAVE_PATH}...")
    torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
    print("✓ Model saved successfully!")

    # Save training history
    history_path = 'training_history_vit.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"✓ Training history saved to {history_path}")

    # Plot results
    plot_training_history(history)

    # Summary
    print(f"\n{'='*60}")
    print("✅ Vision Transformer Training Complete!")
    print(f"{'='*60}")
    print(f"   Best Accuracy: {max(history['valid_acc']):.4f}")
    print(f"   Model saved:   {Config.MODEL_SAVE_PATH}")
    print()
    print("📝 Next Steps:")
    print("   1. Compare ViT vs CNN vs ResNet50 performance")
    print("   2. Update Flask app to use ViT model")
    print("   3. Run: python compare_models.py")
    print("   4. Test: python quick_test.py --model vit")


if __name__ == "__main__":
    main()
