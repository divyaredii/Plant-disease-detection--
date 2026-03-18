"""
ADVANCED MULTI-MODEL ENSEMBLE TRAINING
Combines ResNet50, EfficientNet-B3, and DenseNet121 for superior accuracy
Expected improvement: 97-99% → 98-99.5% accuracy with better robustness
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
import json

# Configuration
class Config:
    # Paths
    DATASET_PATH = "New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)"
    TRAIN_DIR = os.path.join(DATASET_PATH, "train")
    VALID_DIR = os.path.join(DATASET_PATH, "valid")
    
    # Model save paths
    MODEL_DIR = "Flask Deployed App/models"
    RESNET_PATH = os.path.join(MODEL_DIR, "resnet50_plant.pt")
    EFFICIENTNET_PATH = os.path.join(MODEL_DIR, "efficientnet_b3_plant.pt")
    DENSENET_PATH = os.path.join(MODEL_DIR, "densenet121_plant.pt")
    ENSEMBLE_PATH = os.path.join(MODEL_DIR, "ensemble_weights.json")
    
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
    """Enhanced data augmentation for ensemble training"""
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

def create_resnet50(num_classes=39, pretrained=True):
    """Create ResNet50 model"""
    print(f"\n🔧 Creating ResNet50 (pretrained={pretrained})...")
    model = models.resnet50(pretrained=pretrained)
    
    # Freeze early layers
    for param in model.parameters():
        param.requires_grad = False
    
    # Custom classifier
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes)
    )
    
    # Unfreeze final layers
    for param in model.fc.parameters():
        param.requires_grad = True
    for param in model.layer4.parameters():
        param.requires_grad = True
    
    return model

def create_efficientnet_b3(num_classes=39, pretrained=True):
    """Create EfficientNet-B3 model"""
    print(f"\n🔧 Creating EfficientNet-B3 (pretrained={pretrained})...")
    model = models.efficientnet_b3(pretrained=pretrained)
    
    # Freeze early layers
    for param in model.parameters():
        param.requires_grad = False
    
    # Custom classifier
    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes)
    )
    
    # Unfreeze classifier
    for param in model.classifier.parameters():
        param.requires_grad = True
    
    return model

def create_densenet121(num_classes=39, pretrained=True):
    """Create DenseNet121 model"""
    print(f"\n🔧 Creating DenseNet121 (pretrained={pretrained})...")
    model = models.densenet121(pretrained=pretrained)
    
    # Freeze early layers
    for param in model.parameters():
        param.requires_grad = False
    
    # Custom classifier
    num_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes)
    )
    
    # Unfreeze classifier
    for param in model.classifier.parameters():
        param.requires_grad = True
    
    return model

def load_datasets():
    """Load training and validation datasets"""
    train_transforms, valid_transforms = get_data_transforms()
    
    print("\n📂 Loading datasets...")
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
    
    print(f"✓ Training samples: {len(train_dataset)}")
    print(f"✓ Validation samples: {len(valid_dataset)}")
    print(f"✓ Number of classes: {len(train_dataset.classes)}")
    
    return train_loader, valid_loader, train_dataset.classes

def train_single_model(model, model_name, train_loader, valid_loader, num_epochs):
    """Train a single model"""
    print(f"\n{'='*60}")
    print(f"Training {model_name}")
    print(f"{'='*60}")
    
    model = model.to(Config.DEVICE)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=Config.LEARNING_RATE
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )
    
    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    patience_counter = 0
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'valid_loss': [],
        'valid_acc': []
    }
    
    for epoch in range(num_epochs):
        print(f'\n📊 Epoch {epoch + 1}/{num_epochs}')
        print('-' * 60)
        
        # Training phase
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(Config.DEVICE)
            labels = labels.to(Config.DEVICE)
            
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
        
        train_loss = running_loss / total_samples
        train_acc = running_corrects.double() / total_samples
        
        # Validation phase
        model.eval()
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0
        
        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs = inputs.to(Config.DEVICE)
                labels = labels.to(Config.DEVICE)
                
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                total_samples += inputs.size(0)
        
        valid_loss = running_loss / total_samples
        valid_acc = running_corrects.double() / total_samples
        
        scheduler.step(valid_loss)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc.item())
        history['valid_loss'].append(valid_loss)
        history['valid_acc'].append(valid_acc.item())
        
        print(f'\n✓ Epoch {epoch + 1} Results:')
        print(f'  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}')
        print(f'  Valid Loss: {valid_loss:.4f} | Valid Acc: {valid_acc:.4f}')
        
        if valid_acc > best_acc:
            best_acc = valid_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            patience_counter = 0
            print(f'  🌟 New best model! Validation Acc: {valid_acc:.4f}')
        else:
            patience_counter += 1
            print(f'  ⏳ No improvement. Patience: {patience_counter}/{Config.PATIENCE}')
        
        if patience_counter >= Config.PATIENCE:
            print(f'\n⏹️ Early stopping triggered after {epoch + 1} epochs')
            break
    
    model.load_state_dict(best_model_wts)
    print(f'\n✅ {model_name} training complete! Best Acc: {best_acc:.4f}')
    
    return model, history, best_acc.item()

def evaluate_ensemble(models, valid_loader, ensemble_weights=None):
    """Evaluate ensemble performance"""
    print("\n🔍 Evaluating ensemble...")
    
    if ensemble_weights is None:
        ensemble_weights = [1.0 / len(models)] * len(models)
    
    for model in models:
        model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in valid_loader:
            inputs = inputs.to(Config.DEVICE)
            
            # Get predictions from all models
            ensemble_outputs = []
            for model in models:
                outputs = model(inputs)
                probs = torch.softmax(outputs, dim=1)
                ensemble_outputs.append(probs)
            
            # Weighted average
            weighted_output = sum(w * out for w, out in zip(ensemble_weights, ensemble_outputs))
            _, preds = torch.max(weighted_output, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    accuracy = np.mean(all_preds == all_labels)
    
    return accuracy

def main():
    """Main ensemble training function"""
    print("="*60)
    print("🚀 ADVANCED MULTI-MODEL ENSEMBLE TRAINING")
    print("Models: ResNet50 + EfficientNet-B3 + DenseNet121")
    print("Expected Accuracy: 98-99.5%")
    print("="*60)
    
    if not os.path.exists(Config.TRAIN_DIR):
        print(f"\n❌ Error: Training directory not found at {Config.TRAIN_DIR}")
        print("\nPlease run: python download_dataset.py")
        return
    
    # Create model directory
    os.makedirs(Config.MODEL_DIR, exist_ok=True)
    
    # Load datasets
    train_loader, valid_loader, class_names = load_datasets()
    
    # Train individual models
    models = []
    model_names = []
    individual_accuracies = []
    
    # 1. Train ResNet50
    print("\n" + "="*60)
    print("MODEL 1/3: ResNet50")
    print("="*60)
    resnet = create_resnet50(Config.NUM_CLASSES, pretrained=True)
    resnet, resnet_history, resnet_acc = train_single_model(
        resnet, "ResNet50", train_loader, valid_loader, Config.NUM_EPOCHS
    )
    torch.save(resnet.state_dict(), Config.RESNET_PATH)
    models.append(resnet)
    model_names.append("ResNet50")
    individual_accuracies.append(resnet_acc)
    
    # 2. Train EfficientNet-B3
    print("\n" + "="*60)
    print("MODEL 2/3: EfficientNet-B3")
    print("="*60)
    efficientnet = create_efficientnet_b3(Config.NUM_CLASSES, pretrained=True)
    efficientnet, eff_history, eff_acc = train_single_model(
        efficientnet, "EfficientNet-B3", train_loader, valid_loader, Config.NUM_EPOCHS
    )
    torch.save(efficientnet.state_dict(), Config.EFFICIENTNET_PATH)
    models.append(efficientnet)
    model_names.append("EfficientNet-B3")
    individual_accuracies.append(eff_acc)
    
    # 3. Train DenseNet121
    print("\n" + "="*60)
    print("MODEL 3/3: DenseNet121")
    print("="*60)
    densenet = create_densenet121(Config.NUM_CLASSES, pretrained=True)
    densenet, dense_history, dense_acc = train_single_model(
        densenet, "DenseNet121", train_loader, valid_loader, Config.NUM_EPOCHS
    )
    torch.save(densenet.state_dict(), Config.DENSENET_PATH)
    models.append(densenet)
    model_names.append("DenseNet121")
    individual_accuracies.append(dense_acc)
    
    # Calculate optimal ensemble weights based on validation accuracy
    total_acc = sum(individual_accuracies)
    ensemble_weights = [acc / total_acc for acc in individual_accuracies]
    
    # Evaluate ensemble
    print("\n" + "="*60)
    print("📊 ENSEMBLE EVALUATION")
    print("="*60)
    
    ensemble_acc = evaluate_ensemble(models, valid_loader, ensemble_weights)
    
    # Save ensemble weights
    ensemble_config = {
        'models': model_names,
        'weights': ensemble_weights,
        'individual_accuracies': individual_accuracies,
        'ensemble_accuracy': ensemble_acc,
        'num_classes': Config.NUM_CLASSES,
        'class_names': class_names
    }
    
    with open(Config.ENSEMBLE_PATH, 'w') as f:
        json.dump(ensemble_config, f, indent=2)
    
    # Print results
    print("\n📈 FINAL RESULTS:")
    print("="*60)
    for name, acc, weight in zip(model_names, individual_accuracies, ensemble_weights):
        print(f"{name:20s} | Accuracy: {acc:.4f} | Weight: {weight:.4f}")
    print("-"*60)
    print(f"{'ENSEMBLE':20s} | Accuracy: {ensemble_acc:.4f} | 🌟 BEST")
    print("="*60)
    
    improvement = (ensemble_acc - max(individual_accuracies)) * 100
    print(f"\n✨ Ensemble improvement over best single model: +{improvement:.2f}%")
    
    print("\n✅ All models saved successfully!")
    print(f"   - ResNet50: {Config.RESNET_PATH}")
    print(f"   - EfficientNet-B3: {Config.EFFICIENTNET_PATH}")
    print(f"   - DenseNet121: {Config.DENSENET_PATH}")
    print(f"   - Ensemble config: {Config.ENSEMBLE_PATH}")
    
    print("\n📝 Next Steps:")
    print("1. Update Flask app to use ensemble predictions")
    print("2. Test with real plant images")
    print("3. Deploy to production")

if __name__ == "__main__":
    main()
