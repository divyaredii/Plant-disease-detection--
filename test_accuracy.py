"""
Comprehensive Model Accuracy Testing Script
Tests the current plant disease detection model and generates detailed metrics
"""

import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import time
import os
from pathlib import Path
import json

# Configuration
class Config:
    # Paths
    DATASET_PATH = "New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)"
    VALID_DIR = os.path.join(DATASET_PATH, "valid")
    MODEL_PATH = "Flask Deployed App/plant_disease_model_1_latest.pt"
    
    # Settings
    BATCH_SIZE = 32
    NUM_CLASSES = 39
    IMG_SIZE = 224
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    NUM_WORKERS = 4

# Define the model architecture (must match training - from CNN.py)
class CNN(nn.Module):
    def __init__(self, K=39):
        super(CNN, self).__init__()
        self.conv_layers = nn.Sequential(
            # conv1
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            # conv2
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            # conv3
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
            # conv4
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2),
        )

        self.dense_layers = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(50176, 1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, K),
        )

    def forward(self, X):
        out = self.conv_layers(X)
        # Flatten
        out = out.view(-1, 50176)
        # Fully connected
        out = self.dense_layers(out)
        return out

def get_transforms():
    """Get validation transforms"""
    return transforms.Compose([
        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def load_model():
    """Load the trained model"""
    print(f"\n🔄 Loading model from {Config.MODEL_PATH}...")
    
    if not os.path.exists(Config.MODEL_PATH):
        print(f"❌ Model not found at {Config.MODEL_PATH}")
        return None
    
    try:
        model = CNN(Config.NUM_CLASSES)
        model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=Config.DEVICE))
        model.to(Config.DEVICE)
        model.eval()
        print("✅ Model loaded successfully!")
        return model
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("\n💡 Trying alternative loading method...")
        try:
            # Try loading as full model
            model = torch.load(Config.MODEL_PATH, map_location=Config.DEVICE)
            if hasattr(model, 'eval'):
                model.to(Config.DEVICE)
                model.eval()
                print("✅ Model loaded successfully (alternative method)!")
                return model
            else:
                print("❌ Loaded object is not a model")
                return None
        except Exception as e2:
            print(f"❌ Alternative method also failed: {e2}")
            return None

def load_validation_data():
    """Load validation dataset"""
    print(f"\n📂 Loading validation data from {Config.VALID_DIR}...")
    
    if not os.path.exists(Config.VALID_DIR):
        print(f"❌ Validation directory not found at {Config.VALID_DIR}")
        return None, None
    
    transform = get_transforms()
    valid_dataset = datasets.ImageFolder(Config.VALID_DIR, transform=transform)
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if Config.DEVICE.type == 'cuda' else False
    )
    
    print(f"✅ Loaded {len(valid_dataset)} validation images")
    print(f"✅ Number of classes: {len(valid_dataset.classes)}")
    
    return valid_loader, valid_dataset.classes

def test_model(model, valid_loader):
    """Test model and collect predictions"""
    print("\n🧪 Testing model on validation set...")
    
    all_preds = []
    all_labels = []
    all_probs = []
    total_time = 0
    
    model.eval()
    
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(valid_loader):
            inputs = inputs.to(Config.DEVICE)
            labels = labels.to(Config.DEVICE)
            
            # Measure inference time
            start_time = time.time()
            outputs = model(inputs)
            inference_time = (time.time() - start_time) * 1000  # Convert to ms
            total_time += inference_time
            
            # Get predictions
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
            if (batch_idx + 1) % 20 == 0:
                print(f"  Processed {(batch_idx + 1) * Config.BATCH_SIZE} images...")
    
    avg_inference_time = total_time / len(valid_loader)
    
    return np.array(all_preds), np.array(all_labels), np.array(all_probs), avg_inference_time

def calculate_metrics(y_true, y_pred, class_names):
    """Calculate comprehensive metrics"""
    print("\n📊 Calculating metrics...")
    
    # Overall metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # Per-class metrics
    per_class_report = classification_report(
        y_true, y_pred, 
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'per_class': per_class_report,
        'confusion_matrix': cm
    }

def print_results(metrics, avg_inference_time, class_names):
    """Print detailed results"""
    print("\n" + "="*70)
    print(" "*20 + "🎯 MODEL ACCURACY TEST RESULTS")
    print("="*70)
    
    # Overall metrics
    print(f"\n📈 OVERALL METRICS:")
    print("-"*70)
    print(f"  Accuracy:  {metrics['accuracy']*100:.2f}%")
    print(f"  Precision: {metrics['precision']*100:.2f}%")
    print(f"  Recall:    {metrics['recall']*100:.2f}%")
    print(f"  F1-Score:  {metrics['f1_score']*100:.2f}%")
    print(f"  Avg Inference Time: {avg_inference_time:.2f}ms per batch")
    
    # Performance rating
    accuracy_pct = metrics['accuracy'] * 100
    if accuracy_pct >= 98:
        rating = "🌟 EXCELLENT"
    elif accuracy_pct >= 95:
        rating = "✅ VERY GOOD"
    elif accuracy_pct >= 90:
        rating = "👍 GOOD"
    elif accuracy_pct >= 85:
        rating = "⚠️ FAIR"
    else:
        rating = "❌ NEEDS IMPROVEMENT"
    
    print(f"\n  Performance Rating: {rating}")
    
    # Top 10 best performing classes
    print(f"\n🏆 TOP 10 BEST PERFORMING CLASSES:")
    print("-"*70)
    class_accuracies = []
    for class_name in class_names:
        if class_name in metrics['per_class']:
            f1 = metrics['per_class'][class_name]['f1-score']
            class_accuracies.append((class_name, f1))
    
    class_accuracies.sort(key=lambda x: x[1], reverse=True)
    for i, (class_name, f1) in enumerate(class_accuracies[:10], 1):
        print(f"  {i:2d}. {class_name:40s} F1: {f1*100:5.2f}%")
    
    # Bottom 10 worst performing classes
    print(f"\n⚠️ TOP 10 CLASSES NEEDING IMPROVEMENT:")
    print("-"*70)
    for i, (class_name, f1) in enumerate(class_accuracies[-10:], 1):
        print(f"  {i:2d}. {class_name:40s} F1: {f1*100:5.2f}%")
    
    print("\n" + "="*70)

def plot_confusion_matrix(cm, class_names, save_path='confusion_matrix.png'):
    """Plot confusion matrix"""
    print(f"\n📊 Generating confusion matrix...")
    
    plt.figure(figsize=(20, 18))
    
    # Normalize confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Plot
    sns.heatmap(
        cm_normalized,
        annot=False,
        fmt='.2f',
        cmap='Blues',
        xticklabels=[name.replace('___', '\n').replace('_', ' ') for name in class_names],
        yticklabels=[name.replace('___', '\n').replace('_', ' ') for name in class_names],
        cbar_kws={'label': 'Accuracy'}
    )
    
    plt.title('Confusion Matrix (Normalized)', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Confusion matrix saved to {save_path}")
    plt.close()

def plot_class_performance(metrics, class_names, save_path='class_performance.png'):
    """Plot per-class performance"""
    print(f"\n📊 Generating class performance chart...")
    
    # Extract F1 scores
    f1_scores = []
    for class_name in class_names:
        if class_name in metrics['per_class']:
            f1_scores.append(metrics['per_class'][class_name]['f1-score'] * 100)
        else:
            f1_scores.append(0)
    
    # Sort by F1 score
    sorted_indices = np.argsort(f1_scores)[::-1]
    sorted_classes = [class_names[i].replace('___', ' - ').replace('_', ' ') for i in sorted_indices]
    sorted_scores = [f1_scores[i] for i in sorted_indices]
    
    # Plot
    plt.figure(figsize=(15, 12))
    colors = ['green' if score >= 95 else 'blue' if score >= 90 else 'orange' if score >= 85 else 'red' 
              for score in sorted_scores]
    
    bars = plt.barh(range(len(sorted_classes)), sorted_scores, color=colors, alpha=0.7)
    plt.yticks(range(len(sorted_classes)), sorted_classes, fontsize=9)
    plt.xlabel('F1-Score (%)', fontsize=12, fontweight='bold')
    plt.title('Per-Class Performance (F1-Score)', fontsize=14, fontweight='bold')
    plt.axvline(x=90, color='green', linestyle='--', alpha=0.5, label='Excellent (90%+)')
    plt.axvline(x=85, color='orange', linestyle='--', alpha=0.5, label='Good (85%+)')
    plt.legend()
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Class performance chart saved to {save_path}")
    plt.close()

def save_results(metrics, avg_inference_time, class_names, save_path='test_results.json'):
    """Save results to JSON"""
    print(f"\n💾 Saving results to {save_path}...")
    
    results = {
        'overall_metrics': {
            'accuracy': float(metrics['accuracy']),
            'precision': float(metrics['precision']),
            'recall': float(metrics['recall']),
            'f1_score': float(metrics['f1_score']),
            'avg_inference_time_ms': float(avg_inference_time)
        },
        'per_class_metrics': {}
    }
    
    for class_name in class_names:
        if class_name in metrics['per_class']:
            results['per_class_metrics'][class_name] = {
                'precision': float(metrics['per_class'][class_name]['precision']),
                'recall': float(metrics['per_class'][class_name]['recall']),
                'f1_score': float(metrics['per_class'][class_name]['f1-score']),
                'support': int(metrics['per_class'][class_name]['support'])
            }
    
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✅ Results saved successfully!")

def main():
    """Main testing function"""
    print("="*70)
    print(" "*15 + "🌿 PLANT DISEASE MODEL ACCURACY TEST")
    print("="*70)
    print(f"\n💻 Device: {Config.DEVICE}")
    print(f"📦 Batch Size: {Config.BATCH_SIZE}")
    
    # Load model
    model = load_model()
    if model is None:
        print("\n❌ Cannot proceed without model. Exiting...")
        return
    
    # Load validation data
    valid_loader, class_names = load_validation_data()
    if valid_loader is None:
        print("\n❌ Cannot proceed without validation data. Exiting...")
        return
    
    # Test model
    y_pred, y_true, y_probs, avg_inference_time = test_model(model, valid_loader)
    
    # Calculate metrics
    metrics = calculate_metrics(y_true, y_pred, class_names)
    
    # Print results
    print_results(metrics, avg_inference_time, class_names)
    
    # Generate visualizations
    plot_confusion_matrix(metrics['confusion_matrix'], class_names)
    plot_class_performance(metrics, class_names)
    
    # Save results
    save_results(metrics, avg_inference_time, class_names)
    
    print("\n" + "="*70)
    print("✅ TESTING COMPLETE!")
    print("="*70)
    print("\n📁 Generated files:")
    print("  • confusion_matrix.png - Confusion matrix visualization")
    print("  • class_performance.png - Per-class performance chart")
    print("  • test_results.json - Detailed results in JSON format")
    print("\n💡 Next steps:")
    print("  • Review the confusion matrix to see misclassifications")
    print("  • Check class_performance.png for classes needing improvement")
    print("  • If accuracy < 95%, consider training the ensemble model")
    print("="*70)

if __name__ == "__main__":
    main()
