"""
Quick Accuracy Test - Tests on a smaller sample for faster results
"""

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
from sklearn.metrics import accuracy_score
import os

# Configuration
class Config:
    DATASET_PATH = "New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)"
    VALID_DIR = os.path.join(DATASET_PATH, "valid")
    MODEL_PATH = "Flask Deployed App/plant_disease_model_1_latest.pt"
    BATCH_SIZE = 32
    NUM_CLASSES = 39
    IMG_SIZE = 224
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    SAMPLE_SIZE = 1000  # Test on 1000 images for quick results

# CNN Model (from CNN.py)
class CNN(nn.Module):
    def __init__(self, K=39):
        super(CNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
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
        out = out.view(-1, 50176)
        out = self.dense_layers(out)
        return out

def main():
    print("="*70)
    print(" "*15 + "🚀 QUICK ACCURACY TEST (Sample)")
    print("="*70)
    print(f"\n💻 Device: {Config.DEVICE}")
    print(f"📊 Sample Size: {Config.SAMPLE_SIZE} images")
    
    # Load model
    print(f"\n🔄 Loading model...")
    model = CNN(Config.NUM_CLASSES)
    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=Config.DEVICE))
    model.to(Config.DEVICE)
    model.eval()
    print("✅ Model loaded!")
    
    # Load validation data
    print(f"\n📂 Loading validation data...")
    transform = transforms.Compose([
        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    full_dataset = datasets.ImageFolder(Config.VALID_DIR, transform=transform)
    
    # Create random sample
    indices = np.random.choice(len(full_dataset), min(Config.SAMPLE_SIZE, len(full_dataset)), replace=False)
    sample_dataset = Subset(full_dataset, indices)
    
    loader = DataLoader(sample_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
    
    print(f"✅ Testing on {len(sample_dataset)} images")
    
    # Test
    print(f"\n🧪 Running predictions...")
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(Config.DEVICE)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    # Calculate accuracy
    accuracy = accuracy_score(all_labels, all_preds)
    
    # Results
    print("\n" + "="*70)
    print(" "*20 + "📊 QUICK TEST RESULTS")
    print("="*70)
    print(f"\n  Sample Size: {len(sample_dataset)} images")
    print(f"  Accuracy: {accuracy*100:.2f}%")
    
    if accuracy >= 0.98:
        print(f"  Rating: 🌟 EXCELLENT")
    elif accuracy >= 0.95:
        print(f"  Rating: ✅ VERY GOOD")
    elif accuracy >= 0.90:
        print(f"  Rating: 👍 GOOD")
    elif accuracy >= 0.85:
        print(f"  Rating: ⚠️ FAIR")
    else:
        print(f"  Rating: ❌ NEEDS IMPROVEMENT")
    
    print("\n" + "="*70)
    print("\n💡 Note: This is a quick test on a sample.")
    print("   For full accuracy, run: python test_accuracy.py")
    print("="*70)

if __name__ == "__main__":
    main()
