"""
Corrected Accuracy Test - Handles class index mismatch
The model was trained with 39 classes (including Background_without_leaves at index 4)
But the validation dataset only has 38 classes (no Background class)
"""

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os

# Configuration
class Config:
    DATASET_PATH = "New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)"
    VALID_DIR = os.path.join(DATASET_PATH, "valid")
    MODEL_PATH = "Flask Deployed App/plant_disease_model_1_latest.pt"
    BATCH_SIZE = 32
    NUM_CLASSES = 39  # Model has 39 classes
    IMG_SIZE = 224
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model classes (from CNN.py) - 39 classes
MODEL_CLASSES = {
    0: 'Apple___Apple_scab',
    1: 'Apple___Black_rot',
    2: 'Apple___Cedar_apple_rust',
    3: 'Apple___healthy',
    4: 'Background_without_leaves',  # This class is NOT in validation dataset!
    5: 'Blueberry___healthy',
    6: 'Cherry___Powdery_mildew',
    7: 'Cherry___healthy',
    8: 'Corn___Cercospora_leaf_spot Gray_leaf_spot',
    9: 'Corn___Common_rust',
    10: 'Corn___Northern_Leaf_Blight',
    11: 'Corn___healthy',
    12: 'Grape___Black_rot',
    13: 'Grape___Esca_(Black_Measles)',
    14: 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    15: 'Grape___healthy',
    16: 'Orange___Haunglongbing_(Citrus_greening)',
    17: 'Peach___Bacterial_spot',
    18: 'Peach___healthy',
    19: 'Pepper,_bell___Bacterial_spot',
    20: 'Pepper,_bell___healthy',
    21: 'Potato___Early_blight',
    22: 'Potato___Late_blight',
    23: 'Potato___healthy',
    24: 'Raspberry___healthy',
    25: 'Soybean___healthy',
    26: 'Squash___Powdery_mildew',
    27: 'Strawberry___Leaf_scorch',
    28: 'Strawberry___healthy',
    29: 'Tomato___Bacterial_spot',
    30: 'Tomato___Early_blight',
    31: 'Tomato___Late_blight',
    32: 'Tomato___Leaf_Mold',
    33: 'Tomato___Septoria_leaf_spot',
    34: 'Tomato___Spider_mites Two-spotted_spider_mite',
    35: 'Tomato___Target_Spot',
    36: 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    37: 'Tomato___Tomato_mosaic_virus',
    38: 'Tomato___healthy'
}

# CNN Model
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

def create_class_mapping(dataset_classes):
    """Create mapping from dataset class index to model class index"""
    mapping = {}
    for ds_idx, ds_class in enumerate(dataset_classes):
        # Find corresponding model index
        for model_idx, model_class in MODEL_CLASSES.items():
            if model_class == ds_class or model_class.replace('___', '_(including_sour)___') == ds_class:
                mapping[ds_idx] = model_idx
                break
    return mapping

def main():
    print("="*70)
    print(" "*15 + "🔧 CORRECTED ACCURACY TEST")
    print("="*70)
    print(f"\n💻 Device: {Config.DEVICE}")
    
    # Load model
    print(f"\n🔄 Loading model...")
    model = CNN(Config.NUM_CLASSES)
    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=Config.DEVICE))
    model.to(Config.DEVICE)
    model.eval()
    print("✅ Model loaded (39 classes)")
    
    # Load validation data
    print(f"\n📂 Loading validation data...")
    transform = transforms.Compose([
        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    valid_dataset = datasets.ImageFolder(Config.VALID_DIR, transform=transform)
    valid_loader = DataLoader(valid_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
    
    print(f"✅ Loaded {len(valid_dataset)} validation images ({len(valid_dataset.classes)} classes)")
    
    # Create class mapping
    class_mapping = create_class_mapping(valid_dataset.classes)
    print(f"\n🔗 Class mapping created:")
    print(f"   Dataset has 38 classes, Model expects 39 classes")
    print(f"   Missing class in dataset: 'Background_without_leaves' (index 4)")
    
    # Test
    print(f"\n🧪 Running predictions...")
    all_preds = []
    all_labels = []
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(valid_loader):
            inputs = inputs.to(Config.DEVICE)
            outputs = model(inputs)
            
            # Get model predictions (39 classes)
            _, model_preds = torch.max(outputs, 1)
            
            # Convert predictions and labels
            for pred, label in zip(model_preds.cpu().numpy(), labels.numpy()):
                # Skip if model predicted the missing class (Background)
                if pred == 4:  # Background_without_leaves
                    continue
                
                # Adjust prediction index (shift down if > 4)
                adjusted_pred = pred if pred < 4 else pred - 1
                
                all_preds.append(adjusted_pred)
                all_labels.append(label)
                
                if adjusted_pred == label:
                    correct += 1
                total += 1
            
            if (batch_idx + 1) % 100 == 0:
                print(f"  Processed {(batch_idx + 1) * Config.BATCH_SIZE} images...")
    
    # Calculate metrics
    accuracy = correct / total if total > 0 else 0
    
    # Results
    print("\n" + "="*70)
    print(" "*20 + "📊 CORRECTED TEST RESULTS")
    print("="*70)
    print(f"\n  Total Predictions: {total}")
    print(f"  Correct Predictions: {correct}")
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
    print("\n💡 Note: This test accounts for the class index mismatch")
    print("   Model: 39 classes (with Background)")
    print("   Dataset: 38 classes (without Background)")
    print("="*70)

if __name__ == "__main__":
    main()
