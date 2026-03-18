"""
ENSEMBLE FLASK APP - Multi-Model Plant Disease Detection
Combines ResNet50, EfficientNet-B3, and DenseNet121 for maximum accuracy
"""

from flask import Flask, render_template, request, jsonify
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import io
import json
import os
import time
import numpy as np

app = Flask(__name__)

# Configuration
class Config:
    MODEL_DIR = "models"
    RESNET_PATH = os.path.join(MODEL_DIR, "resnet50_plant.pt")
    EFFICIENTNET_PATH = os.path.join(MODEL_DIR, "efficientnet_b3_plant.pt")
    DENSENET_PATH = os.path.join(MODEL_DIR, "densenet121_plant.pt")
    ENSEMBLE_CONFIG_PATH = os.path.join(MODEL_DIR, "ensemble_weights.json")
    
    NUM_CLASSES = 39
    IMG_SIZE = 224
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Disease information database
disease_info = {
    'Apple___Apple_scab': {
        'description': 'Apple scab is a fungal disease caused by Venturia inaequalis.',
        'prevention': 'Remove fallen leaves, apply fungicides, use resistant varieties.'
    },
    'Apple___Black_rot': {
        'description': 'Black rot is caused by the fungus Botryosphaeria obtusa.',
        'prevention': 'Prune infected branches, remove mummified fruits, apply fungicides.'
    },
    'Apple___Cedar_apple_rust': {
        'description': 'Cedar apple rust is caused by Gymnosporangium juniperi-virginianae.',
        'prevention': 'Remove nearby cedar trees, apply fungicides in spring.'
    },
    'Apple___healthy': {
        'description': 'Your apple plant appears healthy!',
        'prevention': 'Continue regular care and monitoring.'
    },
    # Add more diseases as needed...
}

# Global variables for models
ensemble_models = {}
ensemble_weights = []
class_names = []

def create_resnet50(num_classes=39):
    """Create ResNet50 model architecture"""
    model = models.resnet50(pretrained=False)
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes)
    )
    return model

def create_efficientnet_b3(num_classes=39):
    """Create EfficientNet-B3 model architecture"""
    model = models.efficientnet_b3(pretrained=False)
    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes)
    )
    return model

def create_densenet121(num_classes=39):
    """Create DenseNet121 model architecture"""
    model = models.densenet121(pretrained=False)
    num_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes)
    )
    return model

def load_ensemble_models():
    """Load all ensemble models"""
    global ensemble_models, ensemble_weights, class_names
    
    print("🔄 Loading ensemble models...")
    
    # Load ensemble configuration
    if os.path.exists(Config.ENSEMBLE_CONFIG_PATH):
        with open(Config.ENSEMBLE_CONFIG_PATH, 'r') as f:
            config = json.load(f)
            ensemble_weights = config['weights']
            class_names = config['class_names']
            print(f"✓ Loaded ensemble config: {config['models']}")
            print(f"  Weights: {ensemble_weights}")
            print(f"  Ensemble accuracy: {config['ensemble_accuracy']:.4f}")
    else:
        # Default equal weights
        ensemble_weights = [1/3, 1/3, 1/3]
        print("⚠️ No ensemble config found, using equal weights")
    
    # Load ResNet50
    if os.path.exists(Config.RESNET_PATH):
        print("  Loading ResNet50...")
        resnet = create_resnet50(Config.NUM_CLASSES)
        resnet.load_state_dict(torch.load(Config.RESNET_PATH, map_location=Config.DEVICE))
        resnet.to(Config.DEVICE)
        resnet.eval()
        ensemble_models['resnet50'] = resnet
        print("  ✓ ResNet50 loaded")
    
    # Load EfficientNet-B3
    if os.path.exists(Config.EFFICIENTNET_PATH):
        print("  Loading EfficientNet-B3...")
        efficientnet = create_efficientnet_b3(Config.NUM_CLASSES)
        efficientnet.load_state_dict(torch.load(Config.EFFICIENTNET_PATH, map_location=Config.DEVICE))
        efficientnet.to(Config.DEVICE)
        efficientnet.eval()
        ensemble_models['efficientnet_b3'] = efficientnet
        print("  ✓ EfficientNet-B3 loaded")
    
    # Load DenseNet121
    if os.path.exists(Config.DENSENET_PATH):
        print("  Loading DenseNet121...")
        densenet = create_densenet121(Config.NUM_CLASSES)
        densenet.load_state_dict(torch.load(Config.DENSENET_PATH, map_location=Config.DEVICE))
        densenet.to(Config.DEVICE)
        densenet.eval()
        ensemble_models['densenet121'] = densenet
        print("  ✓ DenseNet121 loaded")
    
    if not ensemble_models:
        print("❌ No models loaded! Please train models first.")
        return False
    
    print(f"✅ Loaded {len(ensemble_models)} models successfully!")
    return True

def get_transforms():
    """Get image transforms"""
    return transforms.Compose([
        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def predict_ensemble(image_bytes):
    """Make ensemble prediction"""
    start_time = time.time()
    
    # Load and transform image
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    transform = get_transforms()
    image_tensor = transform(image).unsqueeze(0).to(Config.DEVICE)
    
    # Get predictions from all models
    all_predictions = []
    model_outputs = {}
    
    with torch.no_grad():
        for model_name, model in ensemble_models.items():
            outputs = model(image_tensor)
            probs = torch.softmax(outputs, dim=1)
            all_predictions.append(probs)
            model_outputs[model_name] = {
                'prediction': class_names[torch.argmax(probs).item()],
                'confidence': torch.max(probs).item() * 100
            }
    
    # Weighted ensemble
    if len(ensemble_weights) == len(all_predictions):
        weighted_output = sum(w * pred for w, pred in zip(ensemble_weights, all_predictions))
    else:
        # Equal weights if mismatch
        weighted_output = sum(all_predictions) / len(all_predictions)
    
    # Get final prediction
    probabilities = weighted_output[0].cpu().numpy()
    predicted_idx = np.argmax(probabilities)
    confidence = probabilities[predicted_idx] * 100
    
    # Get top 3 predictions
    top3_idx = np.argsort(probabilities)[-3:][::-1]
    top3_predictions = [
        {
            'disease': class_names[idx],
            'confidence': f"{probabilities[idx] * 100:.2f}%",
            'probability': float(probabilities[idx] * 100)
        }
        for idx in top3_idx
    ]
    
    inference_time = (time.time() - start_time) * 1000
    
    predicted_disease = class_names[predicted_idx]
    
    # Get disease info
    info = disease_info.get(predicted_disease, {
        'description': 'Information not available for this disease.',
        'prevention': 'Consult with a plant pathologist.'
    })
    
    return {
        'disease': predicted_disease,
        'confidence': f"{confidence:.2f}%",
        'description': info['description'],
        'prevention': info['prevention'],
        'top3_predictions': top3_predictions,
        'individual_models': model_outputs,
        'inference_time': f"{inference_time:.2f}ms",
        'ensemble_size': len(ensemble_models),
        'class_index': int(predicted_idx)
    }

@app.route('/')
def index():
    """Home page"""
    return render_template('index_ensemble.html', 
                         num_models=len(ensemble_models),
                         model_names=list(ensemble_models.keys()))

@app.route('/predict', methods=['POST'])
def predict():
    """Prediction endpoint"""
    if 'image' not in request.files:
        return render_template('index_ensemble.html', error='No image uploaded')
    
    file = request.files['image']
    if file.filename == '':
        return render_template('index_ensemble.html', error='No image selected')
    
    try:
        image_bytes = file.read()
        result = predict_ensemble(image_bytes)
        return render_template('submit_ensemble.html', **result)
    except Exception as e:
        return render_template('index_ensemble.html', error=f'Error: {str(e)}')

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions"""
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400
    
    try:
        image_bytes = file.read()
        result = predict_ensemble(image_bytes)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': len(ensemble_models),
        'model_names': list(ensemble_models.keys()),
        'device': str(Config.DEVICE)
    })

if __name__ == '__main__':
    print("="*60)
    print("🚀 ENSEMBLE PLANT DISEASE DETECTION APP")
    print("="*60)
    
    # Load models
    if load_ensemble_models():
        print(f"\n🌐 Starting Flask server on http://localhost:5000")
        print(f"📊 Ensemble size: {len(ensemble_models)} models")
        print(f"💻 Device: {Config.DEVICE}")
        print("="*60)
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("\n❌ Failed to load models. Please train the ensemble first:")
        print("   python train_ensemble.py")
