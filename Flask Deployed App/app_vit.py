"""
Flask App with Vision Transformer (ViT) Support
================================================
Supports multiple models: CNN, ResNet50, and ViT
Uses the best available model for predictions.
"""

import os
import time
from flask import Flask, redirect, render_template, request, url_for, jsonify
from PIL import Image
import torchvision.transforms.functional as TF
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import json

# Try to import optional dependencies
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    GRADCAM_AVAILABLE = True
except ImportError:
    GRADCAM_AVAILABLE = False

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False

import CNN
from crop_model import calculate_crop_suitability, get_recommendation_text


# ============================================================
# Model Loading
# ============================================================
def load_vit_model(model_path, num_classes=39):
    """Load Vision Transformer model."""
    if not TIMM_AVAILABLE:
        print("⚠️  timm not installed. ViT not available.")
        return None

    try:
        model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=num_classes)
        in_features = model.head.in_features
        model.head = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Dropout(0.3),
            nn.Linear(in_features, 512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        print(f"✓ ViT model loaded from {model_path}")
        return model
    except Exception as e:
        print(f"⚠️  Could not load ViT model: {e}")
        return None


def load_cnn_model(model_path, num_classes=39):
    """Load original CNN model."""
    try:
        model = CNN.CNN(num_classes)
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        print(f"✓ CNN model loaded from {model_path}")
        return model
    except Exception as e:
        print(f"⚠️  Could not load CNN model: {e}")
        return None


# Load data
disease_info = pd.read_csv('disease_info.csv', encoding='cp1252')
supplement_info = pd.read_csv('supplement_info.csv', encoding='cp1252')

# Load models - prioritize ViT > CNN
print("\n" + "=" * 60)
print("🔧 Loading Models...")
print("=" * 60)

active_model = None
active_model_name = None

# Try ViT first (best accuracy)
vit_path = "plant_disease_model_vit.pt"
if os.path.exists(vit_path):
    active_model = load_vit_model(vit_path)
    if active_model:
        active_model_name = "Vision Transformer (ViT)"

# Fallback to CNN
if active_model is None:
    cnn_path = "plant_disease_model_1_latest.pt"
    if os.path.exists(cnn_path):
        active_model = load_cnn_model(cnn_path)
        if active_model:
            active_model_name = "CNN"

if active_model is None:
    print("❌ No model could be loaded! Please train a model first.")
    exit(1)

print(f"\n🎯 Active model: {active_model_name}")

# Class names
class_names = list(disease_info['disease_name'])


# ============================================================
# Prediction Functions
# ============================================================
def prediction_with_confidence(image_path):
    """Enhanced prediction with confidence scores and top-3 results."""
    start_time = time.time()

    image = Image.open(image_path)
    image = image.convert('RGB')
    original_image = np.array(image.resize((224, 224))) / 255.0

    image_resized = image.resize((224, 224))
    input_data = TF.to_tensor(image_resized)

    # Normalize for ViT / pretrained models
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    if active_model_name == "Vision Transformer (ViT)":
        input_data = (input_data - mean) / std

    input_data = input_data.unsqueeze(0)

    with torch.no_grad():
        output = active_model(input_data)
        probabilities = F.softmax(output, dim=1)

    top3_prob, top3_indices = torch.topk(probabilities, 3)

    predicted_class = top3_indices[0][0].item()
    confidence = top3_prob[0][0].item() * 100

    top3_results = []
    for i in range(3):
        idx = top3_indices[0][i].item()
        prob = top3_prob[0][i].item() * 100
        top3_results.append({
            'class': idx,
            'disease': disease_info['disease_name'][idx],
            'confidence': f"{prob:.2f}%",
            'probability': prob
        })

    inference_time = (time.time() - start_time) * 1000

    return {
        'predicted_class': predicted_class,
        'confidence': confidence,
        'top3': top3_results,
        'inference_time': f"{inference_time:.2f}ms",
        'model_used': active_model_name,
        'original_image': original_image,
        'input_tensor': input_data
    }


def generate_gradcam(image_path, predicted_class):
    """Generate Grad-CAM heatmap."""
    if not GRADCAM_AVAILABLE:
        return None

    # Grad-CAM works differently for ViT vs CNN
    try:
        image = Image.open(image_path).convert('RGB')
        image_resized = image.resize((224, 224))
        input_tensor = TF.to_tensor(image_resized).unsqueeze(0)
        rgb_img = np.array(image_resized) / 255.0

        if active_model_name == "CNN":
            target_layers = [active_model.conv_layers[-1]]
        elif active_model_name == "Vision Transformer (ViT)":
            # For ViT, use the last transformer block's norm layer
            target_layers = [active_model.blocks[-1].norm1]
        else:
            return None

        cam = GradCAM(model=active_model, target_layers=target_layers)
        grayscale_cam = cam(input_tensor=input_tensor, targets=None)
        grayscale_cam = grayscale_cam[0, :]

        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

        heatmap_path = os.path.join('static/uploads', 'gradcam_' + os.path.basename(image_path))
        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.imshow(rgb_img)
        plt.title('Original Image')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(visualization)
        plt.title(f'Grad-CAM ({active_model_name})')
        plt.axis('off')

        plt.tight_layout()
        plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
        plt.close()

        return 'uploads/' + os.path.basename(heatmap_path)
    except Exception as e:
        print(f"Grad-CAM error: {e}")
        return None


# ============================================================
# Flask App
# ============================================================
app = Flask(__name__)
app.secret_key = 'your-secret-key-here'


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/contact')
def contact():
    return render_template('contact-us.html')


@app.route('/index')
def ai_engine_page():
    return render_template('index.html')


@app.route('/mobile-device')
def mobile_device_detected_page():
    return render_template('mobile-device.html')


@app.route('/crop')
def crop():
    return render_template('crop.html')


@app.route('/crop_recommendation', methods=['POST'])
def crop_recommendation():
    try:
        n = float(request.form['nitrogen'])
        p = float(request.form['phosphorous'])
        k = float(request.form['pottasium'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])
        state = request.form['state']
        city = request.form['city']

        recommendations = calculate_crop_suitability(n, p, k)
        input_params = {'N': n, 'P': p, 'K': k}
        prediction_text = get_recommendation_text(recommendations, input_params)

        recommended_crops = {}
        for rec in recommendations:
            npk = f"N={rec['requirements']['N']}, P={rec['requirements']['P']}, K={rec['requirements']['K']}"
            recommended_crops[rec['crop']] = npk

        return render_template('crop-result.html',
                             crops=list(recommended_crops.keys()),
                             npk_values=recommended_crops,
                             n=n, p=p, k=k,
                             prediction=prediction_text)
    except Exception as e:
        print(f"Error in crop_recommendation: {str(e)}")
        return render_template('crop.html', error=str(e))


@app.route('/submit', methods=['GET', 'POST'])
def submit():
    if request.method == 'POST':
        try:
            image = request.files['image']
            filename = image.filename

            os.makedirs('static/uploads', exist_ok=True)

            file_path = os.path.join('static/uploads', filename)
            image.save(file_path)

            print(f"Processing: {filename} with {active_model_name}")

            result = prediction_with_confidence(file_path)
            pred = result['predicted_class']
            confidence = result['confidence']
            top3 = result['top3']
            inference_time = result['inference_time']
            model_used = result['model_used']

            gradcam_path = generate_gradcam(file_path, pred)

            title = disease_info['disease_name'][pred]
            description = disease_info['description'][pred]
            prevent = disease_info['Possible Steps'][pred]
            image_url = disease_info['image_url'][pred]
            supplement_name = supplement_info['supplement name'][pred]
            supplement_image_url = supplement_info['supplement image'][pred]
            supplement_buy_link = supplement_info['buy link'][pred]

            if confidence >= 90:
                confidence_level = "Very High"
                confidence_color = "success"
            elif confidence >= 75:
                confidence_level = "High"
                confidence_color = "info"
            elif confidence >= 60:
                confidence_level = "Moderate"
                confidence_color = "warning"
            else:
                confidence_level = "Low"
                confidence_color = "danger"

            print(f"✓ [{model_used}] Prediction: {title} ({confidence:.2f}%) in {inference_time}")

            return render_template('submit.html',
                                 title=title,
                                 desc=description,
                                 prevent=prevent,
                                 image_url=image_url,
                                 pred=pred,
                                 confidence=f"{confidence:.2f}",
                                 confidence_level=confidence_level,
                                 confidence_color=confidence_color,
                                 top3=top3,
                                 inference_time=inference_time,
                                 model_used=model_used,
                                 gradcam_path=gradcam_path,
                                 sname=supplement_name,
                                 simage=supplement_image_url,
                                 buy_link=supplement_buy_link)
        except Exception as e:
            print(f"Error in prediction: {str(e)}")
            import traceback
            traceback.print_exc()
            return render_template('index.html', error=f"Error processing image: {str(e)}")


@app.route('/market', methods=['GET', 'POST'])
def market():
    return render_template('market.html',
                         supplement_image=list(supplement_info['supplement image']),
                         supplement_name=list(supplement_info['supplement name']),
                         disease=list(disease_info['disease_name']),
                         buy=list(supplement_info['buy link']))


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions."""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400

        image = request.files['image']
        filename = image.filename

        os.makedirs('static/uploads', exist_ok=True)
        file_path = os.path.join('static/uploads', filename)
        image.save(file_path)

        result = prediction_with_confidence(file_path)
        pred = result['predicted_class']

        return jsonify({
            'disease': disease_info['disease_name'][pred],
            'confidence': f"{result['confidence']:.2f}%",
            'description': disease_info['description'][pred],
            'prevention': disease_info['Possible Steps'][pred],
            'top3_predictions': result['top3'],
            'inference_time': result['inference_time'],
            'model_used': result['model_used'],
            'class_index': int(pred)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health')
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'active_model': active_model_name,
        'classes': 39,
        'features': [
            'Vision Transformer (ViT) support',
            'Confidence scores',
            'Top-3 predictions',
            'Grad-CAM visualization',
            'Performance metrics',
            'Multi-model architecture'
        ]
    })


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("🚀 Plant Disease Detection System with Transformers")
    print("=" * 60)
    print(f"🎯 Active Model: {active_model_name}")
    print("✅ Confidence Scores")
    print("✅ Top-3 Predictions")
    print("✅ Grad-CAM Visualization")
    print("✅ Performance Metrics")
    print("✅ Vision Transformer Support")
    print("=" * 60 + "\n")

    app.run(debug=True, host='0.0.0.0', port=5000)
