"""
IMPROVED Flask App with Enhanced Features
- Confidence scores
- Top-3 predictions
- Grad-CAM visualization
- Performance optimization
- Better error handling
"""

import os
import time
from flask import Flask, redirect, render_template, request, url_for, jsonify
from PIL import Image
import torchvision.transforms.functional as TF
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import cv2
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import CNN
from crop_model import calculate_crop_suitability, get_recommendation_text

# Load data
disease_info = pd.read_csv('disease_info.csv', encoding='cp1252')
supplement_info = pd.read_csv('supplement_info.csv', encoding='cp1252')

# Load model
print("Loading model...")
model = CNN.CNN(39)
model.load_state_dict(torch.load("plant_disease_model_1_latest.pt", map_location=torch.device('cpu')))
model.eval()
print("✓ Model loaded successfully!")

# Class names for better display
class_names = list(disease_info['disease_name'])

def prediction_with_confidence(image_path):
    """
    Enhanced prediction with confidence scores and top-3 results
    """
    start_time = time.time()
    
    # Load and preprocess image
    image = Image.open(image_path)
    image = image.convert('RGB')
    original_image = np.array(image.resize((224, 224))) / 255.0
    
    # Prepare for model
    image_resized = image.resize((224, 224))
    input_data = TF.to_tensor(image_resized)
    input_data = input_data.unsqueeze(0)
    
    # Get predictions
    with torch.no_grad():
        output = model(input_data)
        probabilities = F.softmax(output, dim=1)
    
    # Get top-3 predictions
    top3_prob, top3_indices = torch.topk(probabilities, 3)
    
    # Primary prediction
    predicted_class = top3_indices[0][0].item()
    confidence = top3_prob[0][0].item() * 100
    
    # Top-3 results
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
    
    inference_time = (time.time() - start_time) * 1000  # Convert to ms
    
    return {
        'predicted_class': predicted_class,
        'confidence': confidence,
        'top3': top3_results,
        'inference_time': f"{inference_time:.2f}ms",
        'original_image': original_image,
        'input_tensor': input_data
    }

def generate_gradcam(image_path, predicted_class):
    """
    Generate Grad-CAM heatmap to visualize what the model is looking at
    """
    try:
        # Load image
        image = Image.open(image_path).convert('RGB')
        image_resized = image.resize((224, 224))
        input_tensor = TF.to_tensor(image_resized).unsqueeze(0)
        
        # Prepare image for overlay
        rgb_img = np.array(image_resized) / 255.0
        
        # Create Grad-CAM
        # Use the last convolutional layer
        target_layers = [model.conv_layers[-1]]
        cam = GradCAM(model=model, target_layers=target_layers)
        
        # Generate CAM
        grayscale_cam = cam(input_tensor=input_tensor, targets=None)
        grayscale_cam = grayscale_cam[0, :]
        
        # Create visualization
        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        
        # Save heatmap
        heatmap_path = os.path.join('static/uploads', 'gradcam_' + os.path.basename(image_path))
        plt.figure(figsize=(10, 5))
        
        plt.subplot(1, 2, 1)
        plt.imshow(rgb_img)
        plt.title('Original Image')
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(visualization)
        plt.title('Grad-CAM: What the Model Sees')
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return 'uploads/' + os.path.basename(heatmap_path)
    except Exception as e:
        print(f"Grad-CAM error: {e}")
        return None

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
            
            # Create uploads directory
            os.makedirs('static/uploads', exist_ok=True)
            
            file_path = os.path.join('static/uploads', filename)
            image.save(file_path)
            
            print(f"Processing: {filename}")
            
            # Get enhanced prediction
            result = prediction_with_confidence(file_path)
            pred = result['predicted_class']
            confidence = result['confidence']
            top3 = result['top3']
            inference_time = result['inference_time']
            
            # Generate Grad-CAM visualization
            gradcam_path = generate_gradcam(file_path, pred)
            
            # Get disease information
            title = disease_info['disease_name'][pred]
            description = disease_info['description'][pred]
            prevent = disease_info['Possible Steps'][pred]
            image_url = disease_info['image_url'][pred]
            supplement_name = supplement_info['supplement name'][pred]
            supplement_image_url = supplement_info['supplement image'][pred]
            supplement_buy_link = supplement_info['buy link'][pred]
            
            # Confidence level indicator
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
            
            print(f"✓ Prediction: {title} ({confidence:.2f}%) in {inference_time}")
            
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
    """
    API endpoint for predictions with confidence scores
    """
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
            'class_index': int(pred)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model': 'CNN (PyTorch)',
        'classes': 39,
        'features': [
            'Confidence scores',
            'Top-3 predictions',
            'Grad-CAM visualization',
            'Performance metrics'
        ]
    })

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 IMPROVED Plant Disease Detection System")
    print("="*60)
    print("✅ Confidence Scores")
    print("✅ Top-3 Predictions")
    print("✅ Grad-CAM Visualization")
    print("✅ Performance Metrics")
    print("="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
