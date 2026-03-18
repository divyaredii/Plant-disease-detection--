"""
Flask App using Keras/TensorFlow .h5 model for Plant Disease Detection
"""

import os
from flask import Flask, redirect, render_template, request, url_for, jsonify
from PIL import Image
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras.preprocessing import image as keras_image
import tensorflow as tf

# Load disease and supplement information
disease_info = pd.read_csv('Flask Deployed App/disease_info.csv', encoding='cp1252')
supplement_info = pd.read_csv('Flask Deployed App/supplement_info.csv', encoding='cp1252')

# Load the Keras model
print("Loading Keras model...")
model = keras.models.load_model('plant_disease_model.h5')
print("✓ Model loaded successfully!")
print(f"  Input shape: {model.input_shape}")
print(f"  Output classes: {model.output_shape[-1]}")

# Class names (3 classes - based on actual model)
# Note: This model appears to be a simplified version
class_names = [
    'Healthy',
    'Powdery',
    'Rust'
]

# If you need the full 39-class model, you'll need to train it
# using train_model.py or train_resnet50.py

def prediction_with_confidence(image_path):
    """
    Predict disease from image with confidence score
    Model expects 256x256 images
    """
    # Load and preprocess image (256x256 as per model requirements)
    img = keras_image.load_img(image_path, target_size=(256, 256))
    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize
    
    # Make prediction
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    confidence = float(predictions[0][predicted_class]) * 100
    
    return predicted_class, confidence

app = Flask(__name__, 
            template_folder='Flask Deployed App/templates',
            static_folder='Flask Deployed App/static')
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

@app.route('/submit', methods=['GET', 'POST'])
def submit():
    if request.method == 'POST':
        image = request.files['image']
        filename = image.filename
        
        # Create uploads directory if it doesn't exist
        upload_dir = 'Flask Deployed App/static/uploads'
        os.makedirs(upload_dir, exist_ok=True)
        
        file_path = os.path.join(upload_dir, filename)
        image.save(file_path)
        
        print(f"Processing image: {file_path}")
        
        # Get prediction with confidence
        pred, confidence = prediction_with_confidence(file_path)
        
        # Get disease information
        title = disease_info['disease_name'][pred]
        description = disease_info['description'][pred]
        prevent = disease_info['Possible Steps'][pred]
        image_url = disease_info['image_url'][pred]
        supplement_name = supplement_info['supplement name'][pred]
        supplement_image_url = supplement_info['supplement image'][pred]
        supplement_buy_link = supplement_info['buy link'][pred]
        
        print(f"Prediction: {title} (Confidence: {confidence:.2f}%)")
        
        return render_template('submit.html', 
                             title=title, 
                             desc=description, 
                             prevent=prevent,
                             image_url=image_url, 
                             pred=pred,
                             confidence=f"{confidence:.2f}",
                             sname=supplement_name, 
                             simage=supplement_image_url, 
                             buy_link=supplement_buy_link)

@app.route('/market', methods=['GET', 'POST'])
def market():
    return render_template('market.html', 
                         supplement_image=list(supplement_info['supplement image']),
                         supplement_name=list(supplement_info['supplement name']), 
                         disease=list(disease_info['disease_name']), 
                         buy=list(supplement_info['buy link']))

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions"""
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    image = request.files['image']
    filename = image.filename
    
    upload_dir = 'Flask Deployed App/static/uploads'
    os.makedirs(upload_dir, exist_ok=True)
    
    file_path = os.path.join(upload_dir, filename)
    image.save(file_path)
    
    pred, confidence = prediction_with_confidence(file_path)
    
    return jsonify({
        'disease': disease_info['disease_name'][pred],
        'confidence': f"{confidence:.2f}%",
        'description': disease_info['description'][pred],
        'prevention': disease_info['Possible Steps'][pred],
        'class_index': int(pred)
    })

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model': 'Keras/TensorFlow',
        'model_file': 'plant_disease_model.h5',
        'classes': len(class_names)
    })

if __name__ == '__main__':
    print("\n" + "="*60)
    print("Plant Disease Detection - Keras/TensorFlow Model")
    print("="*60)
    print(f"Model: plant_disease_model.h5")
    print(f"Classes: {len(class_names)}")
    print(f"Framework: TensorFlow/Keras")
    print("="*60 + "\n")
    
    app.run(debug=True, port=5001)  # Using port 5001 to avoid conflict
