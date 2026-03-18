"""
Test Downloaded Keras Models
Tests the trained_model.h5 and trained_model.keras files from Google Drive
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import os
from pathlib import Path

# Configuration
class Config:
    DATASET_PATH = "New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)"
    VALID_DIR = os.path.join(DATASET_PATH, "valid")
    H5_MODEL_PATH = "downloaded_models/trained_model.h5"
    KERAS_MODEL_PATH = "downloaded_models/trained_model.keras"
    IMG_SIZE = 224
    BATCH_SIZE = 32

# Class names (38 classes from the dataset)
CLASS_NAMES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

def load_and_test_model(model_path, model_name):
    """Load and test a model"""
    print(f"\n{'='*70}")
    print(f"Testing: {model_name}")
    print(f"{'='*70}")
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return None
    
    print(f"📂 Loading model from: {model_path}")
    print(f"   Size: {os.path.getsize(model_path) / 1024 / 1024:.1f} MB")
    
    try:
        # Load model
        model = keras.models.load_model(model_path)
        print(f"✅ Model loaded successfully!")
        
        # Print model summary
        print(f"\n📊 Model Architecture:")
        model.summary()
        
        # Get model info
        print(f"\n📋 Model Information:")
        print(f"   Input shape: {model.input_shape}")
        print(f"   Output shape: {model.output_shape}")
        print(f"   Total parameters: {model.count_params():,}")
        print(f"   Number of layers: {len(model.layers)}")
        
        # Test with validation data
        print(f"\n🧪 Testing on validation dataset...")
        
        # Create data generator
        datagen = keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255
        )
        
        valid_generator = datagen.flow_from_directory(
            Config.VALID_DIR,
            target_size=(Config.IMG_SIZE, Config.IMG_SIZE),
            batch_size=Config.BATCH_SIZE,
            class_mode='categorical',
            shuffle=False
        )
        
        print(f"   Found {valid_generator.samples} validation images")
        print(f"   Classes: {valid_generator.num_classes}")
        
        # Evaluate model
        print(f"\n⏳ Evaluating model (this may take a few minutes)...")
        results = model.evaluate(valid_generator, verbose=1)
        
        # Print results
        print(f"\n{'='*70}")
        print(f"📊 RESULTS FOR {model_name}")
        print(f"{'='*70}")
        print(f"   Loss: {results[0]:.4f}")
        print(f"   Accuracy: {results[1]*100:.2f}%")
        
        if results[1] >= 0.98:
            print(f"   Rating: 🌟 EXCELLENT")
        elif results[1] >= 0.95:
            print(f"   Rating: ✅ VERY GOOD")
        elif results[1] >= 0.90:
            print(f"   Rating: 👍 GOOD")
        elif results[1] >= 0.85:
            print(f"   Rating: ⚠️ FAIR")
        else:
            print(f"   Rating: ❌ NEEDS IMPROVEMENT")
        
        print(f"{'='*70}")
        
        return model, results[1]
        
    except Exception as e:
        print(f"❌ Error loading/testing model: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_single_prediction(model, image_path):
    """Test prediction on a single image"""
    print(f"\n🖼️  Testing single image prediction...")
    
    try:
        # Load and preprocess image
        img = Image.open(image_path).convert('RGB')
        img = img.resize((Config.IMG_SIZE, Config.IMG_SIZE))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Make prediction
        predictions = model.predict(img_array, verbose=0)
        predicted_class = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class] * 100
        
        print(f"   Predicted: {CLASS_NAMES[predicted_class]}")
        print(f"   Confidence: {confidence:.2f}%")
        
        # Top 3 predictions
        top3_idx = np.argsort(predictions[0])[-3:][::-1]
        print(f"\n   Top 3 Predictions:")
        for i, idx in enumerate(top3_idx, 1):
            print(f"   {i}. {CLASS_NAMES[idx]}: {predictions[0][idx]*100:.2f}%")
        
    except Exception as e:
        print(f"❌ Error in prediction: {e}")

def main():
    print("="*70)
    print(" "*15 + "🧪 TESTING DOWNLOADED KERAS MODELS")
    print("="*70)
    
    print(f"\n💻 TensorFlow version: {tf.__version__}")
    print(f"🔧 Keras version: {keras.__version__}")
    
    # Test H5 model
    h5_model, h5_accuracy = load_and_test_model(Config.H5_MODEL_PATH, "trained_model.h5")
    
    # Test Keras model
    keras_model, keras_accuracy = load_and_test_model(Config.KERAS_MODEL_PATH, "trained_model.keras")
    
    # Summary
    print(f"\n{'='*70}")
    print(f" "*20 + "📊 FINAL SUMMARY")
    print(f"{'='*70}")
    
    if h5_accuracy is not None:
        print(f"\n✅ trained_model.h5:")
        print(f"   Accuracy: {h5_accuracy*100:.2f}%")
        print(f"   Status: {'Production Ready ✓' if h5_accuracy >= 0.90 else 'Needs Improvement ⚠️'}")
    
    if keras_accuracy is not None:
        print(f"\n✅ trained_model.keras:")
        print(f"   Accuracy: {keras_accuracy*100:.2f}%")
        print(f"   Status: {'Production Ready ✓' if keras_accuracy >= 0.90 else 'Needs Improvement ⚠️'}")
    
    # Recommendation
    print(f"\n{'='*70}")
    print(f"💡 RECOMMENDATION")
    print(f"{'='*70}")
    
    if h5_accuracy and h5_accuracy >= 0.90:
        print(f"\n✅ The H5 model is ready to use!")
        print(f"   Copy to Flask app:")
        print(f"   copy downloaded_models\\trained_model.h5 \"Flask Deployed App\\plant_disease_model.h5\"")
    elif keras_accuracy and keras_accuracy >= 0.90:
        print(f"\n✅ The Keras model is ready to use!")
        print(f"   Copy to Flask app:")
        print(f"   copy downloaded_models\\trained_model.keras \"Flask Deployed App\\plant_disease_model.keras\"")
    else:
        print(f"\n⚠️ Models have lower accuracy than expected.")
        print(f"   Consider training the ensemble model for 98-99.5% accuracy:")
        print(f"   python train_ensemble.py")
    
    print(f"\n{'='*70}")

if __name__ == "__main__":
    main()
