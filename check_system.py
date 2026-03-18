"""
Dataset and Model Checker
Checks if dataset is downloaded and tests the Keras model
"""

import os
import sys
from pathlib import Path
import numpy as np
from PIL import Image

def check_dataset():
    """Check if dataset is downloaded"""
    print("="*60)
    print("Checking Dataset Status")
    print("="*60)
    
    dataset_path = Path("New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)")
    train_path = dataset_path / "train"
    valid_path = dataset_path / "valid"
    
    if dataset_path.exists():
        print("✓ Dataset folder found!")
        
        if train_path.exists():
            train_classes = len([d for d in train_path.iterdir() if d.is_dir()])
            print(f"✓ Training data found: {train_classes} classes")
            
            # Count training images
            total_train_images = sum(len(list((train_path / d).glob('*'))) 
                                    for d in train_path.iterdir() if d.is_dir())
            print(f"  Total training images: {total_train_images:,}")
        else:
            print("❌ Training data not found")
        
        if valid_path.exists():
            valid_classes = len([d for d in valid_path.iterdir() if d.is_dir()])
            print(f"✓ Validation data found: {valid_classes} classes")
            
            # Count validation images
            total_valid_images = sum(len(list((valid_path / d).glob('*'))) 
                                    for d in valid_path.iterdir() if d.is_dir())
            print(f"  Total validation images: {total_valid_images:,}")
        else:
            print("❌ Validation data not found")
        
        return train_path.exists() and valid_path.exists()
    else:
        print("❌ Dataset not downloaded yet")
        print("\nDataset is still downloading...")
        print("Please wait for download_dataset.py to complete")
        return False

def check_model():
    """Check if Keras model exists and can be loaded"""
    print("\n" + "="*60)
    print("Checking Keras Model")
    print("="*60)
    
    model_path = Path("plant_disease_model.h5")
    
    if model_path.exists():
        size_mb = model_path.stat().st_size / (1024 * 1024)
        print(f"✓ Model file found: plant_disease_model.h5")
        print(f"  Size: {size_mb:.2f} MB")
        
        try:
            from tensorflow import keras
            print("\n  Loading model...")
            model = keras.models.load_model(str(model_path))
            print("  ✓ Model loaded successfully!")
            
            # Get model info
            print(f"\n  Model Architecture:")
            print(f"    Input shape: {model.input_shape}")
            print(f"    Output shape: {model.output_shape}")
            print(f"    Total layers: {len(model.layers)}")
            
            total_params = model.count_params()
            print(f"    Total parameters: {total_params:,}")
            
            return True
        except Exception as e:
            print(f"  ❌ Error loading model: {e}")
            return False
    else:
        print("❌ Model file not found: plant_disease_model.h5")
        return False

def test_model_prediction():
    """Test model with a sample image"""
    print("\n" + "="*60)
    print("Testing Model Prediction")
    print("="*60)
    
    # Check if test images exist
    test_images_dir = Path("test_images")
    
    if not test_images_dir.exists():
        print("❌ test_images directory not found")
        return
    
    # Get first image
    test_images = list(test_images_dir.glob('*.jpg')) + list(test_images_dir.glob('*.png'))
    
    if not test_images:
        print("❌ No test images found")
        return
    
    test_image = test_images[0]
    print(f"\nTesting with: {test_image.name}")
    
    try:
        from tensorflow import keras
        from tensorflow.keras.preprocessing import image as keras_image
        
        # Load model
        model = keras.models.load_model('plant_disease_model.h5')
        
        # Load and preprocess image
        img = keras_image.load_img(str(test_image), target_size=(224, 224))
        img_array = keras_image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0
        
        # Make prediction
        print("  Making prediction...")
        predictions = model.predict(img_array, verbose=0)
        predicted_class = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class]) * 100
        
        # Class names
        class_names = [
            'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust',
            'Apple___healthy', 'Background_without_leaves', 'Blueberry___healthy',
            'Cherry___Powdery_mildew', 'Cherry___healthy',
            'Corn___Cercospora_leaf_spot Gray_leaf_spot', 'Corn___Common_rust',
            'Corn___Northern_Leaf_Blight', 'Corn___healthy', 'Grape___Black_rot',
            'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
            'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)',
            'Peach___Bacterial_spot', 'Peach___healthy', 'Pepper,_bell___Bacterial_spot',
            'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight',
            'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
            'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
            'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight',
            'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
            'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
            'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
            'Tomato___healthy'
        ]
        
        print(f"\n  ✓ Prediction Results:")
        print(f"    Disease: {class_names[predicted_class]}")
        print(f"    Confidence: {confidence:.2f}%")
        print(f"    Class Index: {predicted_class}")
        
        # Show top 3 predictions
        top_3_idx = np.argsort(predictions[0])[-3:][::-1]
        print(f"\n  Top 3 Predictions:")
        for i, idx in enumerate(top_3_idx, 1):
            print(f"    {i}. {class_names[idx]}: {predictions[0][idx]*100:.2f}%")
        
    except Exception as e:
        print(f"  ❌ Error during prediction: {e}")
        import traceback
        traceback.print_exc()

def check_dependencies():
    """Check if required packages are installed"""
    print("\n" + "="*60)
    print("Checking Dependencies")
    print("="*60)
    
    required_packages = {
        'tensorflow': 'TensorFlow',
        'keras': 'Keras',
        'numpy': 'NumPy',
        'PIL': 'Pillow',
        'pandas': 'Pandas',
        'flask': 'Flask'
    }
    
    missing_packages = []
    
    for package, name in required_packages.items():
        try:
            __import__(package)
            print(f"✓ {name} installed")
        except ImportError:
            print(f"❌ {name} NOT installed")
            missing_packages.append(package if package != 'PIL' else 'pillow')
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print(f"\nInstall with:")
        print(f"  pip install {' '.join(missing_packages)}")
        return False
    
    return True

def main():
    """Main function"""
    print("\n" + "="*70)
    print(" "*15 + "PLANT DISEASE DETECTION - SYSTEM CHECK")
    print("="*70 + "\n")
    
    # Check dependencies
    deps_ok = check_dependencies()
    
    if not deps_ok:
        print("\n" + "="*70)
        print("⚠️  Please install missing dependencies first")
        print("="*70)
        return
    
    # Check model
    model_ok = check_model()
    
    # Check dataset
    dataset_ok = check_dataset()
    
    # Test model if available
    if model_ok:
        test_model_prediction()
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    print(f"\n{'Dependencies:':<20} {'✓ All installed' if deps_ok else '❌ Missing packages'}")
    print(f"{'Keras Model:':<20} {'✓ Ready' if model_ok else '❌ Not found/Error'}")
    print(f"{'Dataset:':<20} {'✓ Downloaded' if dataset_ok else '❌ Not downloaded yet'}")
    
    print("\n" + "="*70)
    print("NEXT STEPS")
    print("="*70)
    
    if not dataset_ok:
        print("\n1. ⏳ Wait for dataset download to complete")
        print("   (download_dataset.py is currently running)")
    
    if model_ok and not dataset_ok:
        print("\n2. ✅ You can use the existing model now:")
        print("   python app_keras.py")
        print("   (Will run on http://127.0.0.1:5001)")
    
    if dataset_ok:
        print("\n2. ✅ Dataset ready! You can:")
        print("   a) Use existing model: python app_keras.py")
        print("   b) Train new model: python train_resnet50.py")
        print("   c) Train basic CNN: python train_model.py")
    
    print("\n" + "="*70 + "\n")

if __name__ == "__main__":
    main()
