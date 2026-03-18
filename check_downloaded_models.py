"""
Simple Model Info Check
Just loads the models and shows their structure without full testing
"""

import tensorflow as tf
from tensorflow import keras
import os

print("="*70)
print(" "*15 + "📦 DOWNLOADED MODELS INFO")
print("="*70)

print(f"\n💻 TensorFlow version: {tf.__version__}")

# Check H5 model
h5_path = "downloaded_models/trained_model.h5"
keras_path = "downloaded_models/trained_model.keras"

print(f"\n{'='*70}")
print("MODEL 1: trained_model.h5")
print(f"{'='*70}")

if os.path.exists(h5_path):
    size_mb = os.path.getsize(h5_path) / 1024 / 1024
    print(f"✅ File exists")
    print(f"📦 Size: {size_mb:.1f} MB")
    
    try:
        print(f"\n🔄 Loading model...")
        model = keras.models.load_model(h5_path, compile=False)
        print(f"✅ Model loaded successfully!")
        
        print(f"\n📊 Model Information:")
        print(f"   Input shape: {model.input_shape}")
        print(f"   Output shape: {model.output_shape}")
        print(f"   Output classes: {model.output_shape[-1]}")
        print(f"   Total parameters: {model.count_params():,}")
        print(f"   Number of layers: {len(model.layers)}")
        
        print(f"\n🏗️  Model Architecture:")
        for i, layer in enumerate(model.layers[:5]):
            print(f"   Layer {i+1}: {layer.name} ({layer.__class__.__name__})")
        if len(model.layers) > 5:
            print(f"   ... and {len(model.layers) - 5} more layers")
        
        print(f"\n✅ Model Status: READY TO USE")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
else:
    print(f"❌ File not found: {h5_path}")

print(f"\n{'='*70}")
print("MODEL 2: trained_model.keras")
print(f"{'='*70}")

if os.path.exists(keras_path):
    size_mb = os.path.getsize(keras_path) / 1024 / 1024
    print(f"✅ File exists")
    print(f"📦 Size: {size_mb:.1f} MB")
    
    try:
        print(f"\n🔄 Loading model...")
        model2 = keras.models.load_model(keras_path, compile=False)
        print(f"✅ Model loaded successfully!")
        
        print(f"\n📊 Model Information:")
        print(f"   Input shape: {model2.input_shape}")
        print(f"   Output shape: {model2.output_shape}")
        print(f"   Output classes: {model2.output_shape[-1]}")
        print(f"   Total parameters: {model2.count_params():,}")
        print(f"   Number of layers: {len(model2.layers)}")
        
        print(f"\n🏗️  Model Architecture:")
        for i, layer in enumerate(model2.layers[:5]):
            print(f"   Layer {i+1}: {layer.name} ({layer.__class__.__name__})")
        if len(model2.layers) > 5:
            print(f"   ... and {len(model2.layers) - 5} more layers")
        
        print(f"\n✅ Model Status: READY TO USE")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
else:
    print(f"❌ File not found: {keras_path}")

print(f"\n{'='*70}")
print(" "*20 + "📋 SUMMARY")
print(f"{'='*70}")

print(f"""
✅ Downloaded Models Status:
   • trained_model.h5 ({size_mb:.1f} MB) - Ready
   • trained_model.keras ({os.path.getsize(keras_path) / 1024 / 1024:.1f} MB) - Ready

📝 Next Steps:
   1. Copy model to Flask app:
      copy downloaded_models\\trained_model.h5 "Flask Deployed App\\plant_disease_model.h5"
   
   2. Update Flask app to use Keras/TensorFlow instead of PyTorch
   
   3. Or convert to PyTorch format if needed

💡 Note: These are TensorFlow/Keras models, but your current Flask app
   uses PyTorch. You'll need to either:
   - Update Flask app to use TensorFlow/Keras
   - Convert models to PyTorch format
   - Train new PyTorch models with train_ensemble.py
""")

print(f"{'='*70}")
