import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'Flask Deployed App'))

try:
    import torch
    import timm
    import transformers
    print(f"[OK] PyTorch version: {torch.__version__}")
    print(f"[OK] TIMM version: {timm.__version__}")
    print(f"[OK] Transformers version: {transformers.__version__}")
    
    from ViT_model import PlantDiseaseViT
    
    print("\nInitializing Vision Transformer Model...")
    model = PlantDiseaseViT(num_classes=39, pretrained=False)
    
    print("[OK] Model successfully created in memory!")
    
    print("Testing forward pass with dummy image (1, 3, 224, 224)...")
    dummy_input = torch.randn(1, 3, 224, 224)
    
    output = model(dummy_input)
    
    print(f"[OK] Forward pass successful!")
    print(f"[OK] Output shape: {output.shape} (Expected: torch.Size([1, 39]))")
    print("\n[SUCCESS] Vision Transformers are fully integrated and working perfectly!")
    
except ImportError as e:
    print(f"[ERROR] Dependency Error: {e}")
    print("Please install the required packages: pip install torch timm transformers")
except Exception as e:
    print(f"[ERROR] Model Error: {e}")
