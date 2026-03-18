"""
Quick Training Starter
This script checks dataset status and starts training automatically
"""

import os
import sys
from pathlib import Path
import time

def check_dataset_status():
    """Check if dataset is ready"""
    print("="*70)
    print("CHECKING DATASET STATUS")
    print("="*70)
    
    dataset_path = Path("New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)")
    train_path = dataset_path / "train"
    valid_path = dataset_path / "valid"
    
    if not dataset_path.exists():
        print("\n❌ Dataset folder not found")
        print("📥 Dataset is still downloading...")
        return False, "downloading"
    
    if not train_path.exists() or not valid_path.exists():
        print("\n❌ Dataset incomplete")
        print("📥 Still extracting or downloading...")
        return False, "extracting"
    
    # Count classes
    try:
        train_classes = len([d for d in train_path.iterdir() if d.is_dir()])
        valid_classes = len([d for d in valid_path.iterdir() if d.is_dir()])
        
        if train_classes < 39 or valid_classes < 39:
            print(f"\n⚠️  Dataset incomplete: {train_classes}/39 train classes, {valid_classes}/39 valid classes")
            return False, "incomplete"
        
        # Count images
        total_train = sum(len(list((train_path / d).glob('*'))) for d in train_path.iterdir() if d.is_dir())
        total_valid = sum(len(list((valid_path / d).glob('*'))) for d in valid_path.iterdir() if d.is_dir())
        
        print(f"\n✅ Dataset Ready!")
        print(f"   Training classes: {train_classes}")
        print(f"   Validation classes: {valid_classes}")
        print(f"   Training images: {total_train:,}")
        print(f"   Validation images: {total_valid:,}")
        
        return True, "ready"
    except Exception as e:
        print(f"\n❌ Error checking dataset: {e}")
        return False, "error"

def start_training(model_type="resnet50"):
    """Start training with the specified model"""
    print("\n" + "="*70)
    print(f"STARTING TRAINING - {model_type.upper()}")
    print("="*70)
    
    if model_type == "resnet50":
        script = "train_resnet50.py"
        print("\n🚀 Training ResNet50 (Best Accuracy: 97-99%)")
    else:
        script = "train_model.py"
        print("\n🚀 Training Basic CNN")
    
    print(f"\nExecuting: python {script}")
    print("\nThis will take:")
    print("  • GPU: 1-2 hours")
    print("  • CPU: 8-12 hours")
    print("\nYou can monitor progress in the terminal...")
    print("="*70 + "\n")
    
    # Start training
    os.system(f"python {script}")

def main():
    """Main function"""
    print("\n" + "="*70)
    print(" "*15 + "AUTOMATED TRAINING STARTER")
    print("="*70 + "\n")
    
    # Check dataset
    is_ready, status = check_dataset_status()
    
    if not is_ready:
        print("\n" + "="*70)
        print("DATASET NOT READY YET")
        print("="*70)
        
        if status == "downloading":
            print("\n📥 The dataset is currently downloading...")
            print("   This is a large file (~800MB) and may take 10-30 minutes")
            print("   depending on your internet speed.")
        elif status == "extracting":
            print("\n📦 The dataset is being extracted...")
            print("   This may take a few minutes.")
        elif status == "incomplete":
            print("\n⚠️  The dataset is incomplete.")
            print("   Please wait for the download to finish.")
        
        print("\n💡 OPTIONS:")
        print("   1. Wait for download_dataset.py to complete")
        print("   2. Check download status: python check_system.py")
        print("   3. Manual download from:")
        print("      https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset")
        
        print("\n⏰ Check back in 10-15 minutes and run this script again:")
        print("   python start_training.py")
        
        return
    
    # Dataset is ready - ask user which model to train
    print("\n" + "="*70)
    print("DATASET READY - CHOOSE MODEL")
    print("="*70)
    
    print("\nAvailable models:")
    print("  1. ResNet50 (RECOMMENDED)")
    print("     • Accuracy: 97-99%")
    print("     • Training time: 1-2 hours (GPU) / 8-12 hours (CPU)")
    print("     • Best for production")
    
    print("\n  2. Basic CNN")
    print("     • Accuracy: 92-95%")
    print("     • Training time: 2-3 hours (GPU) / 10-15 hours (CPU)")
    print("     • Good for baseline")
    
    print("\n" + "="*70)
    
    choice = input("\nEnter choice (1 or 2, default=1): ").strip()
    
    if choice == "2":
        start_training("basic")
    else:
        start_training("resnet50")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
