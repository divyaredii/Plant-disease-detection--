"""
Quick Dataset Download Script
Downloads the New Plant Diseases Dataset from Kaggle
"""

import os
import subprocess
import sys
import zipfile
from pathlib import Path

def check_kaggle_installed():
    """Check if Kaggle CLI is installed"""
    try:
        subprocess.run(['kaggle', '--version'], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def install_kaggle():
    """Install Kaggle CLI"""
    print("Installing Kaggle CLI...")
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'kaggle'], check=True)
    print("✓ Kaggle CLI installed successfully!")

def check_kaggle_credentials():
    """Check if Kaggle API credentials are configured"""
    kaggle_dir = Path.home() / '.kaggle'
    kaggle_json = kaggle_dir / 'kaggle.json'
    
    if kaggle_json.exists():
        return True
    
    print("\n❌ Kaggle API credentials not found!")
    print("\nTo setup Kaggle API credentials:")
    print("1. Go to https://www.kaggle.com/account")
    print("2. Scroll to 'API' section")
    print("3. Click 'Create New API Token'")
    print("4. This will download 'kaggle.json'")
    print(f"5. Move it to: {kaggle_dir}")
    print("\nOn Windows, run these commands:")
    print(f"   mkdir {kaggle_dir}")
    print(f"   move Downloads\\kaggle.json {kaggle_json}")
    
    return False

def download_dataset():
    """Download the dataset from Kaggle"""
    dataset_name = "vipoooool/new-plant-diseases-dataset"
    
    print(f"\nDownloading dataset: {dataset_name}")
    print("This may take several minutes depending on your internet speed...")
    
    try:
        subprocess.run([
            'kaggle', 'datasets', 'download',
            '-d', dataset_name
        ], check=True)
        print("✓ Dataset downloaded successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error downloading dataset: {e}")
        return False

def extract_dataset():
    """Extract the downloaded dataset"""
    zip_file = "new-plant-diseases-dataset.zip"
    
    if not os.path.exists(zip_file):
        print(f"❌ ZIP file not found: {zip_file}")
        return False
    
    print(f"\nExtracting {zip_file}...")
    try:
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall('.')
        print("✓ Dataset extracted successfully!")
        
        # Remove zip file to save space
        print("\nRemoving ZIP file to save space...")
        os.remove(zip_file)
        print("✓ ZIP file removed!")
        
        return True
    except Exception as e:
        print(f"❌ Error extracting dataset: {e}")
        return False

def verify_dataset():
    """Verify that the dataset was extracted correctly"""
    expected_path = Path("New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)")
    train_path = expected_path / "train"
    valid_path = expected_path / "valid"
    
    if train_path.exists() and valid_path.exists():
        # Count classes
        train_classes = len([d for d in train_path.iterdir() if d.is_dir()])
        valid_classes = len([d for d in valid_path.iterdir() if d.is_dir()])
        
        print(f"\n✓ Dataset verified!")
        print(f"  Training classes: {train_classes}")
        print(f"  Validation classes: {valid_classes}")
        print(f"  Dataset location: {expected_path.absolute()}")
        return True
    else:
        print(f"\n❌ Dataset structure not found at expected location: {expected_path}")
        return False

def main():
    """Main function"""
    print("="*60)
    print("Plant Disease Dataset Downloader")
    print("="*60)
    
    # Step 1: Check and install Kaggle CLI
    if not check_kaggle_installed():
        print("\nKaggle CLI not found. Installing...")
        try:
            install_kaggle()
        except Exception as e:
            print(f"❌ Failed to install Kaggle CLI: {e}")
            print("\nPlease install manually:")
            print("  pip install kaggle")
            return
    else:
        print("\n✓ Kaggle CLI is installed")
    
    # Step 2: Check Kaggle credentials
    if not check_kaggle_credentials():
        return
    
    print("\n✓ Kaggle API credentials found")
    
    # Step 3: Download dataset
    print("\n" + "="*60)
    print("Downloading Dataset")
    print("="*60)
    
    if not download_dataset():
        print("\n❌ Failed to download dataset")
        print("\nAlternative: Download manually from:")
        print("https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset")
        return
    
    # Step 4: Extract dataset
    print("\n" + "="*60)
    print("Extracting Dataset")
    print("="*60)
    
    if not extract_dataset():
        print("\n❌ Failed to extract dataset")
        return
    
    # Step 5: Verify dataset
    print("\n" + "="*60)
    print("Verifying Dataset")
    print("="*60)
    
    if verify_dataset():
        print("\n" + "="*60)
        print("✓ Setup Complete!")
        print("="*60)
        print("\nYou can now run the training script:")
        print("  python train_model.py")
    else:
        print("\n❌ Dataset verification failed")
        print("Please check the extracted files manually")

if __name__ == "__main__":
    main()
