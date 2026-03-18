"""
Download Trained Models from Google Drive
Downloads the pre-trained models from the shared Google Drive folder
"""

import os
import requests
from pathlib import Path

# Google Drive file IDs (extracted from the shared folder)
# You'll need to get individual file IDs for direct download
FOLDER_URL = "https://drive.google.com/drive/folders/1wGNFhdjxl7J1rljxicbArto9Z9AJT8WA"

# Model files found in the folder
MODEL_FILES = {
    'trained_model.h5': {
        'size': '89.9 MB',
        'type': 'HDF5 Model (Keras/TensorFlow)',
        'description': 'Main trained model in HDF5 format'
    },
    'trained_model.keras': {
        'size': '30 MB',
        'type': 'Keras Model',
        'description': 'Keras native format model'
    }
}

def download_from_gdrive(file_id, destination):
    """Download file from Google Drive"""
    URL = "https://docs.google.com/uc?export=download"
    
    session = requests.Session()
    response = session.get(URL, params={'id': file_id}, stream=True)
    token = get_confirm_token(response)
    
    if token:
        params = {'id': file_id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)
    
    save_response_content(response, destination)

def get_confirm_token(response):
    """Get confirmation token for large files"""
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None

def save_response_content(response, destination):
    """Save downloaded content to file"""
    CHUNK_SIZE = 32768
    
    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:
                f.write(chunk)

def main():
    print("="*70)
    print(" "*15 + "📥 DOWNLOAD TRAINED MODELS")
    print("="*70)
    
    print(f"\n📂 Google Drive Folder:")
    print(f"   {FOLDER_URL}")
    
    print(f"\n📋 Available Models:")
    for filename, info in MODEL_FILES.items():
        print(f"\n   • {filename}")
        print(f"     Size: {info['size']}")
        print(f"     Type: {info['type']}")
        print(f"     Description: {info['description']}")
    
    print("\n" + "="*70)
    print("⚠️  MANUAL DOWNLOAD REQUIRED")
    print("="*70)
    
    print("""
To download the models:

1. Open the Google Drive folder in your browser:
   https://drive.google.com/drive/folders/1wGNFhdjxl7J1rljxicbArto9Z9AJT8WA

2. Download the model files:
   
   Option A: Download individual files
   - Click on 'trained_model.h5' → Download
   - Click on 'trained_model.keras' → Download
   
   Option B: Download all files
   - Click the "Download all" button (top right)
   - This will download a zip file with all contents

3. Move the downloaded models to:
   Flask Deployed App/
   
4. Rename if needed:
   - trained_model.h5 → plant_disease_model.h5
   - Or update the Flask app to use the new filename

5. Test the model:
   python test_accuracy.py
""")
    
    print("\n" + "="*70)
    print("💡 ALTERNATIVE: Use gdown for automatic download")
    print("="*70)
    
    print("""
Install gdown:
   pip install gdown

Then run:
   gdown --folder https://drive.google.com/drive/folders/1wGNFhdjxl7J1rljxicbArto9Z9AJT8WA
""")
    
    print("\n" + "="*70)
    
    # Check if gdown is available
    try:
        import gdown
        print("\n✅ gdown is installed!")
        print("\nWould you like to download now? (This will download the entire folder)")
        
        response = input("Download now? (y/n): ").lower()
        if response == 'y':
            print("\n📥 Downloading from Google Drive...")
            output_dir = "downloaded_models"
            os.makedirs(output_dir, exist_ok=True)
            
            # Download the folder
            gdown.download_folder(FOLDER_URL, output=output_dir, quiet=False)
            
            print(f"\n✅ Download complete!")
            print(f"   Files saved to: {output_dir}/")
            print(f"\nNext steps:")
            print(f"   1. Copy model files to 'Flask Deployed App/'")
            print(f"   2. Test with: python test_accuracy.py")
        else:
            print("\n⏭️  Skipping download. Follow manual instructions above.")
            
    except ImportError:
        print("\n⚠️  gdown not installed")
        print("   Install with: pip install gdown")
        print("   Or follow manual download instructions above")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    main()
