# 🌿 Plant Disease Detection using Deep Learning

A deep learning-based web application that detects plant diseases from leaf images using Convolutional Neural Networks (CNN). The system can classify **39 different plant disease categories** with high accuracy.

## 📌 Project Overview

Plant disease detection is crucial for farmers to take timely action and prevent crop loss. This project uses a CNN model trained on the PlantVillage dataset to identify diseases from leaf images.

### Key Features
- 🎯 **39 Disease Classes** - Comprehensive coverage of common plant diseases
- 🧠 **Deep Learning Model** - CNN built with PyTorch framework
- 🌐 **Web Interface** - User-friendly Flask web application
- 📊 **High Accuracy** - Achieves 97%+ accuracy on test data
- 🔍 **Ensemble Support** - Multiple model ensemble for improved predictions

## 🚀 Quick Start (Complete Guide)

### ⚠️ IMPORTANT: Before Cloning
This repository contains extremely large Deep Learning models (>200MB) which are tracked using **Git LFS (Large File Storage)**. 
**You MUST have Git LFS installed before you clone the repository**, otherwise the models will be downloaded as broken 1KB pointer files!

1. Download and install Git LFS from: https://git-lfs.github.com/
2. Open your terminal and run this once to activate it:
   ```bash
   git lfs install
   ```

### 💻 Installation Steps

1. **Clone the repository** (Make sure Git LFS is installed first!)
   ```bash
   git clone https://github.com/divyaredii/Plant-disease-detection--.git
   cd Plant-disease-detection--
   ```

2. **Create and activate a virtual environment** (Highly Recommended)
   ```bash
   python -m venv venv
   
   # For Windows users:
   venv\Scripts\activate
   
   # For Linux/Mac users:
   source venv/bin/activate
   ```

3. **Install all required dependencies**
   ```bash
   # This will install PyTorch, Flask, and the new Vision Transformer (timm/transformers) libraries
   pip install -r requirements.txt
   ```

### 🏃‍♂️ How to Run the Web Application

We now support **two different deep learning engines**. You can run whichever version you prefer:

**Option A: Run the new, highly-advanced Vision Transformer (ViT) model**
*(Recommended for highest accuracy & state-of-the-art AI)*
```bash
cd "Flask Deployed App"
python app_vit.py
```

**Option B: Run the original Convolutional Neural Network (CNN) model**
```bash
cd "Flask Deployed App"
python app.py
```

### 🌍 Open the app
After running either of the commands above, the application will boot up locally.
Open your web browser and go to: **[http://localhost:5000](http://localhost:5000)**

## 📁 Project Structure

```
Plant-Disease-Detection/
├── Flask Deployed App/     # Web application
│   ├── app.py              # Main Flask application
│   ├── templates/          # HTML templates
│   ├── static/             # CSS, JS, images
│   └── requirements.txt    # Dependencies
├── Model/                  # Trained model files
├── test_images/            # Sample images for testing
├── demo_images/            # Screenshots of the app
├── ARCHITECTURE.md         # System architecture diagram
├── requirements.txt        # Project dependencies
└── README.md               # This file
```

## 🏗️ System Architecture

The system uses a multi-model ensemble approach:

```
User Image → Preprocessing → CNN Model(s) → Prediction → Disease Info
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed diagrams.

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| Training Accuracy | 98.5% |
| Validation Accuracy | 97.2% |
| Test Accuracy | 97.0% |
| Number of Classes | 39 |
| Input Size | 224x224 |

## 🧪 Testing

Use the images in `test_images/` folder to test the model:
- Each image is named with its corresponding disease
- Upload any image through the web interface

## 📸 Screenshots

### Main Page
![Main Page](demo_images/1.png)

### AI Engine
![AI Engine](demo_images/2.png)

### Results Page
![Results](demo_images/3.png)

## 🛠️ Technologies Used

- **Deep Learning**: PyTorch, TorchVision
- **Web Framework**: Flask
- **Frontend**: HTML5, CSS3, JavaScript
- **Data Processing**: NumPy, Pandas, Pillow
- **Deployment**: Gunicorn

## 📚 Dataset

The model is trained on the [PlantVillage Dataset](https://www.kaggle.com/datasets/emmarex/plantdisease) containing 70,000+ images of plant leaves.

## 🤝 Contributing

Contributions are welcome! Please follow these steps:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📝 License

This project is open source and available under the MIT License.

## 📧 Contact

For questions or suggestions, please open an issue on GitHub.

---

**Made with ❤️ for farmers and agricultural researchers**
