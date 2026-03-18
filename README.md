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

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/Plant-Disease-Detection.git
   cd Plant-Disease-Detection
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   cd "Flask Deployed App"
   python app.py
   ```

5. **Open in browser**
   ```
   http://localhost:5000
   ```

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
