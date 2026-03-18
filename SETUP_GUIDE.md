# 🚀 Project Setup Guide

## Plant Disease Detection - Complete Setup Instructions

Follow these steps to run the project on your computer.

---

## 📋 Prerequisites

Before starting, make sure you have:

| Requirement | Version | Download Link |
|-------------|---------|---------------|
| Python | 3.8 or higher | [python.org/downloads](https://www.python.org/downloads/) |
| pip | Latest | Comes with Python |
| Git (optional) | Any | [git-scm.com](https://git-scm.com/) |

### Check if Python is installed:
```bash
python --version
```
You should see something like `Python 3.8.x` or higher.

---

## 📥 Step 1: Extract the Project

1. Extract the ZIP file to a folder (e.g., `C:\Projects\Plant-Disease-Detection`)
2. Open **Command Prompt** or **PowerShell**
3. Navigate to the project folder:
   ```bash
   cd C:\Projects\Plant-Disease-Detection
   ```

---

## 🔧 Step 2: Create Virtual Environment

A virtual environment keeps project dependencies separate from your system Python.

### Windows:
```bash
# Create virtual environment
python -m venv venv

# Activate it
venv\Scripts\activate
```

### Linux/Mac:
```bash
# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate
```

> ✅ You should see `(venv)` at the beginning of your command line.

---

## 📦 Step 3: Install Dependencies

Install all required Python packages:

```bash
pip install -r requirements.txt
```

This will install:
- Flask (web framework)
- PyTorch (deep learning)
- Pillow (image processing)
- NumPy, Pandas (data handling)

> ⏳ This may take 5-10 minutes depending on your internet speed.

---

## ▶️ Step 4: Run the Application

Navigate to the Flask app folder and start the server:

```bash
cd "Flask Deployed App"
python app.py
```

You should see output like:
```
 * Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)
```

---

## 🌐 Step 5: Open in Browser

Open your web browser and go to:

```
http://localhost:5000
```

or

```
http://127.0.0.1:5000
```

---

## 🧪 Step 6: Test the Application

1. Click on **"AI Engine"** or the upload section
2. Upload a plant leaf image from the `test_images` folder
3. Wait for the prediction
4. View the disease detection result!

### Sample test images location:
```
Plant-Disease-Detection/test_images/
```

---

## 🛑 How to Stop the Server

Press `Ctrl + C` in the command prompt to stop the server.

---

## ❓ Troubleshooting

### Problem: "python is not recognized"
**Solution:** Install Python and add it to PATH during installation.

### Problem: "pip install" fails
**Solution:** Try:
```bash
python -m pip install -r requirements.txt
```

### Problem: "ModuleNotFoundError: No module named 'torch'"
**Solution:** Make sure virtual environment is activated (you see `(venv)` in terminal).

### Problem: Port 5000 already in use
**Solution:** Edit `app.py` and change the port:
```python
app.run(debug=True, port=5001)
```

### Problem: Model file not found
**Solution:** Make sure `plant_disease_model_1_latest.pt` is in the `Flask Deployed App` folder.

---

## 📁 Project Structure

```
Plant-Disease-Detection/
│
├── Flask Deployed App/          # Main web application
│   ├── app.py                   # Flask server (RUN THIS)
│   ├── templates/               # HTML pages
│   ├── static/                  # CSS, JS, images
│   ├── plant_disease_model_1_latest.pt  # Trained model (210MB)
│   └── requirements.txt         # Dependencies
│
├── test_images/                 # Sample images for testing
├── demo_images/                 # Screenshots
├── ARCHITECTURE.md              # System architecture
├── README.md                    # Project overview
├── SETUP_GUIDE.md               # This file
└── requirements.txt             # Root dependencies
```

---

## 🎉 Success!

If you can see the web interface and upload images, you're all set!

For any issues, contact the project owner.

---

**Happy Testing! 🌿**
