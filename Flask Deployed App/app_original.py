import os
from flask import Flask, redirect, render_template, request, url_for
from PIL import Image
import torchvision.transforms.functional as TF
import CNN
import numpy as np
import torch
import pandas as pd
from crop_model import calculate_crop_suitability, get_recommendation_text

disease_info = pd.read_csv('disease_info.csv' , encoding='cp1252')
supplement_info = pd.read_csv('supplement_info.csv',encoding='cp1252')

model = CNN.CNN(39)    
model.load_state_dict(torch.load("plant_disease_model_1_latest.pt"))
model.eval()

def prediction(image_path):
    image = Image.open(image_path)
    image = image.convert('RGB')  # Convert to RGB to ensure 3 channels
    image = image.resize((224, 224))
    input_data = TF.to_tensor(image)
    # Add batch dimension and ensure correct shape
    input_data = input_data.unsqueeze(0)  # Add batch dimension
    output = model(input_data)
    output = output.detach().numpy()
    index = np.argmax(output)
    return index

def weather_fetch(city_name):
    """
    Fetch and returns the temperature and humidity of a city
    :params: city_name
    :return: temperature, humidity
    """
    api_key = config.weather_api_key
    base_url = "http://api.openweathermap.org/data/2.5/weather?"

    complete_url = base_url + "appid=" + api_key + "&q=" + city_name
    response = requests.get(complete_url)
    x = response.json()

    if x["cod"] != "404":
        y = x["main"]

        temperature = round((y["temp"] - 273.15), 2)
        humidity = y["humidity"]
        return temperature, humidity
    else:
        return None

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Change this to a secure random key in production

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/contact')
def contact():
    return render_template('contact-us.html')

@app.route('/index')
def ai_engine_page():
    return render_template('index.html')

@app.route('/mobile-device')
def mobile_device_detected_page():
    return render_template('mobile-device.html')

@app.route('/crop')
def crop():
    return render_template('crop.html')

@app.route('/crop_recommendation', methods=['POST'])
def crop_recommendation():
    try:
        # Get form data
        n = float(request.form['nitrogen'])
        p = float(request.form['phosphorous'])
        k = float(request.form['pottasium'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])
        state = request.form['state']
        city = request.form['city']

        # Calculate crop recommendations using the crop_model functions
        recommendations = calculate_crop_suitability(n, p, k)
        
        # Format the recommendations for display
        input_params = {'N': n, 'P': p, 'K': k}
        prediction_text = get_recommendation_text(recommendations, input_params)
        
        # Create a dictionary of recommended crops and their NPK values
        recommended_crops = {}
        for rec in recommendations:
            npk = f"N={rec['requirements']['N']}, P={rec['requirements']['P']}, K={rec['requirements']['K']}"
            recommended_crops[rec['crop']] = npk

        # Return the template with predictions and parameters
        return render_template('crop-result.html',
                             crops=list(recommended_crops.keys()),
                             npk_values=recommended_crops,
                             n=n, p=p, k=k,
                             prediction=prediction_text)

    except Exception as e:
        print(f"Error in crop_recommendation: {str(e)}")  # Add debug logging
        # If any error occurs, render the crop.html template with the error
        return render_template('crop.html', error=str(e))

@app.route('/submit', methods=['GET', 'POST'])
def submit():
    if request.method == 'POST':
        image = request.files['image']
        filename = image.filename
        file_path = os.path.join('static/uploads', filename)
        image.save(file_path)
        print(file_path)
        pred = prediction(file_path)
        title = disease_info['disease_name'][pred]
        description =disease_info['description'][pred]
        prevent = disease_info['Possible Steps'][pred]
        image_url = disease_info['image_url'][pred]
        supplement_name = supplement_info['supplement name'][pred]
        supplement_image_url = supplement_info['supplement image'][pred]
        supplement_buy_link = supplement_info['buy link'][pred]
        return render_template('submit.html' , title = title , desc = description , prevent = prevent , 
                               image_url = image_url , pred = pred ,sname = supplement_name , simage = supplement_image_url , buy_link = supplement_buy_link)

@app.route('/market', methods=['GET', 'POST'])
def market():
    return render_template('market.html', supplement_image = list(supplement_info['supplement image']),
                           supplement_name = list(supplement_info['supplement name']), disease = list(disease_info['disease_name']), buy = list(supplement_info['buy link']))

if __name__ == '__main__':
    app.run(debug=True)
