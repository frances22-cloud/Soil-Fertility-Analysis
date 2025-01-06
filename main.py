import os
import requests
import pandas as pd
import joblib
from flask import Flask, render_template, request, jsonify, redirect, url_for, session
from flask_mysqldb import MySQL
import MySQLdb.cursors
import logging

# Initialize Flask app and configure logging
app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure MySQL database connection
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'project'
app.config['MYSQL_PORT'] = 3307
mysql = MySQL(app)

# Secret key for session management
app.secret_key = os.urandom(24)

# Paths to the model and scaler
MODEL_PATH = r"C:/Users/frances2021/Desktop/Project/Models/soil_fertility_model.pkl"
SCALER_PATH = r"C:/Users/frances2021/Desktop/Project/Models/soil_scaler.pkl"

# Load the trained model and scaler
try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
except FileNotFoundError as e:
    logger.error(f"Error loading model or scaler: {e}")
    raise

def map_soil_fertility(prediction):
    """
    Map the predicted class to soil fertility status and recommended crops.
    """
    mapping = {
        0: {
            "status": "Low fertility",
            "crops": ["Millet", "Cassava", "Sorghum", "Sweet Potatoes", "Taro", "Groundnuts", "Sunflower", "Pigeon Pea"],
            "insights": "Consider adding organic matter and fertilizers to improve fertility. Regular soil testing is recommended to track nutrient levels."
        },
        1: {
            "status": "Moderate fertility",
            "crops": ["Maize", "Beans", "Potatoes", "Cowpeas", "Chili Peppers", "Cabbage", "Tomatoes", "Soybeans"],
            "insights": "Supplement with balanced nutrients for optimal yields. Use cover crops and compost to maintain soil health."
        },
        2: {
            "status": "High fertility",
            "crops": ["Wheat", "Sugarcane", "Tomatoes", "Rice", "Coffee", "Cocoa", "Avocados", "Bananas", "Peppers"],
            "insights": "Maintain current practices to sustain fertility. Diversify crops to ensure long-term productivity and reduce soil erosion."
        }
    }
    return mapping.get(prediction, {"status": "Unknown", "crops": [], "insights": "No insights available."})

class SoilDataError(Exception):
    """Custom exception for soil data related errors"""
    pass

def get_soil_data(lat, lon):
    """
    Fetch soil properties from SoilGrids API for the given latitude and longitude.
    """
    url = f"https://rest.isric.org/soilgrids/v2.0/properties/query?lon={lon}&lat={lat}&properties=nitrogen,phh2o,soc,sand,silt,clay,cec,bdod"
    try:
        response = requests.get(url)
        response.raise_for_status()

        # Print raw response for debugging
        print("Raw API response:", response.text)

        data = response.json()

        # Extract 0-5 cm depth range data
        soil_properties = {}
        if 'properties' in data and 'layers' in data['properties']:
            layers = {layer['name']: layer for layer in data['properties']['layers']}
            for key, layer in layers.items():
                depth_data = next((d for d in layer['depths'] if d['label'] == '0-5cm'), None)
                if depth_data and 'values' in depth_data:
                    mean_value = depth_data['values'].get('mean')
                    # Safely handle None values
                    soil_properties[key] = (mean_value / 10.0) if mean_value is not None else 0
                else:
                    soil_properties[key] = 0
        else:
            # Handle case where soil data is missing
            raise ValueError("Soil data not found for the given coordinates.")

        return {
            "N": soil_properties.get('nitrogen', 0),
            "ph": soil_properties.get('phh2o', 0),
            "sand": soil_properties.get('sand', 0),
            "silt": soil_properties.get('silt', 0),
            "cec": soil_properties.get('cec', 0),
            "bulk density": soil_properties.get('bdod', 0),
            "clay": soil_properties.get('clay', 0),
            "soc": soil_properties.get('soc', 0)
        }
    except requests.exceptions.RequestException as e:
        print(f"Error fetching soil data: {e}")
        return None
    except ValueError as e:
        print(f"Error: {e}")
        return None


def preprocess_and_predict(soil_data):
    """
    Preprocess soil data and predict soil fertility using the trained model.
    """
    if not soil_data:
        return "Error: No valid soil data available"

    input_data = pd.DataFrame([[
        soil_data["N"],
        soil_data["ph"],
        soil_data["sand"],
        soil_data["silt"],
        soil_data["cec"],
        soil_data["bulk density"],
        soil_data["clay"],
        soil_data["soc"]
    ]], columns=['N', 'ph', 'sand', 'silt', 'cec', 'bulk density', 'clay', 'soc'])

    scaled_input = scaler.transform(input_data)
    return model.predict(scaled_input)[0]

@app.route('/')
def home():
    if 'user' in session:
        return redirect(url_for('index'))
    return render_template('home.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    message = ''
    if request.method == 'POST' and 'email' in request.form and 'password' in request.form:
        email = request.form['email']
        password = request.form['password']
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM user WHERE email = %s AND password = %s', (email, password))
        user = cursor.fetchone()
        if user:
            session['loggedin'] = True
            session['userid'] = user['userid']
            session['name'] = user['name']
            session['email'] = user['email']
            return redirect(url_for('index'))
        else:
            message = 'Invalid email or password!'
    return render_template('login.html', message=message)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    message = ''
    if request.method == 'POST' and 'name' in request.form and 'password' in request.form and 'email' in request.form:
        name = request.form['name']
        password = request.form['password']
        email = request.form['email']
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM user WHERE email = %s', (email,))
        account = cursor.fetchone()
        if account:
            message = 'Account already exists!'
        else:
            cursor.execute('INSERT INTO user VALUES (NULL, %s, %s, %s)', (name, email, password))
            mysql.connection.commit()
            message = 'Registration successful!'
    return render_template('register.html', message=message)

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if request is AJAX/API call or form submission
        is_api_request = request.headers.get('X-Requested-With') == 'XMLHttpRequest'
        
        # Get form/JSON data based on request type
        if request.is_json:
            data = request.get_json()
            lat = float(data.get('latitude', 0))
            lon = float(data.get('longitude', 0))
        else:
            lat = float(request.form.get('latitude', 0))
            lon = float(request.form.get('longitude', 0))


        # Get soil data from API
        soil_data = get_soil_data(lat, lon)
        if soil_data is None:
            raise ValueError("Could not retrieve soil data from SoilGrids API")

        # Make prediction
        prediction = preprocess_and_predict(soil_data)
        if prediction is None or isinstance(prediction, str):
            raise ValueError("Model prediction failed")

        fertility_info = map_soil_fertility(prediction)
        
        # Prepare response data
        response_data = {
            #'latitude': lat,
            #'longitude': lon,
            #'soil_data': soil_data,
            'fertility_info': fertility_info,
            'prediction': int(prediction),
            'fertility_status': fertility_info['status'],
            'recommended_crops': fertility_info['crops'],
            'insights': fertility_info['insights']
        }

        # Return response based on request type
        if is_api_request or request.is_json:
            return jsonify(response_data)
        else:
            return render_template('results.html', **response_data, error=None)

    except ValueError as ve:
        error_msg = str(ve)
        logger.error(f"Validation error: {error_msg}")
        if is_api_request or request.is_json:
            return jsonify({'error': error_msg}), 400
        return render_template('results.html', error=error_msg)
        
    except Exception as e:
        error_msg = f"An unexpected error occurred: {str(e)}"
        logger.error(f"Unexpected error: {error_msg}")
        if is_api_request or request.is_json:
            return jsonify({'error': error_msg}), 500
        return render_template('results.html', error=error_msg)
    
    @app.route('/get_soil_data')
    def get_soil_data_route():
     try:
        lat = float(request.args.get('latitude'))
        lon = float(request.args.get('longitude'))
        
        soil_data = get_soil_data(lat, lon)
        if soil_data is None:
            return jsonify({'error': 'Could not retrieve soil data'}), 400
            
        return jsonify(soil_data)
        
     except Exception as e:
        logger.error(f"Error in get_soil_data_route: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)