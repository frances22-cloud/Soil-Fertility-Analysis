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
MODEL_PATH = "Models/soil_fertility_model.pkl"
SCALER_PATH = "Models/soil_scaler.pkl"

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
            "insights": [
                "Add compost or manure to the soil and use NPK fertilizers to provide balanced nutrients for crops",
                "Cover the soil with mulch and reduce digging to save water and protect the soil.",
                "Grow beans or peas (legumes) after other crops to add natural fertilizer to the soil.",
                "Plant drought-resistant crops like millet or sorghum to ensure a good harvest with less water.",
                "Apply fertilizers in small amounts at different times to help crops absorb nutrients better.",
                "Build trenches, terraces, or small dams to collect and save rainwater for your farm.",
                "Test your soil every 2-3 years to know what it needs and improve it over time.",
                "Check soil pH regularly and add lime or sulfur to keep it balanced for better growth."
            ],
            "soil_indicators": [
                "Low nutrient content ie the soil lacks key nutrients for plants",
                "Poor water retention ie the soil dries out quickly.",
                "Limited organic matter ie the soil has little decomposed material",
                "The soil's pH is unbalanced"
            ],
            "management_practices": [
                "Use fertilizers like urea or DAP in smaller portions during planting and growth stages to avoid waste and improve nutrient uptake.",
                "Plant legumes such as clover, alfalfa, or cowpeas and mix them into the soil to increase nitrogen and organic matter",
                "Grow maize, beans, or sorghum along the natural curves of sloped land to reduce soil erosion and water loss",
                "Consider drip irrigation for water efficiency",
                "Use mulching to conserve moisture and reduce soil erosion",
                "Test the soil, and if it’s too acidic, add lime, or if it’s too alkaline, add sulfur to maintain a healthy range for crops."
            ]
        },
        1: {
            "status": "Moderately fertile soils",
            "crops": ["Maize", "Beans", "Potatoes", "Cowpeas", "Chili Peppers", "Cabbage", "Tomatoes", "Soybeans"],
            "insights": [
                "Add the right mix of nutrients like NPK to boost moderate soil productivity.",
                "Plant cover crops and add compost to improve soil structure and fertility.",
                "Rotate crops to prevent nutrient depletion and sustain soil fertility.",
                "Check crops for signs of nutrient deficiencies and adjust fertilization accordingly.",
                "Implement precision farming techniques where possible",
                "Apply irrigation carefully to ensure soil retains enough moisture.",
                "Control pests using a mix of biological and chemical methods",
                "Track fertilizer use to improve nutrient management over time."
            ],
            "soil_indicators": [
                "Soil has enough nutrients for moderate plant growth.",
                "Soil holds some water, but may need irrigation during dry periods.",
                "Soil has enough organic material, but may benefit from more input.",
                "Soil pH is usually suitable, but adjustments may be needed."
            ],
            "management_practices": [
                "Test soil regularly to monitor nutrient levels and pH.",
                "Apply fertilizers based on soil tests and crop needs.",
                "Grow compatible crops together for better nutrient use",
                "Use cover crops to improve fertility and soil health.",
                "Combine chemical and organic fertilizers for soil health."
            ]
        },
        2: {
            "status": "Highly fertile soils",
            "crops": ["Wheat", "Sugarcane", "Tomatoes", "Rice", "Coffee", "Cocoa", "Avocados", "Bananas", "Peppers"],
            "insights": [
                "Keep using balanced fertilizers and organic matter to preserve soil fertility.",
                "Grow a variety of crops to prevent soil depletion and pests.",
                "Test the soil regularly to avoid nutrient deficiencies or excesses.",
                "Use minimal tillage and cover crops to protect soil and improve water use.",
                "Use high-yielding crop varieties: Select improved seeds for better harvests and resilience.",
                "Adopt drip or sprinkler systems to efficiently water crops.",
                "Focus on soil biodiversity maintenance: Encourage beneficial organisms by adding organic matter.",
                "Consider value-added crop production: Process crops into products to boost income."
            ],
            "soil_indicators": [
                "Soil has a proper mix of essential nutrients for healthy plant growth.",
                "Soil holds enough water to support crops during dry periods.",
                "Soil contains decomposed plant and animal material, improving fertility and water retention.",
                "Soil pH is balanced, allowing crops to easily absorb nutrients."
            ],
            "management_practices": [
                "Monitor nutrient levels regularly",
                "Rotate crops to prevent nutrient depletion and control pests.",
                "Plant cover crops to improve soil fertility and reduce erosion.",
                "Space crops properly for efficient growth.",
                "Use practices like terracing and cover crops to protect soil."
            ]
        }
    }
    
    default_value = {
        "status": "Unknown",
        "crops": [],
        "insights": ["No insights available."],
        "soil_indicators": ["No soil indicators available."],
        "management_practices": ["No management practices available."]
    }
    
    return mapping.get(prediction, default_value)

def get_soil_data(lat, lon):
    """
    Fetch soil properties from SoilGrids API for the given latitude and longitude.
    """
    url = f"https://rest.isric.org/soilgrids/v2.0/properties/query?lon={lon}&lat={lat}&properties=nitrogen,phh2o,soc,sand,silt,clay,cec,bdod"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        if 'properties' not in data or 'layers' not in data['properties']:
            raise ValueError("Invalid response from SoilGrids API")

        soil_properties = {}
        layers = data['properties']['layers']
        
        for layer in layers:
            property_name = layer['name']
            depth_data = next((d for d in layer['depths'] if d['label'] == '0-5cm'), None)
            if depth_data and 'values' in depth_data:
                mean_value = depth_data['values'].get('mean')
                soil_properties[property_name] = (mean_value / 10.0) if mean_value is not None else 0

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
        logger.error(f"SoilGrids API request failed: {e}")
        return None
    except (KeyError, ValueError) as e:
        logger.error(f"Error processing soil data: {e}")
        return None

def preprocess_and_predict(soil_data):
    """
    Preprocess soil data and predict soil fertility using the trained model.
    """
    try:
        if not soil_data:
            raise ValueError("No soil data provided")

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
        prediction = model.predict(scaled_input)[0]
        return prediction
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return None

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
    if 'loggedin' not in session:
        return redirect(url_for('login'))
    return render_template('index.html')

@app.route('/get_soil_data')
def get_soil_data_route():
    try:
        lat = float(request.args.get('latitude'))
        lon = float(request.args.get('longitude'))
        
        soil_data = get_soil_data(lat, lon)
        if soil_data is None:
            return jsonify({'error': 'Could not retrieve soil data'}), 400
            
        return jsonify(soil_data)
    except (TypeError, ValueError) as e:
        return jsonify({'error': f'Invalid coordinates: {str(e)}'}), 400
    except Exception as e:
        logger.error(f"Error in get_soil_data_route: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json() if request.is_json else request.form.to_dict()
        
        # Validate input data
        required_fields = ['latitude', 'longitude']
        if not all(field in data for field in required_fields):
            raise ValueError("Missing required fields")

        lat = float(data['latitude'])
        lon = float(data['longitude'])

        # Get soil data from API if not provided
        soil_data = get_soil_data(lat, lon)
        if soil_data is None:
            raise ValueError("Could not retrieve soil data from SoilGrids API")

        # Make prediction
        prediction = preprocess_and_predict(soil_data)
        if prediction is None:
            raise ValueError("Error making prediction")

        # Get fertility information
        fertility_info = map_soil_fertility(prediction)
        
        response_data = {
            'prediction': int(prediction),
            'fertility_status': fertility_info['status'],
            'recommended_crops': fertility_info['crops'],
            'insights': fertility_info['insights'],
            'soil_indicators': fertility_info['soil_indicators'],
            'management_practices': fertility_info['management_practices']
        }

        return jsonify(response_data)

    except ValueError as ve:
        return jsonify({'error': str(ve)}), 400
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': 'An unexpected error occurred'}), 500
    
@app.route('/results', methods=['GET'])
def results():
    # Get the prediction from query parameters
    prediction = int(request.args.get('prediction', 0))
    
    # Get the soil analysis data using the mapping function
    soil_data = map_soil_fertility(prediction)
    
    # Format the data for the template
    data = {
        'fertility_status': soil_data['status'],
        'recommended_crops': soil_data['crops'],
        'insights': '\n'.join([f"- {insight}" for insight in soil_data['insights']]),
        'soil_indicators': soil_data['soil_indicators'],
        'management_practices': soil_data['management_practices']
    }
    
    return render_template('results.html', data=data)



if __name__ == '__main__':
    app.run(debug=True)
