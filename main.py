import os
import requests
import pandas as pd
import joblib
import folium
from flask import Flask, render_template, request, jsonify
from sklearn.preprocessing import StandardScaler

# Initialize Flask app
app = Flask(__name__)

# Paths to the model and scaler
MODEL_PATH = r"C:/Users/frances2021/Desktop/Project/Models/soil_fertility_model.pkl"
SCALER_PATH = r"C:/Users/frances2021/Desktop/Project/Models/soil_scaler.pkl"

# Load the trained model and scaler
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
if not os.path.exists(SCALER_PATH):
    raise FileNotFoundError(f"Scaler file not found at {SCALER_PATH}")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

def map_soil_fertility(prediction):
    """
    Map the predicted class to soil fertility status and recommended crops.
    """
    mapping = {
        0: {
            "status": "Low fertility",
            "crops": ["Millet", "Cassava", "Sorghum"],
            "insights": "Consider adding organic matter and fertilizers to improve fertility."
        },
        1: {
            "status": "Moderate fertility",
            "crops": ["Maize", "Beans", "Potatoes"],
            "insights": "Supplement with balanced nutrients for optimal yields."
        },
        2: {
            "status": "High fertility",
            "crops": ["Wheat", "Sugarcane", "Tomatoes"],
            "insights": "Maintain current practices to sustain fertility."
        }
    }
    return mapping.get(prediction, {"status": "Unknown", "crops": [], "insights": "No insights available."})

def get_soil_data(lat, lon):
    """
    Fetch soil properties from SoilGrids API for the given latitude and longitude.
    Extract data specifically for the 0-5 cm depth range.
    """
    url = f"https://rest.isric.org/soilgrids/v2.0/properties/query?lon={lon}&lat={lat}&properties=nitrogen,phh2o,soc,sand,silt,clay,cec,bdod"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        # Extract 0-5 cm depth range data
        soil_properties = {}
        layers = {layer['name']: layer for layer in data['properties']['layers']}
        for key, layer in layers.items():
            depth_data = next((d for d in layer['depths'] if d['label'] == '0-5cm'), None)
            if depth_data:
                soil_properties[key] = depth_data['values'].get('mean', 0) / 10.0  # Adjusted scale if necessary

        # Map to desired keys - ensuring consistent naming
        return {
            "N": soil_properties.get('nitrogen', 0),
            "ph": soil_properties.get('phh2o', 0),
            "sand": soil_properties.get('sand', 0),
            "silt": soil_properties.get('silt', 0),
            "cec": soil_properties.get('cec', 0),
            "bulk_density": soil_properties.get('bdod', 0), 
            "clay": soil_properties.get('clay', 0),
            "organic_carbon": soil_properties.get('soc', 0) 
        }
    except Exception as e:
        print(f"Error fetching soil data: {e}")
        return None

def preprocess_and_predict(soil_data):
    """
    Preprocess soil data and predict soil fertility using the trained model.
    """
    if not soil_data:
        return "Error: No valid soil data available"

    # Convert soil data to DataFrame with correct order and consistent naming
    input_data = pd.DataFrame([[
        soil_data["N"],
        soil_data["ph"],
        soil_data["sand"],
        soil_data["silt"],
        soil_data["cec"],
        soil_data["bulk_density"],
        soil_data["clay"],
        soil_data["organic_carbon"]
    ]], columns=['N', 'ph', 'sand', 'silt', 'cec', 'bulk density', 'clay', 'soc'])

    # Scale the input data
    scaled_input = scaler.transform(input_data)

    # Predict soil fertility
    return model.predict(scaled_input)[0]

def create_map(lat, lon, soil_data, fertility_info):
    """
    Create a folium map with soil data and fertility status.
    """
    m = folium.Map(location=[lat, lon], zoom_start=10)
    popup_message = f"""
    <b>Soil Properties</b><br>
    N: {soil_data['N']}<br>
    pH: {soil_data['ph']}<br>
    Sand: {soil_data['sand']}<br>
    Silt: {soil_data['silt']}<br>
    CEC: {soil_data['cec']}<br>
    Bulk Density: {soil_data['bulk_density']}<br>
    Clay: {soil_data['clay']}<br>
    Organic Carbon: {soil_data['organic_carbon']}<br>
    <b>Status:</b> {fertility_info['status']}<br>
    <b>Crops:</b> {', '.join(fertility_info['crops'])}<br>
    <b>Insights:</b> {fertility_info['insights']}
    """
    folium.Marker([lat, lon], popup=popup_message).add_to(m)
    static_dir = os.path.join(os.getcwd(), 'static')
    os.makedirs(static_dir, exist_ok=True)
    map_path = os.path.join(static_dir, 'soil_map.html')
    m.save(map_path)
    return map_path

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        lat = float(data.get('latitude', 0))
        lon = float(data.get('longitude', 0))

        # Check if user provided soil properties
        user_soil_data = data.get('soil_properties', None)
        if user_soil_data:
            soil_data = {key: float(value) for key, value in user_soil_data.items()}
        else:
            # Fetch soil data if not provided
            soil_data = get_soil_data(lat, lon)

        print("Fetched/Provided Soil Data:", soil_data)  # Debug log

        if not soil_data:
            return jsonify({'error': 'Could not retrieve soil data for the given coordinates'}), 400

        # Predict soil fertility
        prediction = preprocess_and_predict(soil_data)
        print("Prediction:", prediction)  # Debug log

        fertility_info = map_soil_fertility(prediction)

        # Create a map
        map_path = create_map(lat, lon, soil_data, fertility_info)

        # Prepare response
        response = {
            'soil_data': soil_data,
            'fertility_status': fertility_info['status'],
            'recommended_crops': fertility_info['crops'],
            'insights': fertility_info['insights'],
            'map_path': '/static/soil_map.html',
            'latitude': lat,  # Include latitude
            'longitude': lon 
        }

        return jsonify(response)
    except Exception as e:
        print("Error in /predict:", str(e))  
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
