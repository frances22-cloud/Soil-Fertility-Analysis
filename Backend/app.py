import requests
import pandas as pd
import joblib
import folium
from sklearn.preprocessing import StandardScaler

# Load the trained model and scaler
model = joblib.load('C:/Users/frances2021/Desktop/Project/Models/soil_fertility_model.pkl')
scaler = joblib.load('C:/Users/frances2021/Desktop/Project/Models/soil_scaler.pkl') 

def get_soil_data(lat, lon):
    """
    Fetch soil properties from SoilGrids API for the given latitude and longitude.
    """
    url = f"https://rest.isric.org/soilgrids/v2.0/properties/query?lon={lon}&lat={lat}&properties=nitrogen,phh2o,soc,sand,silt,clay"

    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        # Extract soil properties from the API response
        soil_properties = {
            "N": data['properties']['layers'][0]['depths'][0]['values']['mean'] / 10.0,  # Convert to appropriate scale
            "ph": data['properties']['layers'][1]['depths'][0]['values']['mean'] / 10.0,  # Convert to appropriate scale
            "oc": data['properties']['layers'][2]['depths'][0]['values']['mean'] / 10.0,  # Convert to appropriate scale
            "Sand": data['properties']['layers'][3]['depths'][0]['values']['mean'] / 10.0,
            "Silt": data['properties']['layers'][4]['depths'][0]['values']['mean'] / 10.0,
            "Clay": data['properties']['layers'][5]['depths'][0]['values']['mean'] / 10.0
        }
        return soil_properties

    except Exception as e:
        print(f"Error fetching soil data: {e}")
        return None

def preprocess_and_predict(soil_data):
    """
    Preprocess soil data and predict soil fertility using the trained model.
    """
    if not soil_data:
        return "Error: No valid soil data available"

    # Check if all required keys are present and have non-None values
    required_keys = ['N', 'ph', 'oc']
    if not all(soil_data.get(key) is not None for key in required_keys):
        print("Missing or None values in soil data")
        return "Error: Incomplete soil data"

    # Convert soil data to DataFrame with appropriate feature order
    input_data = pd.DataFrame([[
        soil_data.get("N", 0), 0, 0, soil_data.get("ph", 0), 0, 
        soil_data.get("oc", 0), 0, 0, 0, 0, 0, 0
    ]], columns=['N', 'P', 'K', 'ph', 'ec', 'oc', 'S', 'zn', 'fe', 'cu', 'Mn', 'B'])

    # Scale the input data
    scaled_input = scaler.transform(input_data)

    # Predict soil fertility
    prediction = model.predict(scaled_input)[0]
    return prediction

# ... rest of the code remains the same

def map_soil_fertility(prediction):
    """
    Map soil fertility prediction to a status and recommended crops.
    """
    fertility_mapping = {
        0: {"status": "Low Soil Fertility", 
            "crops": ["Millets", "Sorghum", "Legumes", "Cassava"]},
        1: {"status": "Moderate Soil Fertility", 
            "crops": ["Maize", "Wheat", "Barley", "Groundnuts"]},
        2: {"status": "Highly Fertile Soil", 
            "crops": ["Rice", "Sugarcane", "Vegetables", "Fruits"]}
    }

    fertility_info = fertility_mapping.get(prediction, {"status": "Unknown", "crops": []})
    return fertility_info

def display_map(lat, lon, soil_data, fertility_info):
    """
    Display a map with soil data and fertility status.
    """
    # Create a map centered at the given latitude and longitude
    m = folium.Map(location=[lat, lon], zoom_start=10)

    # Create a popup message with soil data and fertility status
    popup_message = f"""
    <b>Soil Properties</b><br>
    Nitrogen (N): {soil_data['N']}<br>
    pH: {soil_data['ph']}<br>
    Organic Carbon (OC): {soil_data['oc']}<br>
    Clay: {soil_data['Clay']}<br>
    Sand: {soil_data['Sand']}<br>
    Silt: {soil_data['Silt']}<br>
    <br><b>Fertility Status:</b> {fertility_info['status']}<br>
    Recommended Crops: {', '.join(fertility_info['crops'])}
    """
    folium.Marker([lat, lon], popup=popup_message).add_to(m)

    # Return the map
    return m

def main():
    # Step 1: User input for latitude and longitude
    lat = float(input("Enter latitude: "))
    lon = float(input("Enter longitude: "))

    # Step 2: Fetch soil data from SoilGrids
    print("Fetching soil data...")
    soil_data = get_soil_data(lat, lon)
    if not soil_data:
        print("Failed to fetch soil data. Exiting...")
        return

    print("Soil Data:", soil_data)

    # Step 3: Predict soil fertility
    print("Predicting soil fertility...")
    prediction = preprocess_and_predict(soil_data)
    fertility_info = map_soil_fertility(prediction)

    print(f"\nSoil Fertility Status: {fertility_info['status']}")
    print("Recommended Crops:")
    for crop in fertility_info['crops']:
        print(f"- {crop}")

    # Step 4: Display map with soil properties and recommendations
    print("Generating map...")
    soil_map = display_map(lat, lon, soil_data, fertility_info)
    soil_map.save("soil_fertility_map.html")
    print("Map saved as 'soil_fertility_map.html'.")

if __name__ == "__main__":
    main()
