from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import joblib
import numpy as np
import os

# Initialize Flask app
app = Flask(__name__, static_folder='.')
CORS(app)  # Allow frontend requests from any origin

# Load the trained Random Forest model
MODEL_PATH = "Best_Water_RandomForest_Model.pkl"
try:
    model = joblib.load(MODEL_PATH)
    print(f"✅ Model loaded successfully from {MODEL_PATH}")
except FileNotFoundError:
    print(f"❌ Error: Model file '{MODEL_PATH}' not found!")
    model = None

@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Predict water potability using the trained Random Forest model
    Expected JSON: {
        "ph": float,
        "hardness": float,
        "organic_carbon": float,
        "conductivity": float,
        "turbidity": float
    }
    """
    try:
        if model is None:
            return jsonify({"error": "Model not loaded"}), 500

        # Parse JSON data from frontend
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400

        # Extract parameters from request
        ph = float(data.get('ph', 0))
        hardness = float(data.get('hardness', 0))
        organic_carbon = float(data.get('organic_carbon', 0))
        conductivity = float(data.get('conductivity', 0))
        turbidity = float(data.get('turbidity', 0))

        # Validate input ranges
        if not (0 <= ph <= 14):
            return jsonify({"error": "pH must be between 0 and 14"}), 400
        if hardness < 0 or organic_carbon < 0 or conductivity < 0 or turbidity < 0:
            return jsonify({"error": "All parameters must be non-negative"}), 400

        # Prepare feature array in the same order as training data
        features = np.array([[ph, hardness, organic_carbon, conductivity, turbidity]])

        # Make prediction
        prediction = model.predict(features)[0]
        
        # Get probability of the predicted class
        probability = None
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(features)[0]
            probability = float(probabilities[int(prediction)])
        
        # Construct readable message
        if prediction == 1:
            message = "Water is POTABLE ✅"
            status = "safe"
        else:
            message = "Water is NOT POTABLE ⚠️"
            status = "unsafe"

        # Response JSON
        return jsonify({
            "prediction": int(prediction),
            "potable": bool(prediction == 1),
            "probability": round(probability, 4) if probability else None,
            "message": message,
            "status": status,
            "confidence": round(probability * 100, 2) if probability else None,
            "parameters": {
                "ph": ph,
                "hardness": hardness,
                "organic_carbon": organic_carbon,
                "conductivity": conductivity,
                "turbidity": turbidity
            }
        }), 200

    except ValueError as e:
        return jsonify({"error": f"Invalid parameter format: {str(e)}"}), 400
    except Exception as e:
        return jsonify({"error": f"Prediction error: {str(e)}"}), 500

# Additional routes
@app.route('/api/status', methods=['GET'])
def status():
    """Check if API and model are running"""
    return jsonify({
        "status": "running",
        "model_loaded": model is not None,
        "version": "1.0.0"
    }), 200

@app.route('/api/model-info', methods=['GET'])
def model_info():
    """Get information about the loaded model"""
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500
    
    return jsonify({
        "model_type": type(model).__name__,
        "parameters": {
            "n_estimators": getattr(model, 'n_estimators', 'N/A'),
            "max_depth": getattr(model, 'max_depth', 'N/A'),
        },
        "features": ["pH", "Hardness", "Organic Carbon", "Conductivity", "Turbidity"],
        "output_classes": ["Not Potable", "Potable"]
    }), 200

# Run the Flask app
if __name__ == '__main__':
    print("=" * 60)
    print("🌊 Water Quality Prediction API")
    print("=" * 60)
    print(f"Model Status: {'✅ Loaded' if model is not None else '❌ Not Loaded'}")
    print(f"API Endpoint: http://localhost:5000/api/predict")
    print(f"Status Check: http://localhost:5000/api/status")
    print("=" * 60)
    app.run(host="0.0.0.0", port=5000, debug=True)
