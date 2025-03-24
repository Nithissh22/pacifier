from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
import logging
import os
import datetime

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Enable logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# File paths
MODEL_FILE = "models/hydration_model.pkl"
SCALER_FILE = "models/scaler.pkl"
HISTORY_FILE = "models/hydration_history.csv"

# Load AI Model & Scaler
model, scaler = None, None

if os.path.exists(MODEL_FILE) and os.path.exists(SCALER_FILE):
    try:
        model = joblib.load(MODEL_FILE)
        scaler = joblib.load(SCALER_FILE)
        logging.info("✅ Model & Scaler loaded successfully.")
    except Exception as e:
        logging.error(f"⚠️ Failed to load model or scaler: {str(e)}")
else:
    logging.warning("⚠️ Model or Scaler file not found. Predictions will not work.")

@app.route('/')
def home():
    return jsonify({"message": "Welcome to YourPacifier API!"})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        humidity = data.get("humidity")
        weight = data.get("weight")
        age = data.get("age")

        # Validate Inputs
        if humidity is None or weight is None or age is None:
            return jsonify({"error": "⚠️ Missing humidity, weight, or age values"}), 400
        
        try:
            humidity = float(humidity)
            weight = float(weight)
            age = int(age)
        except ValueError:
            return jsonify({"error": "⚠️ Invalid input types. Ensure humidity & weight are numbers, age is an integer."}), 400

        if not model or not scaler:
            return jsonify({"error": "⚠️ AI Model or Scaler not found"}), 500

        # Convert & scale input
        input_data = np.array([[humidity, weight, age]])
        input_scaled = scaler.transform(input_data)

        # Make prediction
        prediction = model.predict(input_scaled)[0]
        status = "Dehydrated" if prediction == 1 else "Normal"
        advice = "💧 Drink more water!" if status == "Dehydrated" else "✅ Hydration is good."
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Save history
        history_entry = pd.DataFrame([[timestamp, humidity, weight, age, status]], 
                                     columns=["Timestamp", "Humidity", "Weight", "Age", "Hydration_Status"])
        
        # Append without adding duplicate headers
        history_entry.to_csv(HISTORY_FILE, mode='a', header=not os.path.exists(HISTORY_FILE) or os.stat(HISTORY_FILE).st_size == 0, index=False)

        logging.info(f"Prediction - {timestamp}, Humidity: {humidity}, Weight: {weight}, Age: {age}, Status: {status}")

        return jsonify({
            "timestamp": timestamp,
            "hydration_status": status,
            "advice": advice
        })
    except Exception as e:
        logging.error(f"❌ Error in /predict: {str(e)}")
        return jsonify({"error": f"⚠️ Server error occurred: {str(e)}"}), 500

@app.route('/hydration-data', methods=['GET'])
def get_hydration_data():
    try:
        if not os.path.exists(HISTORY_FILE) or os.stat(HISTORY_FILE).st_size == 0:
            return jsonify([])  # Return an empty list if no data exists

        history_data = pd.read_csv(HISTORY_FILE)
        return jsonify(history_data.to_dict(orient="records"))
    except Exception as e:
        logging.error(f"❌ Error in /hydration-data: {str(e)}")
        return jsonify({"error": "⚠️ Could not retrieve hydration data"}), 500

@app.route('/history/download', methods=['GET'])
def download_history():
    try:
        if os.path.exists(HISTORY_FILE) and os.stat(HISTORY_FILE).st_size > 0:
            return send_file(HISTORY_FILE, as_attachment=True)
        return jsonify({"error": "⚠️ No history available"}), 404
    except Exception as e:
        logging.error(f"❌ Error in /history/download: {str(e)}")
        return jsonify({"error": "⚠️ Could not download history"}), 500

@app.route('/history/reset', methods=['POST'])
def reset_history():
    try:
        if os.path.exists(HISTORY_FILE):
            os.remove(HISTORY_FILE)
            logging.info("✅ Hydration history reset successfully.")
            return jsonify({"message": "✅ History reset successfully."}), 200
        return jsonify({"error": "⚠️ No history to reset"}), 404
    except Exception as e:
        logging.error(f"❌ Error in /history/reset: {str(e)}")
        return jsonify({"error": "⚠️ Could not reset history"}), 500

if __name__ == '__main__':
    app.run(debug=True)
