import numpy as np
import pandas as pd
import joblib
import os
import logging
import json
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Enable logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ✅ Sample training data (humidity, weight, age, dehydration_status)
data = {
    "humidity": [80, 75, 60, 50, 40, 35, 90, 85, 65, 45, 55, 70, 30, 20, 95],
    "weight": [3.5, 4.0, 5.0, 6.0, 4.5, 3.8, 6.2, 5.5, 4.2, 3.7, 4.8, 5.3, 3.6, 2.9, 7.0],
    "age": [2, 3, 4, 5, 3, 2, 6, 4, 3, 2, 5, 4, 1, 1, 6],  # ✅ Added "age" column
    "dehydration_status": [0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0]  # 0 = Normal, 1 = Dehydrated
}

# Convert to DataFrame
df = pd.DataFrame(data)

# ✅ Split data into training and testing sets
X = df[["humidity", "weight", "age"]]  # ✅ Now includes "age"
y = df["dehydration_status"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Normalize features (helps with model performance)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ✅ Train a Logistic Regression model with balanced class weights
try:
    model = LogisticRegression(class_weight='balanced', random_state=42)
    model.fit(X_train_scaled, y_train)
    logging.info("✅ Model training completed.")
except Exception as e:
    logging.error(f"❌ Model training failed: {str(e)}")
    exit(1)

# ✅ Evaluate model
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
logging.info(f"✅ Model trained with accuracy: {accuracy:.2f}")

# ✅ Ensure model directory exists
model_dir = "models"
os.makedirs(model_dir, exist_ok=True)

# ✅ Save the trained model & scaler
try:
    joblib.dump(model, os.path.join(model_dir, "hydration_model.pkl"))
    joblib.dump(scaler, os.path.join(model_dir, "scaler.pkl"))
    logging.info("✅ Model & scaler saved successfully in 'models/' directory.")
except Exception as e:
    logging.error(f"❌ Failed to save model: {str(e)}")

# ✅ Save model performance metrics
metrics = {"accuracy": accuracy}
with open(os.path.join(model_dir, "results.json"), "w") as f:
    json.dump(metrics, f, indent=4)
    logging.info("✅ Model performance metrics saved in 'models/results.json'.")
