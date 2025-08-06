from flask import Flask, jsonify, request
import joblib
from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load your trained artifacts
scaler = joblib.load("scaler.pkl")
model = load_model("autoencoder.h5")
threshold = joblib.load("threshold.pkl")

@app.route('/')
def index():
    return "Edge Anomaly Detector is running", 200

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    df = pd.DataFrame([data])
    scaled = scaler.transform(df.values)
    recon = model.predict(scaled)
    mse = np.mean(np.square(scaled - recon), axis=1)[0]
    label = int(mse > threshold)
    return jsonify(reconstruction_error=float(mse), is_anomaly=label)

if __name__ == '__main__':
    app.run(debug=True)
