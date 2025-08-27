# model_predict_sensor_industrial_data
•	Industrial Sensor Anomaly Detection System
o	Built synthetic multivariate time series dataset with controlled anomalies.
o	Trained autoencoder to flag anomalies; achieved precision/recall ~x %.
o	Deployed inference API using Flask + Docker, demonstrating real world deployability.
o	Project reflects key HCLTech strengths: edge AI, anomaly detection, synthetic data generation, and MLOps readiness.

# Real-Time Anomaly Detection for Industrial Sensors

[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![TensorFlow 2.12](https://img.shields.io/badge/TensorFlow-2.12-orange.svg)](https://www.tensorflow.org/)
[![Flask](https://img.shields.io/badge/Flask-API-green.svg)](https://flask.palletsprojects.com/)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com/)

Project Structure -
raw_data/
├── generate_data.py
notebooks/
├── data_analysis.ipynb
├── model_training.ipynb
api/
├── app.py  # Flask API
Dockerfile
requirements.txt
README.md

What the project includes
1. Synthetic dataset generation
•	Simulate multivariate time-series sensor data (e.g. temperature, vibration, pressure).
•	Inject anomalies (spikes, drift, outliers) with configurable frequency.
2. Model training & evaluation
•	Train a simple autoencoder neural network to reconstruct sensor readings.
•	Use reconstruction error to detect anomalies.
•	Provide metrics: precision, recall, F1.

3. Pipeline / MLOps
•	Preprocessing, training & model evaluation notebooks.
•	Model serialized via Joblib or TorchScript.
•	REST API via Flask + Docker + Gunicorn for inference.
•	Brief instructions for scaling deployment locally (extendable to Heroku or cloud).


•	Instructions:
o	Generate data: python raw_data/generate_data.py
o	Train: python training/train_model.py
o	Run app locally: python api/app.py
o	Build Docker: docker build -t sensor-anomaly .
o	Run container: docker run -p 5000:5000 sensor-anomaly
•	Note use of Python 3.10, TensorFlow 2.12, Flask, and joblib for seamless compatibility.

Getting Started

Follow these instructions to get the project up and running on your local machine.

#### ### 1. Clone the Repository
bash
git clone [https://github.com/anshika1234-python/model_predict_sensor_industrial_data.git](https://github.com/anshika1234-python/model_predict_sensor_industrial_data.git)
cd model_predict_sensor_industrial_data

2. Install Dependencies
Bash
pip install -r requirements.txt

3. Run the Pipeline
a. Generate Synthetic Data:
Bash
python raw_data/generate_data.py
b. Train the Model:
Bash
python training/train_model.py

4. Run the API
Option A: Run with Docker (Recommended)
Bash
# Build the Docker image
docker build -t sensor-anomaly .
# Run the container
docker run -p 5000:5000 sensor-anomaly
Option B: Run Locally
Bash
python api/app.py

