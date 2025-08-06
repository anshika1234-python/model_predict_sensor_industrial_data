# model_predict_sensor_industrial_data
•	Instructions:
o	Generate data: python raw_data/generate_data.py
o	Train: python training/train_model.py
o	Run app locally: python api/app.py
o	Build Docker: docker build -t sensor-anomaly .
o	Run container: docker run -p 5000:5000 sensor-anomaly
•	Note use of Python 3.10, TensorFlow 2.12, Flask, and joblib for seamless compatibility.
