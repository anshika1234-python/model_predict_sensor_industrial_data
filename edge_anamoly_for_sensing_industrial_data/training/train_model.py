import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import joblib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def train_autoencoder(data_csv="raw_data/sensor_data.csv"):
    df = pd.read_csv(data_csv)
    X = df.drop(columns=["label"]).values
    y = df["label"].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = Sequential([
        Dense(16, activation="relu", input_shape=(X_scaled.shape[1],)),
        Dense(8, activation="relu"),
        Dense(16, activation="relu"),
        Dense(X_scaled.shape[1])
    ])
    model.compile(optimizer="adam", loss="mse")
    model.fit(X_scaled, X_scaled, epochs=30, batch_size=64, validation_split=0.1)

    recons = model.predict(X_scaled)
    mse = np.mean(np.square(recons - X_scaled), axis=1)
    thresh = np.percentile(mse, 99)
    preds = (mse > thresh).astype(int)

    print("Threshold:", thresh)
    print(classification_report(y, preds))

    joblib.dump(scaler, "../api/scaler.pkl")
    joblib.dump(thresh, "../api/threshold.pkl")
    model.save("../api/autoencoder.h5")
    print("Saved model and scaler.")

if __name__ == "__main__":
    train_autoencoder()
