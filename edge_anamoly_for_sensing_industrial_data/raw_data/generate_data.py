
import numpy as np
import pandas as pd

def generate_sensor_data(n_samples=10000, n_sensors=3, anomaly_ratio=0.01, seed=42):
    np.random.seed(seed)
    t = np.arange(n_samples)
    data = np.stack([
        np.sin(0.02 * t + phase) for phase in range(n_sensors)
    ], axis=1) * np.linspace(1,2,n_sensors) + 0.1 * np.random.randn(n_samples, n_sensors)
    labels = np.zeros(n_samples, dtype=int)
    n_anom = int(n_samples * anomaly_ratio)
    idx = np.random.choice(n_samples, n_anom, replace=False)
    data[idx] += np.random.normal(5,1,(n_anom,n_sensors))
    labels[idx] = 1
    df = pd.DataFrame(data, columns=[f"sensor{i}" for i in range(n_sensors)])
    df["label"] = labels
    df.to_csv("raw_data/sensor_data.csv", index=False)
    print(f"Saved: {df.shape}, anomalies={n_anom}")

if __name__ == "__main__":
    generate_sensor_data()
