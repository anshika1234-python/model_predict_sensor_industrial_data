import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, precision_recall_curve, auc
from tensorflow.keras.models import load_model

def evaluate_autoencoder(data_csv="raw_data/sensor_data.csv"):
    df = pd.read_csv(data_csv)
    X = df.drop(columns=["label"]).values
    y_true = df["label"].values

    scaler = joblib.load("../api/scaler.pkl")
    model = load_model("../api/autoencoder.h5")
    threshold = joblib.load("../api/threshold.pkl")

    X_scaled = scaler.transform(X)
    X_pred = model.predict(X_scaled)
    mse = np.mean(np.square(X_scaled - X_pred), axis=1)

    # Apply threshold
    y_pred = (mse > threshold).astype(int)

    # Classification metrics
    print("=== Classification Report ===")
    print(classification_report(y_true, y_pred))
    try:
        roc = roc_auc_score(y_true, mse)
        print(f"ROC AUC (using MSE scores) : {roc:.3f}")
    except ValueError:
        pass

    # Precision-recall curve
    precision, recall, pr_thresh = precision_recall_curve(y_true, mse)
    pr_auc = auc(recall, precision)
    print(f"Precision-Recall AUC: {pr_auc:.3f}")

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:\n", cm)

    # Plot distributions and PR curve
    plt.figure(figsize=(14,6))
    # Reconstruction error histogram
    plt.subplot(1, 2, 1)
    plt.hist(mse[y_true==0], bins=50, alpha=0.7, label='Normal')
    plt.hist(mse[y_true==1], bins=50, alpha=0.7, label='Anomaly')
    plt.axvline(threshold, color='r', linestyle='--', label=f'Threshold = {threshold:.3f}')
    plt.xlabel("Reconstruction MSE")
    plt.ylabel("Count")
    plt.legend()
    plt.title("Error Distribution")

    # Precision-recall curve
    plt.subplot(1, 2, 2)
    plt.plot(recall, precision, label=f'PR AUC = {pr_auc:.3f}')
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()

    plt.tight_layout()
    plt.savefig("evaluation_plots.png")
    plt.show()

if __name__ == "__main__":
    evaluate_autoencoder()
