import pandas as pd
import numpy as np
import tensorflow as tf
import json
import os
import time
from src.packet import PacketParser

MODELS_DIR = os.path.join(os.getcwd(), "models")
RESULTS_DIR = os.path.join(os.getcwd(), "results")
DATA_DIR = os.path.join(os.getcwd(), "data")

def predict() -> None:
    """
    Loads the trained autoencoder model and evaluates a test dataset to detect anomalies.
    A dynamic threshold is calculated from the reconstruction error distribution.
    Each packet is classified as 'Anomalous' or 'Normal' and printed with its error.
    """
    model: tf.keras.Model = tf.keras.models.load_model(os.path.join(MODELS_DIR, "model_best.keras"))

    print(model.summary())
    
    print("\n--- Predicting anomalies ---\n")
    
    with open(os.path.join(MODELS_DIR, "protocols.json"), "r") as f:
        trained_protocols = json.load(f)

    df: pd.DataFrame = pd.read_csv(os.path.join(DATA_DIR, "validate.csv"))
    parser = PacketParser(df, forced_protocols=trained_protocols)
    X_test: np.ndarray = parser.get_feature_matrix()

    start = time.time()
    X_pred: np.ndarray = model.predict(X_test)
    print(f"Prediction time: {time.time() - start:.2f} seconds")
    errors: np.ndarray = np.mean(np.square(X_test - X_pred), axis=1)

    threshold: float = float(errors.mean() + 2 * errors.std())
    print(f"\nCalculated dynamic threshold: {threshold:.5f}\n")

    anomaly_count = 0
    normal_count = 0

    for i, error in enumerate(errors):
        status: str = "Anomaly" if error > threshold else "Normal"
        # print(f"Packet {i+1}: Error={error:.5f} ➤ {status}")
        if status == "Anomaly":
            anomaly_count += 1
        else:
            normal_count += 1

    total = len(errors)
    
    print("\n--- Metrics ---")
    print(f"Total packets: {total}")
    print(f"Anomalies detected: {anomaly_count} ({(anomaly_count / total * 100):.2f}%)")
    print(f"Normal packets: {normal_count} ({(normal_count / total * 100):.2f}%)")
    print(f"Error mean: {errors.mean():.5f}")
    print(f"Error std deviation: {errors.std():.5f}")
    print(f"Threshold: {threshold:.5f}")
