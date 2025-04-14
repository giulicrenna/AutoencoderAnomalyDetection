import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from src.packet import PacketParser
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dense, Dropout
from keras import regularizers
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger

MODELS_DIR = os.path.join(os.getcwd(), "models")
RESULTS_DIR = os.path.join(os.getcwd(), "results")
DATA_DIR = os.path.join(os.getcwd(), "data")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

os.environ['PATH'] += os.pathsep + r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin'
os.environ['PATH'] += os.pathsep + r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\libnvvp'
os.environ['PATH'] += os.pathsep + r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\extras\CUPTI\lib64'

gpus = tf.config.list_physical_devices('GPU')

if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"{len(gpus)} available GPU(s) detected.")
    except RuntimeError as e:
        print("Error:", e)
else:
    print("No GPU  detected. Using CPU instead.")
    
def calculate_threshold(errors: np.ndarray) -> float:
    """
    Calculate a dynamic threshold based on mean and standard deviation of the reconstruction errors.
    
    Parameters:
        errors (np.ndarray): Array of reconstruction errors.
    
    Returns:
        float: Dynamic threshold value.
    """
    return float(errors.mean() + 2 * errors.std())

def plot_metrics(errors: np.ndarray, threshold: float) -> None:
    """
    Plot the reconstruction error metrics and save the results to disk.

    Parameters:
        errors (np.ndarray): Array of reconstruction errors.
        threshold (float): Threshold value to be plotted.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(errors, label='Reconstruction Error', alpha=0.7)
    plt.axhline(y=threshold, color='r', linestyle='--', label=f'Dynamic threshold ({threshold:.5f})')
    plt.xlabel('Sample index')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.title('Reconstruction Error Distribution')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "error_distribution.png"))
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.hist(errors, bins=50, color='skyblue', edgecolor='black')
    plt.axvline(x=threshold, color='r', linestyle='--', label='Dynamic threshold')
    plt.title('Reconstruction Error Histogram')
    plt.xlabel('Error')
    plt.ylabel('Frequency')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "error_histogram.png"))
    plt.close()

def train(train_df_path: str = os.path.join(DATA_DIR, "train.csv")) -> None:
    """
    Train a deep autoencoder model for anomaly detection on network traffic data.
    The model and training artifacts are saved to disk.

    Parameters:
        train_df_path (str): Path to the CSV file containing the training dataset.
    """
    df: pd.DataFrame = pd.read_csv(train_df_path)
    parser = PacketParser(df)
    X_train: np.ndarray = parser.get_feature_matrix()

    with open(os.path.join(MODELS_DIR, "protocols.json"), "w") as f:
        json.dump(parser.protocols, f)

    multiplier: int = 2
    
    input_dim: int = X_train.shape[1]
    model = Sequential([
        Dense(256, activation='relu', input_shape=(input_dim,),
            activity_regularizer=regularizers.l1(1e-5)),
        Dropout(0.2),
        
        Dense(128, activation='relu'),
        Dropout(0.2),
        
        Dense(64, activation='relu'),
        Dropout(0.2),
        
        Dense(32, activation='relu'),
        Dropout(0.2),

        Dense(64, activation='relu'),
        Dropout(0.2),
        
        Dense(128, activation='relu'),
        Dropout(0.2),
        
        Dense(256, activation='relu'),
        Dropout(0.2),
        
        Dense(input_dim, activation='sigmoid')
])


    model.compile(optimizer='adam', loss='mse', metrics=['mae', 'mape'])

    callbacks = [
        EarlyStopping(monitor='loss', patience=10, restore_best_weights=True),
        ModelCheckpoint(os.path.join(MODELS_DIR, "model_best.keras"), monitor='loss', save_best_only=True),
        ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5),
        CSVLogger(os.path.join(MODELS_DIR, "training_log.csv"))
    ]

    model.fit(X_train, X_train, epochs=100, batch_size=32, callbacks=callbacks, verbose=1)

    X_pred: np.ndarray = model.predict(X_train)
    errors: np.ndarray = np.mean(np.square(X_train - X_pred), axis=1)
    threshold: float = calculate_threshold(errors)

    model.save(os.path.join(MODELS_DIR, "model.keras"))

    with open(os.path.join(MODELS_DIR, "threshold.json"), "w") as f:
        json.dump({"threshold": threshold}, f)

    plot_metrics(errors, threshold)
