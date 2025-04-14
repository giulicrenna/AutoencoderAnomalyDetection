import os
from src.train import train
from src.predict import predict

MODELS_DIR = os.path.join(os.getcwd(), "models")
RESULTS_DIR = os.path.join(os.getcwd(), "results")
DATA_DIR = os.path.join(os.getcwd(), "data")

if __name__ == "__main__":
    if not os.path.isfile(os.path.join(os.path.join(MODELS_DIR), "model_best.keras")):
        train()
    predict()