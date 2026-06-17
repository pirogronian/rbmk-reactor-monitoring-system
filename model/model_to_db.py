import torch
import numpy as np
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS
import joblib
import pickle
import time
from dotenv import load_dotenv
import os
import logging
from pathlib import Path
import sys

# Add parent directory to path to import from src
sys.path.insert(0, str(Path(__file__).parent))

from src.model import AnomalyDetector

load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configuration ---
SEQ_LEN = 30
INPUT_SIZE  = 12   # Number of features in PARAMS list
HIDDEN_SIZE = 256
NUM_LAYERS  = 3
MODEL_TYPE = "lstm"
PARAMS = ["thermal_power_mw", "fuel_reactivity", "orm_value", "partially_inserted", 
"inlet_temp_c", "outlet_temp_c", "coolant_flow_m3h", "v_steam", "xenon_level", "neutron_flux_pct"]
THRESHOLD = 0.5  # MSE threshold for anomaly detection (reconstruction error)
DEVICE = "cpu"

# --- Paths ---
MODEL_PATH = Path(__file__).parent / "checkpoints" / "best_model.pth"
SCALER_PATH = Path(__file__).parent / "checkpoints" / "scaler.pkl"

# --- Load model ---
try:
    # Load checkpoint
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    
    # Create model with same architecture as training
    model = AnomalyDetector(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        model_type=MODEL_TYPE,
    ).to(DEVICE)
    
    # Load saved state dict
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"✓ Model loaded from {MODEL_PATH}")
    print(f"  Architecture: {MODEL_TYPE.upper()}")
    print(f"  Input size: {INPUT_SIZE}, Hidden: {HIDDEN_SIZE}, Layers: {NUM_LAYERS}")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    raise

# --- Load scaler ---
try:
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    
    logger.info(f"✓ Scaler loaded from {SCALER_PATH}")
    logger.info(f"  Scaler expects {scaler.n_features_in_} features")
except FileNotFoundError:
    logger.error(f"Scaler not found at {SCALER_PATH}")
    logger.error("The scaler must be saved during training.")
    logger.error("Run the training script first: python main.py --mode train")
    raise
except Exception as e:
    logger.error(f"Failed to load scaler: {e}")
    raise

# --- InfluxDB client ---
try:
    client = InfluxDBClient(
        url=os.getenv("INFLUXDB_URL"),
        token=os.getenv("INFLUXDB_TOKEN"),
        org=os.getenv("INFLUXDB_ORG")
    )
    query_api = client.query_api()
    write_api = client.write_api(write_options=SYNCHRONOUS)
    logger.info("Connected to InfluxDB")
except Exception as e:
    logger.error(f"Failed to connect to InfluxDB: {e}")
    raise

def get_recent_window():
    """Fetch recent data from InfluxDB."""
    try:
        query = f'''
        from(bucket: "{os.getenv("INFLUXDB_BUCKET")}")
          |> range(start: -{SEQ_LEN}m)
          |> filter(fn: (r) => r._measurement == "rbmk_reactor_metrics")
          |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
          |> sort(columns: ["_time"])
        '''
        df = query_api.query_data_frame(query)
        return df
    except Exception as e:
        logger.error(f"Error querying InfluxDB: {e}")
        return None

def preprocess(df):
    """Preprocess DataFrame: extract features and scale."""
    try:
        # Extract only the required parameters
        arr = df[PARAMS].values  # shape (seq_len, n_features)
        
        # Scale using the fitted scaler
        scaled = scaler.transform(arr)
        
        # Convert to tensor: (1, seq_len, n_features)
        tensor = torch.tensor(scaled, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        return tensor
    except Exception as e:
        logger.error(f"Error preprocessing data: {e}")
        return None

def compute_reconstruction_error(x):
    """Compute reconstruction error using the autoencoder."""
    try:
        with torch.no_grad():
            # Forward pass through AnomalyDetector returns only reconstructed tensor
            reconstructed = model(x)
        
        # Compute MSE between input and reconstruction
        mse = torch.mean((x - reconstructed) ** 2, dim=(1, 2))  # Average over time and features
        return mse.item()
    except Exception as e:
        logger.error(f"Error computing reconstruction error: {e}")
        return None


def main():
    """Main inference loop."""
    logger.info("Starting anomaly detection inference loop...")
    
    while True:
        try:
            # Get recent data
            df = get_recent_window()
            
            if df is None or len(df) == 0:
                logger.warning(f"No data retrieved, sleeping...")
                time.sleep(10)
                continue
            
            # Check if we have enough data
            if len(df) < SEQ_LEN:
                logger.warning(f"Insufficient data: {len(df)} < {SEQ_LEN}")
                time.sleep(10)
                continue
            
            # Preprocess the last SEQ_LEN samples
            x = preprocess(df.tail(SEQ_LEN))
            
            if x is None:
                time.sleep(10)
                continue
            
            # Compute reconstruction error
            error = compute_reconstruction_error(x)
            
            if error is None:
                time.sleep(10)
                continue
            
            # Determine if anomaly
            is_anomaly = int(error > THRESHOLD)
            
            # Log result
            logger.info(f"Reconstruction error: {error:.4f}, Anomaly: {is_anomaly}")
            
            # Write to InfluxDB
            try:
                point = (
                    Point("lstm_anomaly_detection")
                    .field("reconstruction_error", error)
                    .field("is_anomaly", is_anomaly)
                    .field("threshold", THRESHOLD)
                )
                write_api.write(bucket=os.getenv("INFLUXDB_BUCKET"), record=point)
            except Exception as e:
                logger.error(f"Failed to write to InfluxDB: {e}")
            
            # Sleep before next inference
            time.sleep(10)
            
        except Exception as e:
            logger.error(f"Unexpected error in main loop: {e}")
            time.sleep(10)
            continue

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        client.close()