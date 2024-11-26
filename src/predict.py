from src.models.lstm_model import LSTMModel
from src.data.preprocessor import DataPreprocessor
import pandas as pd
import numpy as np
from config.config import Config
import os

def load_latest_data():
    # Load the processed data
    data_path = os.path.join(Config.PROCESSED_DATA_DIR, 'processed_sales.csv')
    return pd.read_csv(data_path)

def make_predictions(model, data, preprocessor, days_ahead=7):
    # Prepare the last sequence for prediction
    X, _ = preprocessor.prepare_sequences(data, Config.SEQUENCE_LENGTH)
    last_sequence = X[-1:]  # Get the most recent sequence
    
    predictions = []
    for _ in range(days_ahead):
        # Make prediction
        pred = model.predict(last_sequence)
        predictions.append(pred[0][0])
        
        # Update sequence for next prediction
        last_sequence = np.roll(last_sequence, -1, axis=1)
        last_sequence[0, -1, 0] = pred[0][0]
    
    return predictions

def main():
    # Load model
    model = LSTMModel.load(os.path.join(Config.MODEL_DIR, "lstm_model.h5"))
    
    # Load data
    data = load_latest_data()
    preprocessor = DataPreprocessor()
    
    # Make predictions
    predictions = make_predictions(model, data, preprocessor)
    
    # Print results
    print("\nNext 7 days sales predictions:")
    for i, pred in enumerate(predictions, 1):
        print(f"Day {i}: {pred:.2f}")

if __name__ == "__main__":
    main() 