from src.data.olist_data_loader import OlistDataLoader
from src.data.data_loader import DataLoader
from src.data.preprocessor import DataPreprocessor
from src.features.feature_engineering import FeatureEngineer
from src.models.lstm_model import LSTMModel
from src.utils.setup_hadoop import setup_hadoop_env
from config.config import Config
import os
import pandas as pd

def main():
    # Setup Hadoop environment
    setup_hadoop_env()
    
    # Load and preprocess Olist data
    data_loader = OlistDataLoader()
    daily_sales = data_loader.load_and_prepare_data()
    
    # Convert date column to datetime if it's not already
    daily_sales['date'] = pd.to_datetime(daily_sales['date'])
    
    # Process data in pandas
    preprocessor = DataPreprocessor()
    feature_engineer = FeatureEngineer()
    
    # Add time-based features
    daily_sales = feature_engineer.add_time_features_pandas(daily_sales)
    
    # Create sequences for LSTM
    X, y = preprocessor.prepare_sequences(daily_sales, Config.SEQUENCE_LENGTH)
    
    # Train model
    model = LSTMModel(input_shape=(X.shape[1], X.shape[2]))
    history = model.train(X, y)
    
    # Save model
    os.makedirs(Config.MODEL_DIR, exist_ok=True)
    model.save(os.path.join(Config.MODEL_DIR, "lstm_model.h5"))

if __name__ == "__main__":
    main()
