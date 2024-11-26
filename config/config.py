import os

class Config:
    # Paths
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(ROOT_DIR, 'data')
    RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
    PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
    MODEL_DIR = os.path.join(ROOT_DIR, 'models/saved_models')
    
    # Model parameters
    SEQUENCE_LENGTH = 30
    BATCH_SIZE = 32
    EPOCHS = 20
    LSTM_UNITS = 50
    
    # API settings
    API_HOST = "0.0.0.0"
    API_PORT = 5000
    
    # Spark settings
    SPARK_APP_NAME = "DemandForecasting"
