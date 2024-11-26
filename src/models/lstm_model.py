import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.losses import MeanSquaredError
from config.config import Config

class LSTMModel:
    def __init__(self, input_shape):
        self.model = self._build_model(input_shape)
    
    def _build_model(self, input_shape):
        model = Sequential([
            LSTM(Config.LSTM_UNITS, 
                 return_sequences=True, 
                 input_shape=input_shape),
            LSTM(Config.LSTM_UNITS),
            Dense(1)
        ])
        
        model.compile(
            optimizer="adam",
            loss=MeanSquaredError()
        )
        return model
    
    def train(self, X, y):
        return self.model.fit(
            X, y,
            epochs=Config.EPOCHS,
            batch_size=Config.BATCH_SIZE
        )
    
    def predict(self, X):
        return self.model.predict(X)
    
    def save(self, path):
        self.model.save(path)
    
    @classmethod
    def load(cls, path):
        model = tf.keras.models.load_model(path)
        return model
