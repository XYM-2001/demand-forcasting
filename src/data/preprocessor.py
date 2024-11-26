from pyspark.sql.functions import col, lag, avg
from pyspark.sql.window import Window
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

class DataPreprocessor:
    def __init__(self):
        self.scaler = MinMaxScaler()
    
    def preprocess_spark_df(self, df):
        # Create window specification
        window_spec = Window.partitionBy("product_id").orderBy("date")
        
        # Add features
        df = df.withColumn("lag_7", lag("sales", 7).over(window_spec))
        df = df.withColumn("rolling_avg", 
                          avg("sales").over(window_spec.rowsBetween(-7, 0)))
        
        return df.na.drop()
    
    def prepare_sequences(self, data, sequence_length):
        """
        Prepare sequences for LSTM model, handling datetime values
        """
        # Convert date to numerical value (days since minimum date)
        if 'date' in data.columns:
            min_date = data['date'].min()
            data['days_since_start'] = (data['date'] - min_date).dt.days
            
            # Select only numerical columns for scaling
            numerical_columns = data.select_dtypes(include=[np.number]).columns
            data_to_scale = data[numerical_columns]
        else:
            data_to_scale = data
            
        # Scale numerical features
        data_scaled = self.scaler.fit_transform(data_to_scale)
        
        # Create sequences
        X, y = [], []
        for i in range(sequence_length, len(data_scaled)):
            X.append(data_scaled[i-sequence_length:i, :])
            y.append(data_scaled[i, data_to_scale.columns.get_loc('sales')])
            
        return np.array(X), np.array(y)
