from pyspark.sql.functions import dayofweek, month, quarter
from pyspark.ml.feature import OneHotEncoder, StringIndexer
import pandas as pd

class FeatureEngineer:
    def __init__(self):
        pass
    
    def add_time_features_pandas(self, df):
        """Add time-based features using pandas"""
        df = df.copy()
        df['day_of_week'] = pd.to_datetime(df['date']).dt.dayofweek
        df['month'] = pd.to_datetime(df['date']).dt.month
        df['quarter'] = pd.to_datetime(df['date']).dt.quarter
        return df
    
    def encode_categorical_pandas(self, df, categorical_cols):
        """Encode categorical variables using pandas"""
        df = df.copy()
        for col in categorical_cols:
            df[f"{col}_encoded"] = pd.get_dummies(df[col], prefix=col)
        return df
