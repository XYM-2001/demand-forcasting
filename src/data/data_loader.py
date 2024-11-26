from pyspark.sql import SparkSession
from config.config import Config

class DataLoader:
    def __init__(self):
        self.spark = SparkSession.builder \
            .appName(Config.SPARK_APP_NAME) \
            .getOrCreate()
    
    def load_csv(self, file_path):
        return self.spark.read.csv(
            file_path,
            header=True,
            inferSchema=True
        )
    
    def load_streaming_data(self, path):
        return self.spark.readStream \
            .format("csv") \
            .option("header", "true") \
            .load(path)
