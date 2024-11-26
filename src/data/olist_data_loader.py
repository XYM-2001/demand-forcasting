import os
import sys

# Add the project root directory to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)

import pandas as pd
from config.config import Config

class OlistDataLoader:
    def __init__(self):
        self.data_dir = Config.RAW_DATA_DIR
        
    def load_and_prepare_data(self):
        # Load relevant tables
        orders = pd.read_csv(os.path.join(self.data_dir, 'olist_orders_dataset.csv'))
        order_items = pd.read_csv(os.path.join(self.data_dir, 'olist_order_items_dataset.csv'))
        products = pd.read_csv(os.path.join(self.data_dir, 'olist_products_dataset.csv'))
        
        # Merge orders and order items
        df = orders.merge(order_items, on='order_id')
        df = df.merge(products, on='product_id')
        
        # Convert order_purchase_timestamp to datetime
        df['date'] = pd.to_datetime(df['order_purchase_timestamp']).dt.date
        
        # Group by date and product_id to get daily sales
        daily_sales = df.groupby(['date', 'product_id', 'product_category_name']).agg({
            'order_id': 'count',  # number of orders
            'price': 'mean',      # average price
        }).reset_index()
        
        # Rename columns
        daily_sales = daily_sales.rename(columns={
            'order_id': 'sales',
            'price': 'avg_price'
        })
        
        # Sort by date
        daily_sales = daily_sales.sort_values('date')
        
        # Save processed data
        processed_path = os.path.join(Config.PROCESSED_DATA_DIR, 'processed_sales.csv')
        os.makedirs(Config.PROCESSED_DATA_DIR, exist_ok=True)
        daily_sales.to_csv(processed_path, index=False)
        
        return daily_sales

if __name__ == "__main__":
    # Test the data loader
    loader = OlistDataLoader()
    df = loader.load_and_prepare_data()
    print("\nFirst few rows of processed data:")
    print(df.head())
    print("\nDataset shape:", df.shape)
    print("\nUnique products:", df['product_id'].nunique())
    print("\nDate range:", df['date'].min(), "to", df['date'].max()) 