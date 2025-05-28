import pandas as pd
import os

def load_data(file_path="C:\\Users\\sruja\\OneDrive\\Desktop\\creditcard.csv"):
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        print("Dataset loaded successfully!")
        print("\nFirst 5 rows:")
        print(df.head())
        return df
    else:
        print(f"Error: File not found at '{file_path}'")
        return None
