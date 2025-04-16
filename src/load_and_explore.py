import pandas as pd
import os

# Load the CSV file (replace filename if needed)
data_path = os.path.join("data", "traffic_data.csv")  # adjust filename if it's different
df = pd.read_csv(data_path)

# Preview the dataset
print("\n🔍 First 5 rows of the dataset:")
print(df.head())

# Dataset shape
print(f"\n📐 Shape of dataset: {df.shape}")

# Check for missing values
print("\n🧼 Missing values:")
print(df.isnull().sum())

# Data types
print("\n📋 Column data types:")
print(df.dtypes)
