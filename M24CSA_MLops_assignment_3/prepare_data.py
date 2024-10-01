import seaborn as sns
import pandas as pd
import os

# Load the Titanic dataset directly from Seaborn
data = sns.load_dataset('titanic')

# Ensure the dataset directory exists
os.makedirs('dataset', exist_ok=True)

# Save the dataset to a CSV file in the dataset folder
data.to_csv('dataset/titanic.csv', index=False)

