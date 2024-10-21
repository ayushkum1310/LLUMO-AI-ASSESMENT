import os
import sys
import pandas as pd
import re
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join(r'data\raw', "train.csv")
    validation_data_path: str = os.path.join(r'data\raw', "validation.csv")
    test_data_path: str = os.path.join(r'data\raw', "test.csv")
    processed_train_data_path: str = os.path.join(r'data\processed', "processed_train.csv")
    processed_validation_data_path: str = os.path.join(r'data\processed', "processed_validation.csv")
    processed_test_data_path: str = os.path.join(r'data\processed', "processed_test.csv")

class DataTransformation:
    def __init__(self):
        self.config = DataIngestionConfig()

    def preprocess_data(self, df: pd.DataFrame):
        """Preprocess the input column and return the cleaned DataFrame."""
        def clean_text(text):
            if isinstance(text, str):  # Check if the input is a string
                # Remove hashtags and mentions, and punctuation
                text = re.sub(r'[@#]', '', text)  # Remove hashtags and mentions
                text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
                return text.lower()  # Convert to lowercase
            return ''  # Return empty string for non-string inputs

        # Clean and preprocess only the input column
        df['Input'] = df['Input'].apply(clean_text)

        # Drop duplicates and null values in the 'input' column
        df.drop_duplicates(inplace=True)
        df.dropna(subset=['Input'], inplace=True)

        return df

    def process_and_save_data(self):
        try:
            # Load raw data and process it
            for split in ['train', 'validation', 'test']:
                df = pd.read_csv(getattr(self.config, f"{split}_data_path"))
                
                print(f"Loaded {split} DataFrame:\n", df.head())  # Debug: inspect the DataFrame
                print(f"Columns in {split} DataFrame:", df.columns.tolist())  # Check column names

                # Check for 'Input' column existence
                if 'Input' not in df.columns:
                    raise ValueError(f"'Input' column not found in {split} DataFrame.")

                # Preprocess the data
                processed_df = self.preprocess_data(df)

                # Save the processed data
                processed_path = getattr(self.config, f"processed_{split}_data_path")
                processed_df.to_csv(processed_path, index=False)
                logging.info(f"Processed data saved for {split} at {processed_path}")
        except Exception as e:
            raise CustomException(e, sys)

# Usage
if __name__ == "__main__":
    # Data Transformation
    data_transformation = DataTransformation()
    data_transformation.process_and_save_data()
