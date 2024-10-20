import os
import sys
import pandas as pd
from src.logger import logging
from src.exception import CustomException

import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts',"train.csv")
    test_data_path: str=os.path.join('artifacts',"test.csv")
    raw_data_path: str=os.path.join('artifacts',"data.csv")


import os
import pandas as pd
from datasets import load_dataset
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join(r'data\raw', "train.csv")
    validation_data_path: str = os.path.join(r'data\raw', "validation.csv")
    test_data_path: str = os.path.join(r'data\raw', "test.csv")

class DataIngestion:
    def __init__(self):
        self.config = DataIngestionConfig()

    def download_and_save_data(self):
        try:
        # Load the dataset
            ds = load_dataset("zqz979/meta-review")

            # Save each split to CSV
            for split in ds.keys():
                df = ds[split].to_pandas()  # Convert to pandas DataFrame

                # Save to the respective CSV file
                if split == 'train':
                    df.to_csv(self.config.train_data_path, index=False)
                elif split == 'validation':
                    df.to_csv(self.config.validation_data_path, index=False)
                elif split == 'test':
                    df.to_csv(self.config.test_data_path, index=False)
            logging.info("Data is downloded and saved")
            print("Train, validation, and test splits saved to CSV.")
        except Exception as e:
            raise CustomException(e,sys)

# Usage
if __name__ == "__main__":
    config = DataIngestionConfig()
    data_ingestion = DataIngestion()
    data_ingestion.download_and_save_data()
