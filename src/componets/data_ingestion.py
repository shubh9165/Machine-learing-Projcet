import sys
import pandas as pd
import os
from src.logger import logging
from src.exception import CustomExcption   # ✅ fixed spelling
from dataclasses import dataclass
from sklearn.model_selection import train_test_split


@dataclass
class DataIngestionConfig:   # ✅ class name should start with capital letter (standard practice)
    train_data_path: str = os.path.join('artifacts', 'train.csv')   # ✅ fixed spelling (artifacts)
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'data.csv')


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("The data ingestion process has been initiated")

        try:
            # ✅ fixed invalid escape sequence (used raw string)
            df = pd.read_csv(r'notebook\data\StudentsPerformance.csv')
            logging.info("Dataset has been successfully read")

            # ✅ ensure folder exists
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            # ✅ save raw data
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info("Raw dataset saved successfully")

            # ✅ split data
            logging.info("Train-test split initiated")
            train_set, test_set = train_test_split(df, test_size=0.20, random_state=42)

            # ✅ save train and test data
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Data ingestion process completed successfully")

        except Exception as e:
            # ✅ fixed spelling (CustomException)
            raise CustomExcption(e, sys)


if __name__ == "__main__":
    obj = DataIngestion()
    obj.initiate_data_ingestion()
