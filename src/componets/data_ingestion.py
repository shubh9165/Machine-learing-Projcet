import sys
import pandas as pd
import os
from src.logger import logging
from src.exception import CustomExcption   # ✅ fixed spelling
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from src.componets.data_transformation import DataTransformation
from src.componets.data_transformation import DataTransformationConfig

from src.componets.model_trainer import modelTrainer
from src.componets.model_trainer import modelTrainerConfig



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
            df = pd.read_csv(r'notebook\data\StudentsPerformance.csv')
            df.columns = df.columns.str.lower().str.replace(" ", "_").str.replace("/", "_")
            logging.info("Dataset has been successfully read")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info("Raw dataset saved successfully")

            logging.info("Train-test split initiated")
            train_set, test_set = train_test_split(df, test_size=0.20, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Data ingestion process completed successfully")

            # ✅ Add this return statement
            return (
             self.ingestion_config.train_data_path,
              self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomExcption(e, sys)


if __name__ == "__main__":
    obj=DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion()

    data_transformation=DataTransformation()
    train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data,test_data)

    model_trainer=modelTrainer()
    print(model_trainer.initate_model_trainer(train_arr,test_arr))