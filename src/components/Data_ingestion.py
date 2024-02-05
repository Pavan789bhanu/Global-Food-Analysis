import sys
import os
import pandas as pd
from src.logger import logging
from src.exception import CustomException

from src.components.Data_transformation import DataTransformation
from src.components.Data_transformation import DataTransformationConfig

from src.components.Model_train import ModelTrainerConfig
from src.components.Model_train import ModelTrainer

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass

class DataIngestionConfig:
    train_data_path: str=os.path.join("Data","train.csv")
    test_data_path: str=os.path.join("Data","test.csv")
    raw_data_path: str=os.path.join("Data","data.csv")


class DataIngestion:
    def __init__(self):
        self.ingestionconfig = DataIngestionConfig()
    
    def data_split(self):
        logging.info("Loading the dataset")

        try:
            df = pd.read_csv("Jupytercode\dataset\Foodprices.csv")

            os.makedirs(os.path.dirname(self.ingestionconfig.train_data_path),exist_ok=True)

            df.to_csv(self.ingestionconfig.raw_data_path,index=False,header=True)

            train_set,test_set = train_test_split(df,test_size=0.2,random_state=100)

            train_set.to_csv(self.ingestionconfig.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestionconfig.test_data_path,index=False,header=True)

            return(
                self.ingestionconfig.train_data_path,
                self.ingestionconfig.test_data_path
            )

        except Exception as e:
            raise CustomException(e,sys)

if __name__=='__main__':
    obj = DataIngestion()
    train_data,test_data = obj.data_split()

    data_transformation_obj = DataTransformation()
    train_arr,test_arr,_ = data_transformation_obj.data_transformation(train_data,test_data)

    model_obj = ModelTrainer()
    print(model_obj.modeltraining(train_arr,test_arr))

