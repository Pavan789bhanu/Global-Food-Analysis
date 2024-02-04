import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging
import os

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('data',"proprocessor.pkl")


class DataTransformation(self):
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def datatransformation(self):
        try:
            logging.info("Dropping the mp_commoditysource column")
            
        except Exception as e:
            raise CustomException(e,sys)
