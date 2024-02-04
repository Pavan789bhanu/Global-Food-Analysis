import sys
import os
import pandas as pd 
import numpy as np 
from src.logger import logging
from src.exception import CustomException

from sklearn.preprocessing import LabelEncoder


class DataTransformation(self):
    def __init__(self):
        return null
    def datatransformation(self):
        try:
            logging.info("Dropping the mp_commoditysource column")
        except Exception as e:
            raise CustomException(e,sys)
