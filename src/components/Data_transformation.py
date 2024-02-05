import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
#from src.utils import encoding

import os

@dataclass
class DataTransformationConfig:
    '''
        We are providing inputs to data transformation
    '''
    preprocessor_obj_file_path=os.path.join('data',"proprocessor.pkl")


class DataTransformation:
    '''
    Initalising the input value
    '''
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
    
    def datatransformation(self,x):
        '''
        Writing the function to perform the preprocessing of data using pipelines
        '''
        try:
            '''
            Getting the numerical and categorical variables
            '''
            num_columns = x.select_dtypes(exclude = "object").columns
            cat_columns = x.select_dtypes(include = "object").columns
            
            '''
            Handle the missing values, Standarization and converting the categorical features into numerical featues using Onehot encoder
            Imputer is used to handle the missing values in the data
            Standardscaler used for saclling the data
            Label encoder is used to convert the categorical features into numerical features
            '''
            num_pipeline = Pipeline(
                steps = [
                    ("imputer",SimpleImputer(strategy="median")),
                    ("scaler",StandardScaler())
                ]
            )

            # cat_pipeline = Pipeline(
            #     steps = [
            #         ("imputer",SimpleImputer(strategy="most_frequent")),
            #         ("labelencoder",ColumnTransformer([("labelencoder",LabelEncoder(),cat_columns)],remainder='passthrough')),
            #         ("scaler",StandardScaler())
            #     ]
            # )

            logging.info(f"Categorical columns: {cat_columns}")
            logging.info(f"Numerical columns: {num_columns}")

            '''
            We are going to use the column transformer to combine both the pipelines
            '''

            preprocessor = ColumnTransformer(
                [("num_pipeline",num_pipeline,num_columns)
                #("cat_pipeline",cat_pipeline,cat_columns)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e,sys)
    
    def data_transformation(self,train_path,test_path):
        try:
            '''
            Read the traing and testing data
            '''
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")
            '''
            Selecting the required features for model training 
            '''
            logging.info("Initalising the required features")

            x = train_df[['adm0_id','adm1_id','cm_id','pt_id','mkt_id','mp_year']]
            #y = test_df[['adm0_id','adm1_id','cm_id','cm_name','pt_id','pt_name','mkt_id','mkt_name','mp_year']]

            '''
            Performing the Preprocessig on the selecting features using the preprocessing object
            '''
            logging.info("Obtaining preprocessing object")

            preprocessing_obj = self.datatransformation(x)

            target_column_name = "mp_price"
            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info("Applying the preprocessing object to tarining and testing data")

            feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            '''
            use of NumPy's np.c_ function to concatenate arrays column-wise 
            '''

            train_arr = np.c_[feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e,sys)
