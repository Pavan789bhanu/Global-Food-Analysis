import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path=os.path.join("Data","model.pkl")
            preprocessor_path=os.path.join('Data','preprocessor.pkl')
            print("Before Loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)



class CustomData:
    def __init__(  self,
        adm0_id: int,
        adm1_id: int,
        cm_id: int,
        pt_id: int,
        mkt_id: int,
        mp_year: int):

        self.adm0_id = adm0_id

        self.adm1_id = adm1_id

        self.cm_id = cm_id

        self.pt_id = pt_id

        self.mkt_id = mkt_id

        self.mp_year = mp_year

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "adm0_id": [self.adm0_id],
                "adm1_id": [self.adm1_id],
                "cm_id": [self.cm_id],
                "pt_id": [self.pt_id],
                "mkt_id": [self.mkt_id],
                "mp_year": [self.mp_year]
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)