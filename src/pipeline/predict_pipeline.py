import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import joblib
import pandas as pd
from src.exception import CustomException
from src.logger import logging

from src.utils import load_obj

class PredictPipeline:
    def __init__(self):
        self.model_path=os.path.join('artifacts','model.pkl')
        self.preprocessor_path=os.path.join('artifacts','preprocessor.pkl')

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file missing: {self.model_path}")
        if not os.path.exists(self.preprocessor_path):
            raise FileNotFoundError(f"Preprocessor file missing: {self.preprocessor_path}")

        self.model=joblib.load(self.model_path)

    def predict(self,features):
        try:
#            model_path='artifacts\model.pkl'
#            preprocessor_path='artifacts\preprocessor.pkl'
            model=load_obj(file_path=self.model_path)
            preprocessor=load_obj(file_path=self.preprocessor_path)
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)    

class CustomData:
    def __init__(self,year,kmdriven,fuel,mileage,power,transmission,engine,seller,owner):
        self.year=year
        self.kmdriven=kmdriven
        self.fuel=fuel
        self.mileage=mileage
        self.power=power
        self.transmission=transmission
        self.engine=engine
        self.seller=seller
        self.owner=owner

    def get_data_as_df(self):
        try:
            custom_data_input_dict={
                "year":[self.year],
                "km_driven":[self.kmdriven],
                "fuel":[self.fuel],
                "seller_type":[self.seller],
                "transmission":[self.transmission],
                "owner":[self.owner],
                "mileage(km/ltr/kg)":[self.mileage],
                "engine":[self.engine],
                "max_power":[self.power],
            }    
            df=pd.DataFrame(custom_data_input_dict)
            print("Custom Data Frame: ")
            print(df)
            return df
        except Exception as e:
            raise CustomException(e,sys)                