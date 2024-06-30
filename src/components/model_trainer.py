import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from sklearn.model_selection import GridSearchCV

from src.utils import evaluate_model, save_object

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Model training started")

            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models={
                'Linear Regression':LinearRegression(),
                'KNN':KNeighborsRegressor(),
                'SVR':SVR(),
                'Decision Tree':DecisionTreeRegressor(),
                'Random Forest':RandomForestRegressor(),
                'XGBoost':XGBRegressor(),
                'AdaBoost':AdaBoostRegressor()
            }

            params={
                "Linear Regression":{},
                "KNN":{
                    'n_neighbors':np.arange(1,20),
                    'weights':['uniform','distance']
                },
                "SVR":{
                    'kernel':['linear'],
                    'C':[10,50,100,150,200],
                    'epsilon':[0.01,0.1,0.5,1.0,1.5,2.0]
                },
                "Decision Tree":{
                    'criterion':['squared_error','friedman_mse','absolute_error','poisson'],
                    'max_depth':np.arange(1,20),
                    'min_samples_split':np.arange(2,20),
                },
                "Random Forest":{
                    'n_estimators':[10,50,100,150,200,250,300]
                },
                "XGBoost":{
                    'learning_rate':[0.001,0.01,0.1,1.0,1.5,2.0],
                    'n_estimators':[10,50,100,150,200,250,300]
                },
                "AdaBoost":{
                    'n_estimators':[10,50,100,150,200,250,300],
                    'learning_rate':[0.001,0.01,0.1,1.0,1.5,2.0],
                    'loss':['linear','square','exponential']
                }

                }
            model_report:dict=evaluate_model(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                params=params
            )
            best_model_score=max(sorted(model_report.values()))
            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model=models[best_model_name]

            logging.info("Best model found")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)
            r2score=r2_score(y_test,predicted)
            return r2score


        except Exception as e:
            raise CustomException(e,sys)

if __name__=="__main__":
    try:
        logging.info("model trainer script started")

        from data_ingestion import DataIngestion
        from data_transformation import DataTransform

        ingestion=DataIngestion()
        train_data_path,test_data_path=ingestion.initiate_data_ingestion()

        transformer = DataTransform()
        train_arr, test_arr, preprocessor_path = transformer.initiate_data_transform(train_data_path, test_data_path)

        trainer = ModelTrainer()
        r2_score = trainer.initiate_model_trainer(train_arr, test_arr)

        logging.info(f"Model training completed successfully with R2 score: {r2_score}")

    except Exception as e:
        logging.error(f"Error during the model training script execution: {e}", exc_info=True)
        raise        
            