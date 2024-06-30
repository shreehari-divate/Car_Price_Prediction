import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import dill
import pickle

from src.exception import CustomException
from src.logger import logging
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

def save_object(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,"wb") as file_obj:
            dill.dump(obj,file_obj)
    except Exception as e:
        raise CustomException(e,sys)  

'''def evaluate_model(X_train,y_train,X_test,y_test,models,params):
    try:
        report={}
        for i in range(len(list(models))):
            model=list(models.values())[i]
            para=params[list(models.keys())[i]]

            gs=GridSearchCV(estimator=model,param_grid=para,cv=5)
            gs.fit(X_train,y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)

            ytrain_pred=model.predict(X_train)
            ytest_pred=model.predict(X_test)

            train_model_score=r2_score(y_train,ytrain_pred)
            test_model_score=r2_score(y_test,ytest_pred)
            
            report[list(model.keys())[i]]=test_model_score
        return report
    except Exception as e:
        raise CustomException(e,sys)
    '''
def evaluate_model(X_train, y_train, X_test, y_test, models, params):
    try:
        report = {}

        for model_name, model in models.items():
            
            param_grid = params.get(model_name, {})

            if param_grid:
                # Perform GridSearchCV only if parameters are provided
                gs = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1)
                gs.fit(X_train, y_train)
                best_params = gs.best_params_
                
                model.set_params(**best_params)
            else:
                # Skip GridSearchCV if no parameters are provided
                logging.info(f"No parameters to tune for {model_name}, skipping GridSearchCV")

            # Fit the model
            model.fit(X_train, y_train)

            # Predict on training and testing data
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Calculate R2 scores
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            logging.info(f"{model_name} - Train R2 score: {train_model_score}, Test R2 score: {test_model_score}")

            # Store the test score in the report
            report[model_name] = test_model_score
        
        logging.info("Model evaluation completed")
        return report
    
    except Exception as e:
        raise CustomException(e, sys)
        
        
              