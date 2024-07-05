import os
import sys
from dataclasses import dataclass
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.exception import CustomException
from src.logger import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocess_obj_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransform:
    def __init__(self):
        self.data_transform_config = DataTransformationConfig()

    def get_data_transform_object(self):
        try:
            logging.info("Setting up data transformation pipelines")

             # Pipeline for numerical columns
            num_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])

            # Separate imputation for 'mileage' with mean strategy
            mileage_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler())
            ])

            # Pipeline for categorical columns
            cat_pipeline = Pipeline(steps=[
                ('one-hot-encoder', OneHotEncoder(handle_unknown='ignore'))
            ])

            # Combine all transformations
            transformers = [
                ('mileage_pipeline', mileage_pipeline, ['mileage(km/ltr/kg)']),
                ('num_pipeline', num_pipeline, ['engine', 'max_power']),
                ('cat_pipeline', cat_pipeline, ['fuel', 'seller_type', 'transmission', 'owner'])
            ]

            # Combine all transformations using ColumnTransformer
            preprocessor = ColumnTransformer(transformers)

            logging.info("Preprocessing setup completed")
            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transform(self, train_path: str, test_path: str):
        try:
            logging.info("Loading training and test datasets")
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info('Read train and test data completed')

            logging.info("DataFrame columns:")
            logging.info(f"Train columns: {train_df.columns.tolist()}")
            logging.info(f"Test columns: {test_df.columns.tolist()}")

            logging.info("Checking for non-numeric data in numeric columns")
            for col in ['engine', 'max_power', 'seats']:
                non_numeric_entries = train_df[col][pd.to_numeric(train_df[col], errors='coerce').isna()].unique()
                if len(non_numeric_entries) > 0:
                    logging.warning(f"Non-numeric entries found in {col}: {non_numeric_entries}")

            logging.info("Obtaining preprocessing object")
            preprocess_obj = self.get_data_transform_object()

            target_col_name = 'selling_price'

            # Drop 'name' column and target column from input features
            input_feature_train_df = train_df.drop(columns=[target_col_name, 'name','seats'], axis=1)
            target_feature_train_df = train_df[target_col_name]

            input_feature_test_df = test_df.drop(columns=[target_col_name, 'name','seats'], axis=1)
            target_feature_test_df = test_df[target_col_name]

            logging.info("Applying preprocessing object on training and test dataframes")

            input_feature_train_arr = preprocess_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocess_obj.transform(input_feature_test_df)

            logging.info("Combining transformed input features with target variable")
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info("Creating artifacts directory if it does not exist")
            os.makedirs(os.path.dirname(self.data_transform_config.preprocess_obj_file_path), exist_ok=True)

            logging.info("Saving preprocessing object")
            save_object(
                file_path=self.data_transform_config.preprocess_obj_file_path,
                obj=preprocess_obj
            )

            logging.info(f"Preprocessing object saved at {self.data_transform_config.preprocess_obj_file_path}")
            return (
                train_arr, test_arr, self.data_transform_config.preprocess_obj_file_path
            )

        except Exception as e:
            logging.error(f"Error in data transformation process: {e}")
            raise

if __name__ == "__main__":
    try:
        from data_ingestion import DataIngestion
        ingestion = DataIngestion()
        train_data_path, test_data_path = ingestion.initiate_data_ingestion()

        transformer = DataTransform()
        train_arr, test_arr, preprocessor_path = transformer.initiate_data_transform(train_data_path, test_data_path)
        logging.info(f"Transformation successful. Preprocessor saved at: {preprocessor_path}")

    except Exception as e:
        logging.error(f"Error during data transformation: {e}")
        raise
