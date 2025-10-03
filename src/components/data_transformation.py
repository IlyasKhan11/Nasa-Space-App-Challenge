import os
import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', "preprocessor.pkl")
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')

# Define transformation functions at module level (not inside class methods)
def log_transform(X):
    """Log transformation function that can be pickled"""
    return np.log1p(X)

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        # Create artifacts directory if it doesn't exist
        os.makedirs(os.path.dirname(self.data_transformation_config.train_data_path), exist_ok=True)

    def select_features(self, data):
        """
        Select relevant features for the model
        """
        try:
            features = data[[
                'koi_period', 'koi_duration', 'koi_depth', 'koi_prad',
                'koi_teq', 'koi_insol', 'koi_model_snr', 'koi_steff',
                'koi_slogg', 'koi_srad', 'koi_kepmag', 'koi_fpflag_nt',
                'koi_fpflag_ss', 'koi_fpflag_co', 'koi_fpflag_ec',
                'koi_disposition'
            ]].copy()  # Use .copy() to avoid SettingWithCopyWarning
            return features
        except Exception as e:
            raise CustomException(e, sys)

    def get_data_transformer_object(self):
        '''
        This function is responsible for data transformation
        Creates separate pipelines for different column types
        '''
        try:
            # Columns that need log transformation (skewed columns)
            log_transform_columns = ['koi_prad', 'koi_depth', 'koi_teq', 'koi_insol', 'koi_model_snr']
            
            # Columns that need standard scaling (normally distributed)
            standard_columns = ['koi_period', 'koi_duration', 'koi_steff', 'koi_slogg', 'koi_srad', 'koi_kepmag']
            
            # Binary flag columns
            flag_columns = ['koi_fpflag_nt', 'koi_fpflag_ss', 'koi_fpflag_co', 'koi_fpflag_ec']

            # Pipeline for log transformed columns
            log_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('log_transform', FunctionTransformer(log_transform, validate=False)),
                ('scaler', StandardScaler())
            ])

            # Pipeline for standard columns
            standard_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])

            # Pipeline for flag columns
            flag_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])

            logging.info(f"Log transform columns: {log_transform_columns}")
            logging.info(f"Standard columns: {standard_columns}")
            logging.info(f"Flag columns: {flag_columns}")

            preprocessor = ColumnTransformer([
                ('log_pipeline', log_pipeline, log_transform_columns),
                ('standard_pipeline', standard_pipeline, standard_columns),
                ('flag_pipeline', flag_pipeline, flag_columns)
            ])

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def data_cleaning(self, data):
        """
        Basic data cleaning - remove specific rows and handle missing values
        """
        try:
            # Create a copy to avoid SettingWithCopyWarning
            data_clean = data.copy()
            
            # Remove the specific row as in original code
            data_clean = data_clean[data_clean['koi_fpflag_nt'] != 465]
            
            # Encode target variable
            data_clean['koi_disposition'] = data_clean['koi_disposition'].map({
                'FALSE POSITIVE': 0,
                'CANDIDATE': 1,
                'CONFIRMED': 2
            })
            
            return data_clean
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, raw_data_file_path):
        """
        Main method to initiate data transformation
        """
        try:
            # Read raw data
            logging.info("Reading raw data")
            raw_df = pd.read_csv(raw_data_file_path)
            
            # Feature selection
            logging.info("Selecting features")
            selected_df = self.select_features(raw_df)
            
            # Data cleaning
            logging.info("Cleaning data")
            cleaned_df = self.data_cleaning(selected_df)
            
            # Separate features and target
            logging.info("Separating features and target")
            X = cleaned_df.drop(columns=['koi_disposition'])
            y = cleaned_df['koi_disposition']
            
            # Train-test split
            logging.info("Performing train-test split")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Get preprocessor object
            logging.info("Obtaining preprocessing object")
            preprocessing_obj = self.get_data_transformer_object()
            
            # Apply preprocessing
            logging.info("Applying preprocessing object on training and testing dataframes")
            X_train_transformed = preprocessing_obj.fit_transform(X_train)
            X_test_transformed = preprocessing_obj.transform(X_test)
            
            # Combine features and target
            train_arr = np.c_[X_train_transformed, y_train.values]
            test_arr = np.c_[X_test_transformed, y_test.values]
            
            # Save processed data
            logging.info("Saving processed data")
            pd.DataFrame(train_arr).to_csv(self.data_transformation_config.train_data_path, index=False)
            pd.DataFrame(test_arr).to_csv(self.data_transformation_config.test_data_path, index=False)
            
            # Save preprocessor object
            logging.info("Saving preprocessing object")
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            
            logging.info(f"Data transformation completed successfully!")
            logging.info(f"Train data saved to: {self.data_transformation_config.train_data_path}")
            logging.info(f"Test data saved to: {self.data_transformation_config.test_data_path}")
            logging.info(f"Preprocessor saved to: {self.data_transformation_config.preprocessor_obj_file_path}")
            
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
            
        except Exception as e:
            logging.error("Error in data transformation")
            raise CustomException(e, sys)


# Add this part to actually run the transformation
if __name__ == "__main__":
    try:
        # You need to specify the path to your raw data file
        raw_data_path = "artifacts/data.csv"  # Update this path to your actual data file
        
        # Check if the raw data file exists
        if not os.path.exists(raw_data_path):
            print(f"Error: Raw data file not found at {raw_data_path}")
            print("Please update the 'raw_data_path' variable with the correct path to your data file")
            sys.exit(1)
        
        # Initialize and run data transformation
        data_transformation = DataTransformation()
        train_arr, test_arr, preprocessor_path = data_transformation.initiate_data_transformation(raw_data_path)
        
        print("Data transformation completed successfully!")
        print(f"Train data shape: {train_arr.shape}")
        print(f"Test data shape: {test_arr.shape}")
        print(f"Preprocessor saved at: {preprocessor_path}")
        
    except Exception as e:
        print(f"Error during data transformation: {e}")
        import traceback
        traceback.print_exc()