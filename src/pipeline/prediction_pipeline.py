import sys
import os
import pandas as pd
import numpy as np
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            # Load preprocessor and model
            model_path = 'artifacts/model.pkl'
            preprocessor_path = 'artifacts/preprocessor.pkl'
            
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            
            # Apply the same transformations as in training
            # Apply log transformation to skewed columns
            skewed_columns = ['koi_prad', 'koi_depth', 'koi_teq', 'koi_insol', 'koi_model_snr']
            for col in skewed_columns:
                if col in features.columns:
                    features[col] = np.log1p(features[col])
            
            # Drop original skewed columns
            # features = features.drop(columns=skewed_columns, errors='ignore')
            
            # Transform features using preprocessor
            data_scaled = preprocessor.transform(features)
            
            # Make prediction
            preds = model.predict(data_scaled)
            return preds
            
        except Exception as e:
            raise CustomException(e, sys)

# You might need to update your utils.py to include load_object