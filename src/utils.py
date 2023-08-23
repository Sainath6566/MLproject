import sys
import os
import numpy as np
import pandas as pd
import dill

from src.exception import CustomException
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)

def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}

        for i, model_name in enumerate(models.keys()):
            model = models[model_name]  # Get the model instance
            params = param.get(model_name, {})  # Get hyperparameter search space for the model

            gs = GridSearchCV(model, params, cv=3)
            gs.fit(X_train, y_train)
            best_params = gs.best_params_
            model.set_params(**best_params)
            model.fit(X_train, y_train)  # Train the model using the training data
            
            y_train_pred = model.predict(X_train)  # Predictions on the training set
            y_test_pred = model.predict(X_test)    # Predictions on the test set

            train_model_score = r2_score(y_train, y_train_pred)  # Calculate R^2 score on the training predictions
            test_model_score = r2_score(y_test, y_test_pred)     # Calculate R^2 score on the test predictions

            report[model_name] = test_model_score  # Store the test R^2 score in the report dictionary

        return report  # Return the dictionary containing test R^2 scores for each model
    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path):
    try:
        with open(file_path, 'rb') as file_obj:
            return dill.load(file_obj)  
    except Exception as e:
        raise CustomException(e, sys)
