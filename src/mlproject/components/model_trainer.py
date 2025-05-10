import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging

@dataclass
class ModelTrainerConfig:
    train_model_file_path=os.path.join("artifact","model.pkl")
class MOdelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self, train_array,test_array):
        try:
            logging.info("Spliting training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,:-1],
                test_array[:,:-1],
                test_array[:,:-1]

            )

            models = {
                "Linear Regression": LinearRegression(),
                 "Decision Tree": DecisionTreeRegressor(),
                 "Random Forest Regressor": RandomForestRegressor(),
                 "Gradient Boosting": GradientBoostingRegressor(),
                 "XGBRegressor": XGBRegressor(), 
                 "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                 "AdaBoost Regressor": AdaBoostRegressor()

            }
   
    

       
        except Exception as e:
            raise CustomException(e,sys)