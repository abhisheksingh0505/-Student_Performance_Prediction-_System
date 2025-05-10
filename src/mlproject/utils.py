from sqlalchemy import create_engine
import os
import sys
import pandas as pd
from dotenv import load_dotenv
from src.mlproject.exception import CustomException
from src.mlproject.logger import logging
import pickle
import numpy as np
from urllib.parse import quote_plus
from sklearn.metrics import r2_score
from src.mlproject.exception import CustomException







load_dotenv()
host = os.getenv("host")
user = os.getenv("user")
password = os.getenv("password")
port = os.getenv("port")
db = os.getenv("db")

password_encoded = quote_plus(password)
SQLALCHEMY_DATABASE_URL = f"mysql+pymysql://{user}:{password_encoded}@{host}:{port}/{db}"

print("Connecting using:", SQLALCHEMY_DATABASE_URL)
engine = create_engine(SQLALCHEMY_DATABASE_URL)

def read_sql_data():
    logging.info("Reading SQL database started...")
    try:
        connection_str = f"mysql+pymysql://{user}:{password_encoded}@{host}:{port}/{db}"
        engine = create_engine(connection_str)
        df = pd.read_sql_query('SELECT * FROM student', con=engine)
        logging.info(f"Data retrieved successfully: {df.shape[0]} rows")
        return df
    except Exception as e:
        raise CustomException(e, sys)

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)

def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    try:
        report = {}

        for model_name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            score = r2_score(y_test, y_pred)
            report[model_name] = score

        return report

    except Exception as e:
        raise CustomException(e, sys)
       
