from sqlalchemy import create_engine
import os
import sys
import pandas as pd
from dotenv import load_dotenv
from src.mlproject.exception import CustomException
from src.mlproject.logger import logging

import pickle
import numpy as np

# Load environment variables
load_dotenv()
host = os.getenv("host")
user = os.getenv("user")
password = os.getenv("password")
port = os.getenv("port") 
db = os.getenv("db")



SQLALCHEMY_DATABASE_URL = "mysql+pymysql://root:yourpassword@localhost:3306/mlproject"

print("Connecting using:", SQLALCHEMY_DATABASE_URL)

engine = create_engine(SQLALCHEMY_DATABASE_URL)

from urllib.parse import quote_plus

password_encoded = quote_plus(password)





def read_sql_data():
    logging.info("Reading SQL database started...")
    try:
        # Create SQLAlchemy engine using pymysql as the DBAPI
        connection_str = f"mysql+pymysql://{user}:{password_encoded}@{host}:{port}/{db}"
        engine = create_engine(connection_str)

        # Read data from SQL using the engine created by SQLAlchemy
        df = pd.read_sql_query('SELECT * FROM student', con=engine)
        logging.info(f"Data retrieved successfully: {df.shape[0]} rows")

        return df

    except Exception as e:
        raise CustomException(e, sys)
    
def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e,sys)
    
    
