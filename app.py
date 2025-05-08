import sys
from src.mlproject.logger import logging
from src.mlproject.exception import CustomException
from src.mlproject.components.data_ingestion import DataIngestion, DataIngestionConfig

if __name__ == "__main__":
    logging.info("The execution has started")

    try:
        config = DataIngestionConfig()  # Config object is now passed correctly
        data_ingestion = DataIngestion(config)
        data_ingestion.initiate_data_ingestion()

    except Exception as e:
        logging.info("Exception occurred during ingestion")
        raise CustomException(e, sys)
