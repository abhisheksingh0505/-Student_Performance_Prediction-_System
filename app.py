import sys
import os
from pathlib import Path

# Fix 1: Add project root to Python path (critical)
sys.path.append(str(Path(__file__).parent.parent))

# Fix 2: Correct imports
try:
    from src.mlproject.config.configuration import DataIngestionConfig, ModelTrainerConfig
    from src.mlproject.components.data_ingestion import DataIngestion
    from src.mlproject.components.data_transformation import DataTransformation
    from src.mlproject.components.model_trainer import ModelTrainer
    from src.mlproject.exception import CustomException
    from src.mlproject.logger import logging
except ImportError as e:
    print(f"Import error! Check paths. Current sys.path: {sys.path}")
    raise

if __name__ == "__main__":
    logging.info("Execution started")
    
    try:
        # Data Ingestion
        data_ingestion = DataIngestion()
        train_path, test_path = data_ingestion.initiate_data_ingestion()

        # Data Transformation
        data_transformation = DataTransformation()
        train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_path, test_path)

        # Model Training
        model_trainer = ModelTrainer()
        print(model_trainer.initiate_model_trainer(train_arr, test_arr))
        
    except Exception as e:
        logging.error(f"Error: {str(e)}")
        raise CustomException(e, sys)