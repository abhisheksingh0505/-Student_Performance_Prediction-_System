"""
ML Project Configuration Settings
All paths use 'artifact' (singular) directory instead of 'artifacts'
"""

import os
from dataclasses import dataclass
from pathlib import Path

# Base paths - using 'artifact' instead of 'artifacts'
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
ARTIFACT_DIR = PROJECT_ROOT / 'artifact'
os.makedirs(ARTIFACT_DIR, exist_ok=True)  # Creates directory if doesn't exist

@dataclass(frozen=True)
class DataIngestionConfig:
    """Configurations for Data Ingestion"""
    raw_data_path: str = str(ARTIFACT_DIR / 'raw_data.csv')
    train_data_path: str = str(ARTIFACT_DIR / 'train.csv') 
    test_data_path: str = str(ARTIFACT_DIR / 'test.csv')
    test_size: float = 0.2
    random_state: int = 42

@dataclass(frozen=True) 
class DataTransformationConfig:
    """Configurations for Data Transformation"""
    preprocessor_path: str = str(ARTIFACT_DIR / 'preprocessor.pkl')
    train_array_path: str = str(ARTIFACT_DIR / 'train_array.npy')
    test_array_path: str = str(ARTIFACT_DIR / 'test_array.npy')

@dataclass(frozen=True)
class ModelTrainerConfig:
    """Configurations for Model Training""" 
    trained_model_path: str = str(ARTIFACT_DIR / 'model.pkl')
    base_metrics_path: str = str(ARTIFACT_DIR / 'metrics.json')

@dataclass
class ConfigurationManager:
    """Central manager for all configurations"""
    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        return DataIngestionConfig()
    
    def get_data_transformation_config(self) -> DataTransformationConfig:
        return DataTransformationConfig()
    
    def get_model_trainer_config(self) -> ModelTrainerConfig:
        return ModelTrainerConfig()

# Quick access function
def get_config():
    return ConfigurationManager()