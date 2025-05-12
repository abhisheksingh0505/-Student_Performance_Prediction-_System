import os
import sys
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

project_name = "mlproject"

# Define the complete project structure
list_of_files = [
    # Project root files
    "README.md",
    "requirements.txt",
    "setup.py",
    ".env",
    ".gitignore",
    ".dvcignore",
    "Dockerfile",
    "main.py",
    "app.py",
    
    # Source files
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/exception.py",
    f"src/{project_name}/logger.py",
    f"src/{project_name}/utils.py",
    
    # Configuration
    f"src/{project_name}/config/__init__.py",
    f"src/{project_name}/config/configuration.py",
    
    # Components
    f"src/{project_name}/components/__init__.py",
    f"src/{project_name}/components/data_ingestion.py",
    f"src/{project_name}/components/data_transformation.py",
    f"src/{project_name}/components/model_trainer.py",
    f"src/{project_name}/components/model_monitoring.py",
    
    # Pipelines
    f"src/{project_name}/pipelines/__init__.py",
    f"src/{project_name}/pipelines/training_pipeline.py",
    f"src/{project_name}/pipelines/prediction_pipeline.py",
    
    # Notebooks and data
    "notebooks/experiments.ipynb",
    "notebooks/data/raw.csv",
    
    # Tests
    "tests/__init__.py",
    "tests/test_components.py",
    
    # Artifacts (empty directories)
    "artifact/",
    "logs/"
]

def create_project_structure():
    """Creates the complete project directory and file structure"""
    try:
        for filepath in list_of_files:
            filepath = Path(filepath)
            filedir, filename = os.path.split(filepath)
            
            # Create directory if it doesn't exist
            if filedir != "":
                os.makedirs(filedir, exist_ok=True)
                logging.info(f"Created directory: {filedir}")
            
            # Create empty file if it doesn't exist or is empty
            if os.path.splitext(filepath)[1]:  # Only for files (not directories)
                if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
                    with open(filepath, 'w') as f:
                        if filepath.name == "configuration.py":
                            f.write(CONFIGURATION_TEMPLATE)
                        elif filepath.name == "__init__.py":
                            f.write("# Package initialization\n")
                        elif filepath.name == "requirements.txt":
                            f.write(REQUIREMENTS_TEMPLATE)
                    logging.info(f"Created file: {filepath}")
                else:
                    logging.info(f"File exists: {filepath}")
        
        logging.info("✅ Project structure created successfully!")
        return True
        
    except Exception as e:
        logging.error(f"❌ Failed to create project structure: {e}")
        return False

# Template content for key files
CONFIGURATION_TEMPLATE = '''import os
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifact', 'train.csv')
    test_data_path: str = os.path.join('artifact', 'test.csv')
    raw_data_path: str = os.path.join('artifact', 'raw.csv')

@dataclass
class DataTransformationConfig:
    preprocessor_path: str = os.path.join('artifact', 'preprocessor.pkl')

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join('artifact', 'model.pkl')
'''

REQUIREMENTS_TEMPLATE = '''pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
python-dotenv>=0.19.0
'''

if __name__ == "__main__":
    logging.info("🏗️ Building ML project structure...")
    if create_project_structure():
        logging.info("✨ Project setup completed!")
    else:
        logging.error("💥 Project setup failed")
        sys.exit(1)