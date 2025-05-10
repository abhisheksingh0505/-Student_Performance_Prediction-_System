import sys
from sklearn.metrics import r2_score
from src.mlproject.exception import CustomException
from src.mlproject.logger import logging
from src.mlproject.utils import save_object, evaluate_models
import mlflow
from urllib.parse import urlparse
from src.mlproject.config.configuration import ModelTrainerConfig

from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

class ModelTrainer:
    def __init__(self):
        from mlproject.config.configuration import ModelTrainerConfig
        self.model_trainer_config = ModelTrainerConfig()

    def eval_metrics(self, actual, pred):
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        rmse = mean_squared_error(actual, pred, squared=False)
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)
        return rmse, mae, r2

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and test input data")
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }

            # Placeholder for hyperparameter tuning if needed
            params = {model: {} for model in models}

            model_report = evaluate_models(X_train, y_train, X_test, y_test, models, params)

            best_model_score = max(model_report.values())
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]

            logging.info(f"Best Model: {best_model_name} with a score of {best_model_score:.4f}")

            # MLflow Logging
            mlflow.set_registry_uri("https://dagshub.com/singh050530/mlproject.mlflow")
            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

            with mlflow.start_run():
                predicted = best_model.predict(X_test)
                rmse, mae, r2 = self.eval_metrics(y_test, predicted)

                try:
                    mlflow.log_params(params[best_model_name])
                except Exception as e:
                    logging.warning(f"MLFlow log_params failed: {e}")

                mlflow.log_metric("rmse", rmse)
                mlflow.log_metric("mae", mae)
                mlflow.log_metric("r2", r2)

                if tracking_url_type_store != "file":
                    mlflow.sklearn.log_model(best_model, "model", registered_model_name=best_model_name)
                else:
                    mlflow.sklearn.log_model(best_model, "model")

            logging.info("Saving best model to local artifacts folder")
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            return f"Best Model: {best_model_name} | R² Score: {r2:.4f}"

        except Exception as e:
            raise CustomException(e, sys)
