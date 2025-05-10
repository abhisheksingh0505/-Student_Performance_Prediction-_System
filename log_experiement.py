from dagshub import init
import mlflow

init(repo_owner='singh050530', repo_name='mlproject', mlflow=True)

with mlflow.start_run():
    mlflow.log_param('parameter name', 'value')
    mlflow.log_metric('metric name', 1)
