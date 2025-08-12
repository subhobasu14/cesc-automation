-- Create MLflow database
CREATE DATABASE mlflow;

-- Grant privileges to airflow user for MLflow database
GRANT ALL PRIVILEGES ON DATABASE mlflow TO airflow;