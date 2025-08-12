FROM apache/airflow:2.11.0

USER root

# Install system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ENV FILE_PATH_1=/opt/airflow/data/input/
ENV FILE_PATH_2=/opt/airflow/input/
ENV CONFIG_ENV=docker

# Create mlflow directory with proper permissions
RUN mkdir -p /mlflow/artifacts && \
    chown -R 50000:0 /mlflow && \
    chmod -R 755 /mlflow

USER airflow

# Set the Airflow home and working directories
ENV AIRFLOW_HOME=/opt/airflow
WORKDIR $AIRFLOW_HOME

# Set PYTHONPATH environment variable
ENV PYTHONPATH=/opt/airflow

# Copy requirements file
COPY requirements.txt /requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r /requirements.txt

# Create necessary directories
RUN mkdir -p /opt/airflow/features
RUN mkdir -p /opt/airflow/mlflow

# Set environment variables
ENV AIRFLOW__CORE__LOAD_EXAMPLES=False
ENV AIRFLOW__CORE__DAGS_ARE_PAUSED_AT_CREATION=True
ENV PYTHONPATH="${PYTHONPATH}:/opt/airflow/features"

# Initialize airflow database
# RUN airflow db init

# # Create airflow user
# RUN airflow users create \
#     --username admin \
#     --firstname Admin \
#     --lastname User \
#     --role Admin \
#     --email admin@example.com \
#     --password admin