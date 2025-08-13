#!/bin/bash
# Setup script for Airflow + MLflow Docker environment
echo "Setting up Airflow + MLflow Docker environment..."
# Create necessary directories
echo "Creating directories..."
# mkdir -p dags
# mkdir -p features
mkdir -p logs
mkdir -p plugins
mkdir -p output
# mkdir -p core
# mkdir -p data
mkdir -p model
mkdir -p mlflow/artifacts
# Set proper permissions
echo "Setting permissions..."
# sudo chown -R 50000:0 dags logs plugins mlflow output core data model
# sudo chmod -R 755 dags logs plugins mlflow output core data model
# sudo chown -R 50000:0 dags logs plugins mlflow output core data model
chmod -R 777 dags logs plugins mlflow output core data model

echo "Installing Python dependencies..."
pip install cryptography
# Copy sample files if they don't exist
# if [ ! -f "dags/sample_dag.py" ]; then
#     echo "# Place your DAG files here" > dags/README.md
# fi
if [ ! -f "features/__init__.py" ]; then
    touch features/__init__.py
fi
# Generate Fernet key for Airflow
echo "Generating Fernet key..."
FERNET_KEY=$(python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())")
# Update .env file with generated Fernet key
if [ -f ".env" ]; then
    sed -i "s/your-fernet-key-here/$FERNET_KEY/" .env
    echo "Updated Fernet key in .env file"
fi
# Initialize Airflow (this will be done in the container, but we can prepare)
echo "Environment setup complete!"
echo ""
echo "To start the services:"
echo "1. Run: docker-compose up -d"
echo "2. Wait for services to be healthy"
echo "3. Access Airflow at: http://localhost:8080 (admin/admin)"
echo "4. Access MLflow at: http://localhost:5000"
echo ""
echo "To stop the services:"
echo "docker-compose down"
echo ""
echo "To view logs:"
echo "docker-compose logs -f [service_name]"
