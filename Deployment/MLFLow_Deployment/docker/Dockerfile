# Use a lightweight Python base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your project files
COPY . /app

# Set environment variables (can be overridden at runtime)
ENV MLFLOW_TRACKING_URI=http://host.docker.internal:5000
ENV RUN_ID=4f5853b7afeb418ab73567f74b9325f9

# Expose MLflow's default serving port
EXPOSE 5000

# Serve the model from the specified run
CMD ["sh", "-c", "mlflow models serve -m runs:/$RUN_ID/model -p 5000 --no-conda"]