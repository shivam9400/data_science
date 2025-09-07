# Dockerizing a Machine Learning Model

This project demonstrates how to train a simple **RandomForest model** on the Iris dataset and run it inside a **Docker container**.

---

## Project Structure
```text
.
├── model.py
├── requirements.txt
└── Dockerfile
```

---

## requirements.txt
```text
scikit-learn
```
---

## Dockerfile
```text
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt requirements.txt
COPY model.py model.py
RUN pip install -r requirements.txt
CMD ["python", "model.py"]
```
---

## Build and Run
```text
# Build the Docker image
docker build -t ml-model .

# Run the container
docker run ml-model
```
### Expected Output
```text
Model trained and saved as model.pkl
Prediction for [5.1, 3.5, 1.4, 0.2]: 0
```
---

## Push to DockerHub
```text
# Login to DockerHub
docker login

# Tag the image
docker tag ml-model yourdockerhubusername/ml-model

# Push the image
docker push yourdockerhubusername/ml-model
```
---

## Pull and Run Anywhere
```text
docker pull yourdockerhubusername/ml-model
docker run yourdockerhubusername/ml-model
```