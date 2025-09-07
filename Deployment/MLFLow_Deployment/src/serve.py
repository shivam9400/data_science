import mlflow

mlflow.set_tracking_uri("http://localhost:5000") 
client = mlflow.tracking.MlflowClient()
run_id = "4f5853b7afeb418ab73567f74b9325f9"

artifacts = client.list_artifacts(run_id)
print([a.path for a in artifacts])