import os
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import shutil
import pandas as pd

experiment_type = "local"

if experiment_type == "local":
    mlflow.set_tracking_uri("http://localhost:5000")    # for local mlflow tracking
    mlflow.set_experiment("trial-run")
else:
    mlflow.set_tracking_uri("databricks")    # for cloud tracking
    mlflow.set_experiment("/Users/shivam94000@gmail.com/trial-run")
    
data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, 
                                                    test_size=0.2, random_state=42)
n_estimators = 100
max_depth = 5

#mlflow.create_experiment('test', artifact_location='C:\github_projects\data_science\Deployment\docker\artifacts')
with mlflow.start_run():
    params = {"n_estimators": n_estimators, 
              "max_depth": max_depth, 
              "random_state": 42}
    mlflow.log_params(params)
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    mlflow.log_metric("accuracy", acc)


    if os.path.exists("model"):
        shutil.rmtree("model")

    signature = mlflow.models.infer_signature(X_train, model.predict(X_train))
    input_example = X_test[:2]
    #mlflow.sklearn.log_model(model, artifact_path="model")
    # mlflow.sklearn.log_model(model, artifact_path="model", registered_model_name="iris_rf_model",
    #                          signature=signature, input_example=input_example)
    # mlflow.sklearn.save_model(model, path="model")
    # mlflow.log_artifacts("model", artifact_path="model")
    #pd.DataFrame(X_train).to_csv("iris.csv", index=False)
    #mlflow.log_artifact("iris.csv")
    print('Artifact URI:', mlflow.get_artifact_uri())
    pd.DataFrame(X_train).to_csv("iris.csv", index=False)
    mlflow.log_artifact("iris.csv")

    print(f"Accuracy: {acc:.4f}")

# import mlflow
# import os

# mlflow.set_tracking_uri("http://localhost:5000")
# mlflow.set_experiment("artifact-debug")

# with mlflow.start_run():
#     with open("test_artifact.txt", "w") as f:
#         f.write("Hello Shivam!")
#     mlflow.log_artifact("test_artifact.txt")
#     mlflow.end_run()
