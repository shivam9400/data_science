import requests
import pandas as pd

# Replace with your real feature names and values
data = pd.DataFrame([[0.5]], columns=["feature"])

response = requests.post(
    "http://localhost:5001/invocations",
    json={"dataframe_split": data.to_dict(orient="split")}
)

print("Prediction:", response.json())