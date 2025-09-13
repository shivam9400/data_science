import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Generate fake regression data
X, y = make_regression(n_samples=100, n_features=1, noise=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

mlflow.set_experiment("simple_linear_regression")

with mlflow.start_run():
    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    input_example = X_test[:1]
    # Log parameters, metrics and model
    mlflow.log_param("fit_intercept", model.fit_intercept)
    mlflow.log_metric("mse", mse)
    mlflow.sklearn.log_model(model,
                            name = "model",
                             input_example=input_example,
                             signature=None)

    result = mlflow.register_model(
        model_uri="runs:/"+mlflow.active_run().info.run_id+"/model",
        name="LinearRegressionModel"
        )
    
    print(f"Run complete. MSE: {mse:.2f}")