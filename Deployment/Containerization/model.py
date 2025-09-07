from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import pickle

# Train and save the model
def train_model():
    # Load dataset
    data = load_iris()
    X, y = data.data, data.target
    
    # Train model
    model = RandomForestClassifier()
    model.fit(X, y)
    
    # Save the trained model
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("Model trained and saved as model.pkl")

# Load model and make a prediction using predefined test data
def predict():
    # Load the saved model
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    # Test data (sample input for prediction)
    test_data = [5.1, 3.5, 1.4, 0.2]  # Example features
    prediction = model.predict([test_data])
    
    print(f"Prediction for {test_data}: {int(prediction[0])}")

if __name__ == '__main__':
    train_model()
    predict()