import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle


def load_data(file_path):
    """Loads and returns the dataset."""
    return pd.read_csv(file_path)


def preprocess_data(data):
    """
    Prepares data for modeling.
    Separates features (X) and target (Y), and splits into training and testing sets.
    """
    X = data.drop(columns='target', axis=1)
    Y = data['target']
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, stratify=Y, random_state=2)
    return X_train, X_test, Y_train, Y_test


def train_model(X_train, Y_train):
    """Trains a Logistic Regression model and returns it."""
    model = LogisticRegression()
    model.fit(X_train, Y_train)
    return model


def evaluate_model(model, X_train, Y_train, X_test, Y_test):
    """
    Evaluates the model on training and testing data, and prints the accuracy.
    """
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    train_accuracy = accuracy_score(train_pred, Y_train)
    test_accuracy = accuracy_score(test_pred, Y_test)
    print(f'Accuracy on Training Data: {train_accuracy:.2f}')
    print(f'Accuracy on Testing Data: {test_accuracy:.2f}')
    return train_accuracy, test_accuracy


def predict(model, input_data):
    """
    Predicts the outcome for a single instance.
    """
    input_data_np = np.asarray(input_data).reshape(1, -1)
    prediction = model.predict(input_data_np)
    result = 'The Person has Heart Disease' if prediction[0] == 1 else 'The Person does not have a Heart Disease'
    return result


def save_model(model, filename):
    """Saves the trained model to a file."""
    pickle.dump(model, open(filename, 'wb'))


def load_model(filename):
    """Loads a saved model from a file."""
    return pickle.load(open(filename, 'rb'))


if __name__ == "__main__":
    # File path to the dataset
    file_path = 'heart.csv'
    
    # Load and preprocess data
    heart_data = load_data(file_path)
    X_train, X_test, Y_train, Y_test = preprocess_data(heart_data)
    
    # Train the model
    model = train_model(X_train, Y_train)
    
    # Evaluate the model
    evaluate_model(model, X_train, Y_train, X_test, Y_test)
    
    # Predict for a single input
    sample_input = (62, 0, 0, 140, 268, 0, 0, 160, 0, 3.6, 0, 2, 2)
    prediction_result = predict(model, sample_input)
    print(prediction_result)
    
    # Save the model
    model_filename = 'heart_disease_model.sav'
    save_model(model, model_filename)
    
    # Load and test the saved model
    loaded_model = load_model(model_filename)
    print(f"Saved model prediction: {predict(loaded_model, sample_input)}")
