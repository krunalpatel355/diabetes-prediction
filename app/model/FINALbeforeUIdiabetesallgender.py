# -*- coding: utf-8 -*-
"""Diabetes Prediction: Modularized"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report, r2_score, mean_squared_error
from xgboost import XGBRegressor
import pickle

# Load dataset
def load_data(filepath):
    """Load the dataset from the specified filepath."""
    return pd.read_csv(filepath)

# Exploratory Data Analysis (EDA)
def plot_target_distribution(data, target_col):
    """Plot the distribution of the target variable."""
    plt.figure(figsize=(8, 4))
    sns.countplot(x=target_col, data=data, palette='Set2')
    plt.title("Distribution of Target Variable")
    plt.show()

# Correlation Analysis
def analyze_correlations(data, target_col):
    """Analyze correlations with the specified target column."""
    correlation_matrix = data.corr()
    print(f"Correlation with {target_col}:")
    print(correlation_matrix[target_col].sort_values(ascending=False))

# Impute Missing Blood Pressure
def impute_missing_bp(data, feature_sets):
    """
    Impute missing values in the 'BloodPressure' column using the best feature set
    and train a model with XGBoost for imputation.
    """
    def evaluate_feature_sets(data, feature_sets):
        best_model, best_scaler, best_features, best_r2_val = None, None, None, -float('inf')
        for label, features in feature_sets.items():
            BP_X, BP_y = data[features], data['BloodPressure']
            X_train, X_val, y_train, y_val = train_test_split(
                BP_X, BP_y, test_size=0.2, random_state=42
            )
            scaler = StandardScaler().fit(X_train)
            X_train_scaled, X_val_scaled = scaler.transform(X_train), scaler.transform(X_val)
            train_complete = X_train_scaled[~y_train.isna()]
            y_train_complete = y_train.dropna()
            if train_complete.size == 0: continue
            model = XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
            model.fit(train_complete, y_train_complete)
            r2_val = r2_score(y_val.dropna(), model.predict(X_val_scaled[~y_val.isna()]))
            if r2_val > best_r2_val:
                best_model, best_scaler, best_features, best_r2_val = model, scaler, features, r2_val
        return best_model, best_scaler, best_features

    def fill_missing_bp(data, model, scaler, features):
        missing_indices = data['BloodPressure'].isna()
        if missing_indices.any():
            BP_X_scaled = scaler.transform(data[features])
            data.loc[missing_indices, 'BloodPressure'] = np.round(model.predict(BP_X_scaled[missing_indices])).astype(int)
        return data

    model, scaler, features = evaluate_feature_sets(data, feature_sets)
    data = fill_missing_bp(data, model, scaler, features)
    return data, model, scaler, features

# Train Random Forest Model
def train_random_forest(X_train, y_train):
    """Train a Random Forest Classifier with balanced class weights."""
    rf = RandomForestClassifier(class_weight='balanced', random_state=42, n_estimators=100)
    rf.fit(X_train, y_train)
    return rf

# Evaluate Model
def evaluate_model(model, X_test, y_test):
    """Evaluate the model performance on the test set."""
    predictions = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    auc = roc_auc_score(y_test, proba) if proba is not None else None
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions))
    print("\nClassification Report:")
    print(classification_report(y_test, predictions))
    if auc:
        print(f"AUC: {auc:.2f}")
    return predictions

# Save and Load Model
def save_model(model, filename):
    """Save the trained model to a file."""
    pickle.dump(model, open(filename, 'wb'))
    print(f"Model saved as {filename}")

def load_model(filename):
    """Load a saved model from a file."""
    return pickle.load(open(filename, 'rb'))

# Prediction on New Data
def predict_new_data(model, new_data, feature_order):
    """Predict outcome for new data using the trained model."""
    input_data = np.asarray(new_data).reshape(1, -1)
    prediction = model.predict(input_data)
    proba = model.predict_proba(input_data)[0] if hasattr(model, "predict_proba") else None
    if prediction[0] == 1:
        print(f"\nYou are likely diabetic (Probability: {proba[1]*100:.2f}%)")
    else:
        print(f"\nYou are not likely diabetic (Probability: {proba[0]*100:.2f}%)")
    return prediction

### MAIN EXECUTION

# Filepath and Feature Set Configuration
filepath = "allgenderdiabetes.csv"
feature_sets = {
    "Set 1": ['BMI', 'Glucose', 'Insulin', 'Age'],
    "Set 2": ['BMI'],
    "Set 3": ['Age', 'Pregnancies', 'Glucose', 'SkinThickness', 'Insulin', 'BMI', 'FamilyHistory']
}
target_col = 'Outcome'

# Load data and EDA
data = load_data(filepath)
plot_target_distribution(data, target_col)
analyze_correlations(data, 'BloodPressure')

# Impute Missing BloodPressure
data, model_bp, scaler_bp, features_bp = impute_missing_bp(data, feature_sets)

# Split Data for Training and Testing
X = data.drop(columns=[target_col])
y = data[target_col]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Outcome Prediction Model
rf_model = train_random_forest(X_train, y_train)

# Evaluate the Outcome Prediction Model
evaluate_model(rf_model, X_test, y_test)

# Save the model
save_model(rf_model, "diabetes_rf_model.sav")

# Predict on new data
new_data_example = [0, 106, 78, 15, 21, 25.6, 0, 56]  # Replace with appropriate values
predict_new_data(rf_model, new_data_example, feature_order=X.columns.tolist())
