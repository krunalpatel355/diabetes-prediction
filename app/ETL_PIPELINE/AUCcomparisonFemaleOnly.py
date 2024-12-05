#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from imblearn.under_sampling import RandomUnderSampler

# Load the data
data = pd.read_csv(r"C:\Users\Hazel\Downloads\DiabetesDatasets\diabetes.csv")

# Separate features and target
X = data.drop(columns=["Outcome"])
y = data["Outcome"]

# Split the data before any transformations to prevent data leakage
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale the data (scaling applied only after splitting to avoid data leakage)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Apply undersampling only to the training data
rus = RandomUnderSampler(random_state=42)
X_train_resampled, y_train_resampled = rus.fit_resample(X_train_scaled, y_train)

# Define models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "Naive Bayes": GaussianNB(),
    "Support Vector Machine": SVC(probability=True),
    "Random Forest": RandomForestClassifier()
}

# Dictionary to store results
results = {}

# Train and evaluate each model
for name, model in models.items():
    model.fit(X_train_resampled, y_train_resampled)
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = (
        model.predict_proba(X_test_scaled)[:, 1]
        if hasattr(model, "predict_proba")
        else model.decision_function(X_test_scaled)
    )
    
    # Calculate AUC
    auc = roc_auc_score(y_test, y_pred_proba)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    
    # Store results
    results[name] = {
        "AUC": auc,
        "Confusion Matrix": conf_matrix,
        "Classification Report": class_report
    }

# Output results
for name, metrics in results.items():
    print(f"Model: {name}")
    print(f"AUC: {metrics['AUC']}")
    print("Confusion Matrix:")
    print(metrics["Confusion Matrix"])
    print("Classification Report:")
    print(metrics["Classification Report"])
    print("=" * 50)


# In[ ]:




