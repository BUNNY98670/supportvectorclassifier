import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_csv("data.csv")

# Drop unnecessary columns
if "id" in df.columns:
    df = df.drop("id", axis=1)

# Remove unnamed column if present
if "Unnamed: 32" in df.columns:
    df = df.drop("Unnamed: 32", axis=1)

# Convert target labels
df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})

# Features and target
X = df.drop("diagnosis", axis=1)
y = df["diagnosis"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Train model
model = SVC(kernel="rbf", C=1, gamma="scale")
model.fit(X_train_scaled, y_train)

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("Breast Cancer Diagnosis Prediction using SVM")
st.write("Enter the feature values below.")

inputs = []

# Create one input box for each feature
for feature in X.columns:
    value = st.number_input(feature, value=0.0, format="%.6f")
    inputs.append(value)

# Predict
if st.button("Predict"):
    input_array = np.array([inputs])
    input_scaled = scaler.transform(input_array)
    prediction = model.predict(input_scaled)[0]

    if prediction == 1:
        st.error("Prediction: Malignant (Cancerous)")
    else:
        st.success("Prediction: Benign (Non-Cancerous)")