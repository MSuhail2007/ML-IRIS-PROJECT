import streamlit as st
import joblib
import numpy as np
from sklearn.datasets import load_iris

# Load the trained model from GitHub
model = joblib.load("iris_model.pkl")

# Load Iris species names
iris = load_iris()

# Streamlit App UI
st.title("Iris Flower Classification 🌸")
st.write("Enter the flower measurements below and get a prediction!")

# Input fields for user to enter flower measurements
sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, format="%.2f")
sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, format="%.2f")
petal_length = st.number_input("Petal Length (cm)", min_value=0.0, format="%.2f")
petal_width = st.number_input("Petal Width (cm)", min_value=0.0, format="%.2f")

# Predict Button
if st.button("Predict"):
    # Prepare input data
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    
    # Make prediction
    prediction = model.predict(input_data)
    
    # Show result
    st.success(f"Predicted Species: {iris.target_names[prediction[0]]}")
