import streamlit as st
import pandas as pd
import joblib

model = joblib.load("food_model.pkl")

st.title("Food Health Predictor")

calories = st.number_input("Calories")
protein = st.number_input("Protein")
carbs = st.number_input("Carbs")
fat = st.number_input("Fat")
iron = st.number_input("Iron")
vitamin_c = st.number_input("Vitamin C")

if st.button("Predict"):
    sample = pd.DataFrame([{
        'calories': calories,
        'protein': protein,
        'carbs': carbs,
        'fat': fat,
        'iron': iron,
        'vitamin_c': vitamin_c
    }])

    result = model.predict(sample)[0].strip()
    st.success("Result: " + result)