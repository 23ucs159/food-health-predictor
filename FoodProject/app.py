import streamlit as st
import pandas as pd
import joblib
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "food_model.pkl")
model = joblib.load(model_path)

st.set_page_config(page_title="Food Health Predictor", page_icon="🍎")

st.title("🍎 Food Health Predictor")
st.write("Enter nutritional values to check if food is Healthy, Moderate, or Unhealthy")

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

    result = model.predict(sample)[0].strip().lower()

    if result == "healthy":
        st.success("🟢 Healthy Food")
    elif result == "moderate":
        st.warning("🟡 Moderate Food")
    else:
        st.error("🔴 Unhealthy Food")
