import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Food Health Dashboard",
    page_icon="🍎",
    layout="wide"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "food_model.pkl")
model = joblib.load(model_path)

st.markdown("""
# 🍎 Food Health Dashboard
### AI Powered Nutrition Analysis System
---
""")

col1, col2 = st.columns(2)

with col1:
    st.subheader("📥 Enter Nutrition Values")

    calories = st.number_input("Calories")
    protein = st.number_input("Protein")
    carbs = st.number_input("Carbs")
    fat = st.number_input("Fat")
    iron = st.number_input("Iron")
    vitamin_c = st.number_input("Vitamin C")

    predict_btn = st.button("🚀 Predict")

with col2:
    st.subheader("📊 Prediction Dashboard")

    if predict_btn:

        sample = pd.DataFrame([{
            'calories': calories,
            'protein': protein,
            'carbs': carbs,
            'fat': fat,
            'iron': iron,
            'vitamin_c': vitamin_c
        }])

        result = str(model.predict(sample)[0]).strip().lower()

        if result == "healthy":
            st.success("🟢 HEALTHY FOOD")
            status = "Healthy"
        elif result == "moderate":
            st.warning("🟡 MODERATE FOOD")
            status = "Moderate"
        else:
            st.error("🔴 UNHEALTHY FOOD")
            status = "Unhealthy"

        st.metric("Calories", calories)
        st.metric("Protein", protein)
        st.metric("Carbs", carbs)
        st.metric("Fat", fat)
        st.metric("Iron", iron)
        st.metric("Vitamin C", vitamin_c)

        labels = ["Calories", "Protein", "Carbs", "Fat", "Iron", "Vitamin C"]
        values = [calories, protein, carbs, fat, iron, vitamin_c]

        fig, ax = plt.subplots()
        ax.bar(labels, values)
        plt.xticks(rotation=45)
        st.pyplot(fig)

        st.info(f"Final Classification: {status}")

st.markdown("---")
st.markdown("Built with Streamlit")
