import pickle
import numpy as np
import pandas as pd
import streamlit as st

# Load trained model and MultiLabelBinarizer
with open('random_forest.pkl', 'rb') as f:
    model_data = pickle.load(f)

model = model_data['model']
mlb = model_data['mlb']

# Load the dataset containing disease suggestions
suggestions_data = pd.read_csv('symptoms.csv')  # Ensure this file has necessary details

# Set Page Config
st.set_page_config(page_title="AI Companion for Personalized Healthcare", page_icon="🤖", layout="wide")

# Custom CSS for styling
st.markdown("""
    <style>
        .main { background-color: #F9F9F9; }
        .title { color: #004D40; text-align: center; font-size: 36px; font-weight: bold; }
        .info-card { background-color: #E3F2FD; padding: 15px; border-radius: 10px; margin: 10px 0; }
        .prediction-box { background-color: #FFEBEE; padding: 20px; border-radius: 10px; text-align: center; font-size: 24px; font-weight: bold; color: #B71C1C; }
        .details-box { background-color: #D1C4E9; padding: 15px; border-radius: 10px; margin: 10px 0; color: #4A148C; }
    </style>
""", unsafe_allow_html=True)

# App Title
st.markdown('<h1 class="title">🤖 AI Companion for Personalized Healthcare</h1>', unsafe_allow_html=True)

# Sidebar for About Us
with st.sidebar:
    st.header("About Us 🏥")
    st.write("Our model predicts diseases based on symptoms and provides healthcare recommendations. "
             "Simply select your symptoms, and our AI will analyze them to suggest the most likely disease.")

# Select Symptoms
st.subheader("🔍 Select Symptoms")
selected_symptoms = st.multiselect("Choose from the list below:", options=mlb.classes_)

# Prediction Button
if st.button("🔮 Predict Disease"):
    if selected_symptoms:
        # Convert symptoms to input vector
        input_vector = np.zeros(len(mlb.classes_))
        for symptom in selected_symptoms:
            if symptom in mlb.classes_:
                input_vector[mlb.classes_.tolist().index(symptom)] = 1

        # Predict disease
        predicted_disease = model.predict([input_vector])[0]

        # Fetch suggestions for the predicted disease
        suggestions = suggestions_data[suggestions_data['Disease'] == predicted_disease].iloc[0]

        # Display Prediction
        st.markdown(f'<div class="prediction-box">Predicted Disease: <b>{predicted_disease}</b></div>', unsafe_allow_html=True)

        # Display Disease Details
        st.markdown("<h3>📌 Disease Information</h3>", unsafe_allow_html=True)
        st.markdown(f'<div class="details-box"><b>Description:</b> {suggestions["Description"]}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="details-box"><b>Prescription:</b> {suggestions["Prescription"]}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="details-box"><b>Precautions:</b> {suggestions["Precautions"]}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="details-box"><b>Diet Plan:</b> {suggestions["Diet Plans"]}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="details-box"><b>Workouts:</b> {suggestions["Workouts"]}</div>', unsafe_allow_html=True)
    else:
        st.warning("⚠️ Please select at least one symptom.")

# Footer
st.markdown("---")
st.markdown('<div class="info-card"><b>Disclaimer:</b> This model is for educational purposes only. Consult a medical professional for accurate diagnosis.</div>', unsafe_allow_html=True)
