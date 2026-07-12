import os
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st
from huggingface_hub import hf_hub_download

st.set_page_config(page_title="Tourism Package Predictor", page_icon="✈️", layout="wide")

MODEL_REPO = os.getenv("HF_MODEL_REPO", "YOUR_HF_USERNAME/tourism-package-model")
LOCAL_MODEL = Path(__file__).with_name("tourism_purchase_pipeline.joblib")

@st.cache_resource
def load_model():
    if LOCAL_MODEL.exists():
        return joblib.load(LOCAL_MODEL)
    path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename="tourism_purchase_pipeline.joblib",
        repo_type="model"
    )
    return joblib.load(path)

model = load_model()

st.title("Wellness Tourism Package Purchase Predictor")
st.caption("Predict whether a customer is likely to buy the new tourism package.")

with st.form("prediction_form"):
    c1, c2, c3 = st.columns(3)
    with c1:
        age = st.number_input("Age", 18, 90, 35)
        contact = st.selectbox("Type of Contact", ["Company Invited", "Self Enquiry"])
        city_tier = st.selectbox("City Tier", [1, 2, 3])
        occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
        gender = st.selectbox("Gender", ["Male", "Female"])
        persons = st.number_input("Number of Persons Visiting", 1, 10, 2)
    with c2:
        duration = st.number_input("Duration of Pitch (minutes)", 1.0, 120.0, 15.0)
        followups = st.number_input("Number of Follow-ups", 0.0, 20.0, 3.0)
        product = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
        property_star = st.selectbox("Preferred Property Star", [1.0, 2.0, 3.0, 4.0, 5.0], index=2)
        marital = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
        trips = st.number_input("Annual Number of Trips", 0.0, 30.0, 3.0)
    with c3:
        passport = st.selectbox("Has Passport", [0, 1], format_func=lambda x: "Yes" if x else "No")
        pitch_score = st.selectbox("Pitch Satisfaction Score", [1, 2, 3, 4, 5], index=2)
        own_car = st.selectbox("Owns Car", [0, 1], format_func=lambda x: "Yes" if x else "No")
        children = st.number_input("Children Visiting", 0.0, 10.0, 1.0)
        designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
        income = st.number_input("Monthly Income", 5000.0, 200000.0, 30000.0, step=1000.0)

    submitted = st.form_submit_button("Predict Purchase Probability", use_container_width=True)

if submitted:
    row = pd.DataFrame([{
        "Age": age,
        "TypeofContact": contact,
        "CityTier": city_tier,
        "DurationOfPitch": duration,
        "Occupation": occupation,
        "Gender": gender,
        "NumberOfPersonVisiting": persons,
        "NumberOfFollowups": followups,
        "ProductPitched": product,
        "PreferredPropertyStar": property_star,
        "MaritalStatus": marital,
        "NumberOfTrips": trips,
        "Passport": passport,
        "PitchSatisfactionScore": pitch_score,
        "OwnCar": own_car,
        "NumberOfChildrenVisiting": children,
        "Designation": designation,
        "MonthlyIncome": income
    }])
    probability = float(model.predict_proba(row)[0, 1])
    prediction = int(probability >= 0.5)
    st.metric("Purchase probability", f"{probability:.1%}")
    if prediction:
        st.success("High-potential customer: prioritize for the campaign.")
    else:
        st.info("Lower-potential customer: consider nurturing before direct outreach.")
