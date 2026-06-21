import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the model
model_path = hf_hub_download(repo_id="neerajkumarlive/pacakge-prediction-model", 
                             filename="best_tourism_pacakge_prediction_model_v1.joblib")
model = joblib.load(model_path)

# Streamlit UI for Tourism Package Predictionn
st.title("Tourism Package Prediction App")
st.write("""
This application predicts the Tourism Package will be taken by customer based on its operational parameters.
Please enter the customer data below to get a prediction.
""")

# User input
age = st.number_input("Age", min_value=18, max_value=100, value=30)

typeof_contact = st.selectbox(
    "Type of Contact",
    ["Self Enquiry", "Company Invited"]
)

city_tier = st.selectbox(
    "City Tier",
    [1, 2, 3]
)

duration_of_pitch = st.number_input(
    "Duration of Pitch (minutes)",
    min_value=0,
    value=15
)

occupation = st.selectbox(
    "Occupation",
    ["Salaried", "Small Business", "Large Business", "Free Lancer"]
)

gender = st.radio(
    "Gender",
    ["Male", "Female"]
)

num_person_visiting = st.number_input(
    "Number of Persons Visiting",
    min_value=1,
    value=2
)

num_followups = st.number_input(
    "Number of Follow-ups",
    min_value=0,
    value=2
)

product_pitched = st.selectbox(
    "Product Pitched",
    ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"]
)

preferred_property_star = st.selectbox(
    "Preferred Property Star",
    [1, 2, 3, 4, 5]
)

marital_status = st.selectbox(
    "Marital Status",
    ["Single", "Married", "Divorced"]
)

num_trips = st.number_input(
    "Number of Trips",
    min_value=0,
    value=2
)

passport = st.selectbox(
    "Passport",
    ["Yes", "No"]
)

passport = 1 if passport == "Yes" else 0

pitch_satisfaction_score = st.slider(
    "Pitch Satisfaction Score",
    min_value=1,
    max_value=5,
    value=3
)

own_car = st.selectbox(
    "Own Car",
    ["Yes", "No"]
)

own_car = 1 if own_car == "Yes" else 0

num_children_visiting = st.number_input(
    "Number of Children Visiting",
    min_value=0,
    value=0
)

designation = st.selectbox(
    "Designation",
    ["Executive", "Manager", "Senior Manager", "AVP", "VP"]
)

monthly_income = st.number_input(
    "Monthly Income",
    min_value=0,
    value=20000
)

# Assemble input into DataFrame
input_data = pd.DataFrame([{
        "Age": age,
        "TypeofContact": typeof_contact,
        "CityTier": city_tier,
        "DurationOfPitch": duration_of_pitch,
        "Occupation": occupation,
        "Gender": gender,
        "NumberOfPersonVisiting": num_person_visiting,
        "NumberOfFollowups": num_followups,
        "ProductPitched": product_pitched,
        "PreferredPropertyStar": preferred_property_star,
        "MaritalStatus": marital_status,
        "NumberOfTrips": num_trips,
        "Passport": passport,
        "PitchSatisfactionScore": pitch_satisfaction_score,
        "OwnCar": own_car,
        "NumberOfChildrenVisiting": num_children_visiting,
        "Designation": designation,
        "MonthlyIncome": monthly_income
    }])


if st.button("Predict Product Taken"):
    prediction = model.predict(input_data)[0]
    result = "Product Taken" if prediction == 1 else "Product Not Taken"
    st.subheader("Prediction Result:")
    st.success(f"The model predicts: **{result}**")
