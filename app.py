import streamlit as st
import pandas as pd
import joblib

# 1. Load your saved pipeline
pipeline = joblib.load('travel_Model.joblib')

st.title("Wellness Travel Package Predictor")
st.write("Enter customer details below to predict whether they will purchase the package.")

# 2. Create input fields for all customer features
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=80, value=35)
    typeof_contact = st.selectbox("Type of Contact", ['Self Enquiry', 'Company Invited'])
    city_tier = st.selectbox("City Tier", [1, 2, 3])
    duration_of_pitch = st.number_input("Duration of Pitch (minutes)", min_value=1.0, max_value=60.0, value=12.0)
    occupation = st.selectbox("Occupation", ['Salaried', 'Small Business', 'Large Business', 'Free Lancer'])
    gender = st.selectbox("Gender", ['Female', 'Male'])
    num_person = st.number_input("Number of Adults Visiting", min_value=1, max_value=10, value=2)
    num_children = st.number_input("Number of Children Visiting", min_value=0, max_value=10, value=1)
    num_followups = st.number_input("Number of Followups", min_value=1, max_value=10, value=3)

with col2:
    product_pitched = st.selectbox("Product Pitched", ['Basic', 'Standard', 'Deluxe', 'Super Deluxe', 'King'])
    preferred_star = st.selectbox("Preferred Property Star", [3.0, 4.0, 5.0])
    marital_status = st.selectbox("Marital Status", ['Married', 'Divorced', 'Unmarried'])
    num_trips = st.number_input("Number of Trips per Year", min_value=0, max_value=25, value=2)
    passport = st.selectbox("Has Passport?", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
    pitch_score = st.slider("Pitch Satisfaction Score", min_value=1, max_value=5, value=3)
    own_car = st.selectbox("Owns a Car?", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
    designation = st.selectbox("Designation", ['Executive', 'Manager', 'Senior Manager', 'AVP', 'VP'])
    monthly_income = st.number_input("Monthly Income", min_value=1000, max_value=150000, value=20000)

# 3. Process inputs & predict when button is clicked
if st.button("Predict Purchase"):
    # Create DataFrame with the exact 17 columns expected by your preprocessor
    input_df = pd.DataFrame([{
        'Age': float(age),
        'TypeofContact': typeof_contact,
        'CityTier': int(city_tier),
        'DurationOfPitch': float(duration_of_pitch),
        'Occupation': occupation,
        'Gender': gender,
        'NumberOfFollowups': num_followups,
        'ProductPitched': product_pitched,
        'PreferredPropertyStar': float(preferred_star),
        'MaritalStatus': marital_status,
        'NumberOfTrips': num_trips,
        'Passport': int(passport),
        'PitchSatisfactionScore': int(pitch_score),
        'OwnCar': int(own_car),
        'Designation': designation,
        'MonthlyIncome': monthly_income,
        'TotalVisiting': int(num_person + num_children)
    }])

    # Make prediction (Note: if travel_Model.joblib only contains 'model' and not 'preprocessor', see note below)
    prediction = pipeline.predict(input_df)[0]
    probability = pipeline.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.success(f"Customer is LIKELY to buy the package! (Probability: {probability * 100:.1f}%)")
    else:
        st.info(f"Customer is UNLIKELY to buy the package. (Purchase Probability: {probability * 100:.1f}%)")
