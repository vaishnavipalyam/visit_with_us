import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download the model from the Model Hub
model_path = hf_hub_download(repo_id="VaishnaviPalyam/travel_customer_model", filename="best_travel_customer_model_v1.joblib")

# Load the model
model = joblib.load(model_path)

# Streamlit UI for Buying Customer Prediction
st.title("Customer Prediction App")
st.write("The Customer Prediction App is an internal tool for company staff that predicts whether customers are willing to buy the new wellness tourism package or not.")
st.write("Kindly enter the customer details to check whether they are likely to buy.")

# Collect user input
Age = st.number_input("Age (customer's age in years)", min_value=18, max_value=61, value=18)
NumberOfPersonVisiting = st.number_input("GTotal travellers", min_value=1, max_value=10, value=1)
NumberOfTrips = st.number_input("Average trips by the customer per year", min_value=1, max_value=25, value=1)
NumberOfChildrenVisiting = st.number_input("Number of children below 5 years", min_value=0, max_value=10, value=0)
MonthlyIncome = st.number_input("Monthly income", min_value=0.0, value=10000.0)
DurationOfPitch = st.number_input("Duration of pitch in minutes", min_value=1, value=1)
CityTier = st.selectbox("tier of city of residence", [1, 2, 3])
PreferredPropertyStar = st.selectbox("Preferred property for accomodation", [3, 4, 5])
Passport = st.selectbox("does the customer hold passport? (1:yes, 0:No)", [1, 0])
PitchSatisfactionScore = st.selectbox("Pitch satisfaction score", [1, 2, 3, 4, 5])
OwnCar = st.selectbox("Does the customer own car? (1:yes, 0:No)", [1, 0])
NumberOfFollowups = st.number_input("Total number of follow ups done by the sales person", min_value=0, value=0)
TypeofContact = st.selectbox("Type of contact", ['Self Enquiry', 'Company Invited'])
Occupation = st.selectbox("Occupation", ['Salaried', 'Free Lancer', 'Small Business', 'Large Business'])
Gender = st.selectbox("Gender", ['Male', 'Female'])
MaritalStatus = st.selectbox("Marital Status", ['Single', 'Divorced', 'Married', 'Unmarried'])
Designation = st.selectbox("Designation", ['Manager', 'Executive', 'Senior Manager', 'AVP', 'VP'])
ProductPitched = st.selectbox("Product Pitched", ['Deluxe', 'Basic', 'Standard', 'Super Deluxe', 'King'])

# Convert categorical inputs to match model training
input_data = pd.DataFrame([{
    'Age': Age,
    'NumberOfPersonVisiting': NumberOfPersonVisiting,
    'NumberOfTrips': NumberOfTrips,
    'NumberOfChildrenVisiting': NumberOfChildrenVisiting,
    'MonthlyIncome': MonthlyIncome,
    'DurationOfPitch': DurationOfPitch,
    'CityTier': CityTier,
    'PreferredPropertyStar': PreferredPropertyStar,
    'Passport': Passport,
    'PitchSatisfactionScore': PitchSatisfactionScore,
    'OwnCar': OwnCar,
    'NumberOfFollowups': NumberOfFollowups,
    'TypeofContact': TypeofContact,
    'Occupation': Occupation,
    'Gender': Gender,
    'MaritalStatus': MaritalStatus,
    'Designation': Designation,
    'ProductPitched': ProductPitched
}])

# Set the classification threshold
classification_threshold = 0.45

# Predict button
if st.button("Predict"):
    prediction_proba = model.predict_proba(input_data)[0, 1]
    prediction = (prediction_proba >= classification_threshold).astype(int)
    result = "buy" if prediction == 1 else "NOT buy"
    st.write(f"Based on the information provided, the customer is likely to {result}.")
