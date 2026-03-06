import streamlit as st
import pandas as pd
import numpy as np
import joblib
from scipy.stats import ks_2samp

model = joblib.load("best_credit_risk_model.pkl")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("columns.pkl")

st.set_page_config(page_title="Credit Risk Prediction System", layout="wide")

st.title("Credit Risk Prediction System")

st.write("Enter applicant details to predict loan default risk.")

age = st.number_input("Age", 22, 97, 26)
income = st.number_input("Annual Income (₹)", 4000, 6000000, 55000)
loan_amount = st.number_input("Loan Amount (₹)", 500, 35000, 8000)
interest_rate = st.slider("Interest Rate (%)", 5.42, 23.22, 10.99)
credit_history = st.slider("Credit History Length (years)", 2, 30, 4)
emp_length = st.slider("Employment Length (years)", 0, 40, 5)
person_home_ownership = st.selectbox("Home Ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"])
loan_intent = st.selectbox("Loan Intent", ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE",
                                                   "DEBTCONSOLIDATION", "HOMEIMPROVEMENT"])
loan_grade = st.selectbox("Loan Grade", ["A","B","C","D","E","F","G"])
past_defaults = st.selectbox("Previously Defaulted?", ["Y", "N"])

dti = loan_amount/(income)

input_data = pd.DataFrame([[
    age,
    income,
    emp_length, 
    loan_amount,
    interest_rate,
    credit_history,
    past_defaults,
    loan_intent,
    loan_grade,
    person_home_ownership,
    dti
]], columns=[
    'person_age',
    'person_income',
    'person_emp_length',
    'loan_amnt',
    'loan_int_rate',
    'cb_person_cred_hist_length',
    'cb_person_default_on_file',
    'loan_intent',
    'loan_grade',
    'person_home_ownership',
    'Debt_To_Income_Ratio'
])

for col in columns:
    if col not in input_data.columns:
        input_data[col] = 0

input_data = input_data[columns]

scale_cols = [
    'person_age',
    'person_income',
    'person_emp_length',
    'loan_amnt',
    'loan_int_rate',
    'cb_person_cred_hist_length',
    'Debt_To_Income_Ratio'
]

input_data[scale_cols] = scaler.transform(input_data[scale_cols])

drift_features = [
    'person_income',
    'Debt_To_Income_Ratio',
    'loan_int_rate',
    'loan_amnt',
    'cb_person_cred_hist_length'
]


baseline_df = pd.read_csv("credit_risk_cleaned.csv")
baseline_features = baseline_df[drift_features]
current_df = input_data.copy()
baseline = baseline_features
current = input_data[drift_features]


def detect_behavioral_drift(baseline, current, threshold=0.1):
    drift_results = {}

    for col in baseline.columns:
        ks_stat, p_value = ks_2samp(baseline[col], current[col])
        drift_results[col] = {
            "ks_stat": ks_stat,
            "p_value": p_value,
            "drift": p_value < threshold
        }

    return drift_results
    
st.subheader("Behavioral Drift Detection")
drift_detected = False
drift_results = detect_behavioral_drift(baseline, current)

for feature, result in drift_results.items():
    if result["drift"]:
       drift_detected = True
       st.warning("Drift detected")
if not drift_detected:
    st.success("No significant behavioral drift detected")

if st.button("Predict Loan Risk"):
    # DTI-based override logic
    prob = model.predict_proba(input_data)[0][1]

    if dti < 0.3:
       prob -= 0.1
    elif dti < 0.6:
       prob += 0.05
    else:
      prob += 0.2

      prob = min(max(prob, 0), 1)

    st.subheader("Prediction Result")

    if prob < 0.3:
        st.success(f"Low Risk (Default Probability: {prob:.2f}) → Approve Loan")
    elif prob < 0.6:
        st.warning(f"Medium Risk (Default Probability: {prob:.2f}) → Review Required")
    else:
        st.error(f"High Risk (Default Probability: {prob:.2f}) → Reject Loan")