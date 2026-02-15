import streamlit as st
import json
from pathlib import Path

import requests


API_URL = "http://127.0.0.1:8000/predict"


def main():
    st.set_page_config(page_title="Credit Risk Dashboard", layout="centered")
    st.title("Credit Risk Scoring")

    st.markdown("Enter applicant financial details and press **Predict**")

    with st.form(key="input_form"):
        col1, col2 = st.columns(2)
        with col1:
            person_age = st.number_input("Age", min_value=18, max_value=100, value=30)
            person_income = st.number_input("Annual Income", min_value=0, value=50000)
            person_home_ownership = st.selectbox("Home Ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"])
            person_emp_length = st.number_input("Employment Length (years)", min_value=0.0, value=3.0)
            cb_person_default_on_file = st.selectbox("Default on file", ["Y", "N"])
        with col2:
            loan_intent = st.selectbox("Loan Intent", ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION", "OTHER"]) 
            loan_grade = st.selectbox("Loan Grade", ["A","B","C","D","E","F","G"]) 
            loan_amnt = st.number_input("Loan Amount", min_value=0, value=10000)
            loan_int_rate = st.number_input("Loan Interest Rate (%)", min_value=0.0, value=10.0)
            loan_percent_income = st.number_input("Loan % of Income", min_value=0.0, value=0.2)
            cb_person_cred_hist_length = st.number_input("Credit History Length (years)", min_value=0.0, value=3.0)

        submit = st.form_submit_button("Predict")

    if submit:
        payload = {
            "person_age": person_age,
            "person_income": person_income,
            "person_home_ownership": person_home_ownership,
            "person_emp_length": person_emp_length,
            "loan_intent": loan_intent,
            "loan_grade": loan_grade,
            "loan_amnt": loan_amnt,
            "loan_int_rate": loan_int_rate,
            "loan_percent_income": loan_percent_income,
            "cb_person_default_on_file": cb_person_default_on_file,
            "cb_person_cred_hist_length": cb_person_cred_hist_length,
        }

        try:
            resp = requests.post(API_URL, json=payload, timeout=5)
            resp.raise_for_status()
            result = resp.json()
        except Exception as e:
            st.error(f"Prediction request failed: {e}")
            return

        score = result.get("risk_score")
        category = result.get("risk_category")
        probability = result.get("probability")

        if category is None and score is not None:
            mapping = {0: "Low Risk", 1: "Medium Risk", 2: "High Risk"}
            category = mapping.get(int(score), "Unknown")

        if category is not None:
            if "Low" in category:
                color = "#2ECC71"  # green
            elif "Medium" in category:
                color = "#F39C12"  # orange
            else:
                color = "#E74C3C"  # red
        else:
            color = "#95A5A6"

        st.markdown("---")
        st.markdown(f"### Risk: <span style='color:{color};'>{category}</span>", unsafe_allow_html=True)
        st.write(f"Risk score: {score}")
        if probability is not None:
            st.write(f"Probability: {probability:.2f}")

    st.sidebar.header("Quick Actions")
    if st.sidebar.button("Use sample applicant"):
        sample = {
            "person_age": 30,
            "person_income": 60000,
            "person_home_ownership": "RENT",
            "person_emp_length": 3.0,
            "loan_intent": "PERSONAL",
            "loan_grade": "C",
            "loan_amnt": 15000,
            "loan_int_rate": 12.5,
            "loan_percent_income": 0.25,
            "cb_person_default_on_file": "N",
            "cb_person_cred_hist_length": 4.0,
        }
        st.experimental_set_query_params(**sample)
        st.success("Sample loaded — adjust inputs and press Predict")


if __name__ == "__main__":
    main()
