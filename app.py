import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import plotly.graph_objects as go
from datetime import datetime, timedelta


st.set_page_config(
    page_title="AI Credit Risk Platform",
    layout="wide",
    page_icon="💳",
)

st.markdown("""
# 💳 AI-Powered Credit Risk Intelligence
### Real-time personal finance risk scoring platform
""")


def find_model_path():
    base = Path(__file__).resolve().parent
    p1 = base / "model" / "rf_pipeline.joblib"
    p2 = base / "model" / "risk_model.pkl"
    if p1.exists():
        return p1
    if p2.exists():
        return p2
    return None


def risk_color(category: str) -> str:
    cat = category.lower()
    if "low" in cat:
        return "#00C853"  # green
    if "medium" in cat:
        return "#FFAB00"  # orange
    return "#D50000"  # red


def simulate_prediction():
    # Simulate a cred score, category and confidence
    credit_score = int(np.clip(np.random.normal(700, 80), 300, 900))
    if credit_score >= 750:
        risk = "Low Risk"
    elif credit_score >= 600:
        risk = "Medium Risk"
    else:
        risk = "High Risk"
    confidence = round(float(np.clip(np.random.beta(5, 2), 0.5, 0.99)), 2)

    # Simulated explanation (feature impacts)
    features = [
        {"feature": "payment_history", "impact": np.random.uniform(0.1, 0.6)},
        {"feature": "credit_utilization", "impact": -np.random.uniform(0.1, 0.5)},
        {"feature": "income", "impact": np.random.uniform(0.0, 0.4)},
    ]

    return credit_score, risk, confidence, features


# Try loading a model; fall back to simulation
MODEL = None
model_path = find_model_path()
if model_path is not None:
    try:
        MODEL = joblib.load(model_path)
    except Exception:
        MODEL = None


page = st.sidebar.radio("Navigation", ["Risk Scoring", "Analytics", "About"])


if page == "Risk Scoring":
    st.title("📊 Personal Finance Risk Assessment")

    # Simple input controls to re-run or demo
    with st.sidebar.form(key="demo_form"):
        st.subheader("Input (demo)")
        sample_run = st.form_submit_button("Run Scoring")

    if MODEL is not None and sample_run:
        # If a real model is available, attempt to produce a prediction using default/simulated input
        try:
            # very generic: try to create a feature vector from defaults, model-specific code may be needed
            defaults = {}
            df = pd.DataFrame([defaults])
            pred = MODEL.predict(df)
            conf = 0.8
            credit_score = int(np.clip(600 + conf * 300, 300, 900))
            risk_category = "Low Risk" if pred[0] == 0 else ("Medium Risk" if pred[0] == 1 else "High Risk")
            confidence = round(conf, 2)
            explanation = [{"feature": "model_feature_x", "impact": 0.4}]
        except Exception:
            credit_score, risk_category, confidence, explanation = simulate_prediction()
    else:
        credit_score, risk_category, confidence, explanation = simulate_prediction()

    col1, col2, col3 = st.columns(3)
    col1.metric("Credit Score", credit_score)
    col2.metric("Risk Category", risk_category)
    col3.metric("Confidence", f"{int(confidence*100)}%")

    st.markdown(
        f"<h2 style='color:{risk_color(risk_category)}'>{risk_category}</h2>",
        unsafe_allow_html=True,
    )

    # Gauge chart
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=credit_score,
        title={'text': "Credit Score"},
        gauge={
            'axis': {'range': [300, 900]},
            'bar': {'color': "#1f77b4"},
            'steps': [
                {'range': [300, 599], 'color': "#D50000"},
                {'range': [600, 749], 'color': "#FFAB00"},
                {'range': [750, 900], 'color': "#00C853"},
            ],
        }
    ))

    st.plotly_chart(fig, use_container_width=True)

    # SHAP-style feature importances
    st.subheader("🔍 Key Risk Factors")
    expl_df = pd.DataFrame(explanation).set_index("feature")
    st.bar_chart(expl_df)


elif page == "Analytics":
    st.title("📈 Portfolio Analytics")

    # Simulate historical scores for the last 30 days
    days = 30
    dates = [datetime.now() - timedelta(days=i) for i in range(days)][::-1]
    scores = [int(np.clip(np.random.normal(700, 80), 300, 900)) for _ in range(days)]
    df_hist = pd.DataFrame({"date": dates, "score": scores}).set_index("date")

    st.subheader("Credit Score Over Time")
    st.line_chart(df_hist)

    # Risk distribution
    bins = ["Low Risk", "Medium Risk", "High Risk"]
    def bucket(s):
        if s >= 750:
            return "Low Risk"
        if s >= 600:
            return "Medium Risk"
        return "High Risk"

    dist = pd.Series([bucket(s) for s in scores]).value_counts().reindex(bins, fill_value=0)
    st.subheader("Risk Category Distribution")
    st.bar_chart(dist)

    st.subheader("Average Confidence")
    avg_conf = round(np.random.uniform(0.7, 0.95), 2)
    st.metric("Avg Confidence", f"{int(avg_conf*100)}%")


else:
    st.title("About")
    st.write(
        "This demo dashboard shows an AI-powered credit risk scoring UX. "
        "If you have a trained model in `model/rf_pipeline.joblib`, the app will attempt to load it and switch from simulated values to model outputs." 
    )
    st.write("To run locally: `pip install streamlit plotly` then `streamlit run app.py` from the project root.")
