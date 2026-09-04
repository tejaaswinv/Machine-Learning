from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st

ARTIFACTS = Path("artifacts")
FEATURES = [
    "annual_income", "monthly_debt", "credit_utilization",
    "missed_payments_12m", "credit_history_years", "employment_years",
    "existing_loans", "savings_balance", "age"
]

st.set_page_config(page_title="Credit Scoring Model", page_icon="💳")
st.title("💳 Credit Scoring Model")
st.caption("CodeAlpha Machine Learning Task 1")
st.warning("Educational demo only. Do not use this model to make real lending decisions.")

model_path = ARTIFACTS / "best_credit_model.joblib"
schema_path = ARTIFACTS / "feature_schema.json"
if not model_path.exists() or not schema_path.exists():
    st.info("Train the model first: `python train.py`")
    st.stop()

model = joblib.load(model_path)
schema = json.loads(schema_path.read_text())

values = {}
for feature in FEATURES:
    meta = schema[feature]
    label = feature.replace("_", " ").title()
    if feature in {"missed_payments_12m", "existing_loans", "age"}:
        values[feature] = st.number_input(
            label, min_value=int(meta["min"]), max_value=int(meta["max"]),
            value=int(meta["default"]), step=1
        )
    else:
        values[feature] = st.number_input(
            label, min_value=float(meta["min"]), max_value=float(meta["max"]),
            value=float(meta["default"])
        )

if st.button("Predict creditworthiness", type="primary"):
    X = pd.DataFrame([[values[f] for f in FEATURES]], columns=FEATURES)
    prob = float(model.predict_proba(X)[0, 1])
    pred = int(prob >= 0.5)
    st.subheader("Creditworthy ✅" if pred else "Higher credit risk ⚠️")
    st.metric("Model probability of creditworthiness", f"{prob:.1%}")
