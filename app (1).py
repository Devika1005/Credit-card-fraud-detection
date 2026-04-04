import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib

# Load model    
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Fraud Detection", layout="wide")

st.title("💳 Credit Card Fraud Detection")

# Sidebar
option = st.sidebar.radio("Select Input Method", ["Manual Input", "CSV Upload"])

# ================= MANUAL INPUT =================
if option == "Manual Input":
    st.subheader("Enter Transaction Details")

    # Input fields
    time = st.number_input("Time", value=0.0)
    amount = st.number_input("Amount", value=0.0)

    values = []

    for i in range(1, 29):
        val = st.number_input(f"V{i}", value=0.0)
        values.append(val)

    if st.button("Predict"):
        # Create dataframe
        input_data = pd.DataFrame(
            [[time] + values + [amount]],
            columns=[
                'Time','V1','V2','V3','V4','V5','V6','V7','V8','V9','V10',
                'V11','V12','V13','V14','V15','V16','V17','V18','V19','V20',
                'V21','V22','V23','V24','V25','V26','V27','V28','Amount'
            ]
        )

        # Scale Time & Amount
        input_data[['Time', 'Amount']] = scaler.transform(
            input_data[['Time', 'Amount']]
        )

        prediction = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0][1]

        if prediction == 1:
            st.error(f"🚨 Fraud Detected! Probability: {prob:.2f}")
        else:
            st.success(f"✅ Legitimate Transaction (Fraud Prob: {prob:.2f})")

# ================= CSV UPLOAD =================
else:
    st.subheader("creditcard.csv")

    file = st.file_uploader("Upload your dataset", type=["csv"])

    if file is not None:
        data = pd.read_csv(file)

        # Scale Time & Amount
        data[['Time', 'Amount']] = scaler.transform(
            data[['Time', 'Amount']]
        )

        predictions = model.predict(data)
        data["Prediction"] = predictions

        st.write(data.head())

        fraud_count = sum(predictions)
        normal_count = len(predictions) - fraud_count

        st.write(f"🚨 Fraud Transactions: {fraud_count}")
        st.write(f"✅ Normal Transactions: {normal_count}")

        # Chart
        st.bar_chart(data["Prediction"].value_counts())