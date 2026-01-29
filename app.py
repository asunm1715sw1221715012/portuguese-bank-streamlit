import streamlit as st
import pandas as pd
import pickle

model = pickle.load(open("model.pkl", "rb"))

st.subheader("Enter Customer Details")

age = st.slider("Age", 18, 95, 30)
balance = st.number_input("Account Balance", -5000, 100000, 1000)
housing = st.selectbox("Housing Loan", ["yes", "no"])
loan = st.selectbox("Personal Loan", ["yes", "no"])
campaign = st.slider("Campaign Contacts", 1, 50, 2)
previous = st.slider("Previous Contacts", 0, 50, 0)

if st.button("Predict"):
    input_df = pd.DataFrame([{
        "age": age,
        "balance": balance,
        "housing": housing,
        "loan": loan,
        "campaign": campaign,
        "previous": previous
    }])

    prediction = model.predict(input_df)[0]

    if prediction == 1:
        st.success("Customer WILL subscribe to term deposit")
    else:
        st.error("Customer will NOT subscribe")
