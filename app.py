import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Portuguese Bank Prediction")

st.title("🏦 Portuguese Bank Marketing Prediction")
st.write("Predict whether a customer will subscribe to a term deposit")

# Load dataset (IMPORTANT: delimiter=';')
df = pd.read_csv("bank-full.csv", delimiter=';')

# Show columns (debug – you can remove later)
st.write("Columns:", df.columns)

# Encode target column
df["y"] = df["y"].map({"yes": 1, "no": 0})

# Encode categorical features
le = LabelEncoder()
for col in df.select_dtypes(include="object").columns:
    if col != "y":
        df[col] = le.fit_transform(df[col])

X = df.drop("y", axis=1)
y = df["y"]

# Train model
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

st.subheader("Enter Customer Details")

age = st.slider("Age", 18, 95, 30)
balance = st.number_input("Account Balance", -5000, 100000, 1000)
housing = st.selectbox("Housing Loan", ["yes", "no"])
loan = st.selectbox("Personal Loan", ["yes", "no"])
campaign = st.slider("Campaign Contacts", 1, 50, 2)
previous = st.slider("Previous Contacts", 0, 50, 0)

housing = 1 if housing == "yes" else 0
loan = 1 if loan == "yes" else 0

input_data = pd.DataFrame(
    [[age, balance, housing, loan, campaign, previous]],
    columns=["age", "balance", "housing", "loan", "campaign", "previous"]
)

import streamlit as st
if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1] * 100

    if prediction == 1:
        st.success(f"✅ Customer WILL Subscribe ({prob:.2f}%)")
    else:
        st.error(f"❌ Customer will NOT Subscribe ({prob:.2f}%)")