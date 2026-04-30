import streamlit as st
import pandas as pd
import numpy as np
import joblib

#Load model
bundle = joblib.load("model1.pkl")

model = bundle["model"]
threshold = bundle["threshold"]
features = bundle["features"]
encoders = bundle["encoders"]

# ui design

col1, col2 = st.columns([1, 6])

with col1:
    st.image("https://img.icons8.com/color/96/combo-chart.png", width=60)

with col2:
    st.title("Customer Churn Prediction")
    st.caption("AI-powered insights for customer retention")


st.sidebar.header("Enter Customer Details")

age = st.sidebar.slider("Age", 18, 80, 30)
tenure = st.sidebar.slider("Tenure (Months)", 1, 60, 12)
data_usage = st.sidebar.number_input("Data Usage (MB)", 0, 20000, 5000)
revenue = st.sidebar.number_input("Total Revenue", 0.0, 10000.0, 100.0)
purchases = st.sidebar.number_input("Number of Purchases", 0, 50, 5)
satisfaction = st.sidebar.slider("Satisfaction (1-5)", 1, 5, 3)

gender = st.sidebar.selectbox("Gender", encoders["Gender"].classes_)
device = st.sidebar.selectbox("Device", encoders["MTN Device"].classes_)
state = st.sidebar.selectbox("State", encoders["State"].classes_)
plan = st.sidebar.selectbox("Subscription Plan", encoders["Subscription Plan"].classes_)

#Build input dataframe
input_data = pd.DataFrame({
    "Age": [age],
    "State": [state],
    "MTN Device": [device],
    "Gender": [gender],
    "Satisfaction Rate": [satisfaction],
    "Customer Tenure in months": [tenure],
    "Subscription Plan": [plan],
    "Unit Price": [15],  # adjust if needed
    "Number of Times Purchased": [purchases],
    "Data Usage": [data_usage],
    "Total Revenue": [revenue]
})

# Apply encoders
for col in encoders:
    input_data[col] = encoders[col].transform(input_data[col])

#fix column order
input_data = input_data[features]

#prediction
if st.button("🔍 Analyze Customer"):

    prob = model.predict_proba(input_data)[0][1]
    pred = int(prob > threshold)

    # Segmentation
    if prob > 0.7:
        segment = "High Risk"
    elif prob > 0.4:
        segment = "Medium Risk"
    else:
        segment = "Low Risk"

    #Results
    st.subheader("📌 Customer Risk Analysis")

    col1, col2, col3 = st.columns(3)

    col1.metric("Churn Probability", f"{prob:.2%}")
    col2.metric("Prediction", "Churn" if pred == 1 else "Not Churn")  
    col3.metric("Segment", segment)

    #Risk Level
    st.subheader("⚠ Risk Level")

    if prob > 0.7:
        st.error(f"High Risk Customer ({prob:.2%})")
    elif prob > 0.4:
        st.warning(f"Moderate Risk Customer ({prob:.2%})")
    else:
        st.success(f"Low Risk Customer ({prob:.2%})")

    #Recommended Action
    st.subheader("💡 Recommended Action")

    if segment == "High Risk":
        st.write("""
        - Offer discounts or incentives  
        - Improve customer support  
        - Engage customer immediately  
        """)

    elif segment == "Moderate Risk Customer":
        st.write("""
        - Send personalized offers  
        - Increase engagement campaigns  
        """)

    else:
        st.write("""
        - Maintain relationship  
        - Upsell premium services  
        """)

    # reason for prediction
    st.subheader("📊 Why this prediction?")

    st.write("""
    The model identified these as the most important factors influencing churn:
    
    - Satisfaction Rate  
    - Age  
    - Pricing (Unit Price)  
    - Customer Tenure  
    
    Customers with low satisfaction, shorter tenure, and higher pricing are more likely to churn.
    """)

    import pandas as pd

    feature_imp = pd.Series(model.feature_importances_, index=features)
    top_features = feature_imp.sort_values(ascending=True).tail(5)

    st.subheader("📊 Top Factors Driving Churn")

    st.bar_chart(top_features)

    st.markdown("---")
    st.caption("Model: XGBoost | Threshold: 0.3 | Built for churn risk analysis")
