📊 Customer Churn Prediction & Segmentation

An end-to-end machine learning project that predicts customer churn and segments customers into actionable risk groups using real-world telecom-style data.

🚀# Project Overview

Customer churn is a major problem for telecom companies. This project builds a machine learning model to:

Predict whether a customer will churn
Estimate churn probability
Segment customers into Low, Medium, and High Risk
Provide business recommendations for retention
🎯 Business Goal

The goal of this project is to help businesses:

Identify customers likely to churn
Take proactive retention actions
Improve customer satisfaction
Increase customer lifetime value
🧠 Machine Learning Approach

This project uses multiple classification models:

Logistic Regression
Decision Tree
Random Forest
K-Nearest Neighbors
✅ XGBoost (Final Model)
Why XGBoost?
Best performance on dataset
Handles imbalance well
Provides feature importance
⚙️ Features Used
Age
State
MTN Device
Gender
Satisfaction Rate
Customer Tenure
Subscription Plan
Unit Price
Data Usage
Number of Purchases
Total Revenue
🔧 Data Processing
Label Encoding for categorical variables
Train-test split (80/20)
Standard scaling (for some models)
Threshold tuning for better recall
📈 Model Performance (XGBoost)
Accuracy: ~76%
Balanced performance on churn class
Improved recall using threshold tuning
📊 Feature Importance

Top factors influencing churn:
Satisfaction Rate
Age
Pricing (Unit Price)
Customer Tenure
Data Usage

👉 These insights help businesses understand why customers churn

🧩 Customer Segmentation

Customers are segmented based on churn probability:

🔴 High Risk (> 0.7)
🟡 Medium Risk (0.4 – 0.7)
🟢 Low Risk (< 0.4)
💡 Business Recommendations
High Risk: Discounts, support, urgent engagement
Medium Risk: Personalized offers, engagement campaigns
Low Risk: Upsell and maintain relationship
🖥️ Streamlit App

An interactive web app was built using Streamlit to:

Input customer details
Predict churn probability
Display risk segment
Show feature importance
Provide recommendations
🛠️ Tech Stack
Python
Pandas
NumPy
Scikit-learn
XGBoost
Joblib
Streamlit
Matplotlib / Seaborn
📦 Model Deployment

Model and preprocessing components are saved using:

joblib.dump(bundle, "model.pkl")

The bundle includes:

Trained model
Feature names
Threshold
Encoders

#▶️ How to Run the App
Clone the repository:
git clone https://github.com/your-username/churn-project.git
cd churn-project
Install dependencies:
pip install -r requirements.txt
Run the app:
streamlit run app.py
📌 Project Highlights
End-to-end ML pipeline
Business-focused insights
Threshold optimization
Customer segmentation
Interactive dashboard

👤 Author

Lilian Nyinyayo
Total Revenue
