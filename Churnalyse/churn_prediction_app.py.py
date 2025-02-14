import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Title of the app
st.title("📊 Customer Churn Prediction Dashboard")

# Upload dataset
uploaded_file = st.file_uploader("📂 Upload CSV File", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### 📝 Dataset Preview", df.head())

    # Display dataset info
    if st.checkbox("📌 Show Dataset Info"):
        buffer = df.info(verbose=True, buf=None)
        st.text(buffer)

    # Check for missing values
    st.subheader("🔍 Missing Values in Dataset")
    st.write(df.isnull().sum())

    # Handling missing values
    df.fillna(df.median(numeric_only=True), inplace=True)  # Fill numeric missing values
    categorical_cols = ["Gender", "Subscription Type", "Contract Length"]
    for col in categorical_cols:
        df[col].fillna(df[col].mode()[0], inplace=True)  # Fill categorical missing values

    # Feature Engineering (Creating new columns before feature selection)
    df["InteractionScore"] = df["Usage Frequency"] + df["Support Calls"]
    df["SpendPerMonth"] = df["Total Spend"] / (df["Tenure"] + 1)  # Avoid division by zero

    # Exploratory Data Analysis (EDA) Visualization
    if st.checkbox("📊 Show EDA Visualizations"):
        st.subheader("📌 Age Distribution")
        fig, ax = plt.subplots()
        sns.histplot(df["Age"], kde=True, ax=ax)
        st.pyplot(fig)

        st.subheader("📌 Churn Distribution")
        fig, ax = plt.subplots()
        churn_counts = df["Churn"].value_counts()
        ax.pie(churn_counts, labels=churn_counts.index, autopct="%1.1f%%", startangle=140)
        ax.axis("equal")
        st.pyplot(fig)

    # Preprocessing
    st.subheader("🚀 Model Training")

    # Encode categorical variables
    encoder = LabelEncoder()
    for col in categorical_cols:
        if df[col].dtype == "object":  # Ensure only string columns are encoded
            df[col] = encoder.fit_transform(df[col])

    # Select features after feature engineering
    feature_columns = ["Age", "Gender", "Tenure", "Usage Frequency", "Support Calls", 
                       "Payment Delay", "Subscription Type", "Contract Length", "Total Spend", 
                       "InteractionScore", "SpendPerMonth"]
    X = df[feature_columns]
    y = df["Churn"]

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Scaling numerical features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Model Selection
    model_option = st.selectbox("🛠 Choose a model for prediction", ["Random Forest", "XGBoost"])
    model = RandomForestClassifier(random_state=42) if model_option == "Random Forest" else XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)

    # Train the model
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Model Evaluation
    st.write("### ✅ Model Evaluation")
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"🎯 Accuracy: **{accuracy:.2f}**")

    st.subheader("📜 Classification Report")
    st.text(classification_report(y_test, y_pred))

    # Prediction
    st.subheader("🔮 Make Predictions")

    # Collect inputs for prediction
    age = st.number_input("🧑 Age", min_value=18, max_value=100, value=30)
    usage_frequency = st.number_input("📱 Usage Frequency", min_value=0, max_value=100, value=10)
    support_calls = st.number_input("☎️ Support Calls", min_value=0, max_value=50, value=5)
    payment_delay = st.number_input("⏳ Payment Delay (days)", min_value=0, max_value=365, value=10)
    total_spend = st.number_input("💵 Total Spend", min_value=0, max_value=100000, value=5000)
    tenure = st.number_input("📆 Tenure (months)", min_value=0, max_value=240, value=12)
    contract_length = st.selectbox("📜 Contract Length", ["Monthly", "Annual", "Bi-Annual"])

    gender = st.radio("⚤ Gender", ["Male", "Female"])
    subscription_type = st.selectbox("💳 Subscription Type", ["Basic", "Premium"])

    # Encode categorical inputs
    gender_encoded = 1 if gender == "Male" else 0
    subscription_encoded = 1 if subscription_type == "Premium" else 0
    contract_encoded = 0 if contract_length == "Monthly" else (1 if contract_length == "Bi-Annual" else 2)

    # Feature Engineering for prediction input
    interaction_score = usage_frequency + support_calls
    spend_per_month = total_spend / (tenure + 1)  # Avoid division by zero

    # Create prediction input
    prediction_input = np.array([
        age, gender_encoded, tenure, usage_frequency, support_calls, payment_delay, 
        subscription_encoded, contract_encoded, total_spend, interaction_score, spend_per_month
    ]).reshape(1, -1)

    # Scale the input
    prediction_input_scaled = scaler.transform(prediction_input)

    if st.button("🔮 Predict Churn"):
        prediction = model.predict(prediction_input_scaled)
        st.success("✅ Customer is likely to **stay**." if prediction == 0 else "⚠️ Customer is likely to **churn**.")
