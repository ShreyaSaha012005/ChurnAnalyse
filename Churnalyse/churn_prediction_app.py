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

# Streamlit App Title
st.title('📊 Customer Churn Prediction Dashboard')

# File uploader
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    # Read dataset
    df = pd.read_csv(uploaded_file)
    st.write("### Dataset Preview", df.head())

    # Data Preprocessing
    st.subheader("Data Preprocessing")
    
    # Handling Categorical Variables
    cat_cols = ['Geography', 'Gender']
    encoder = LabelEncoder()
    for col in cat_cols:
        df[col] = encoder.fit_transform(df[col])
    
    # Feature Engineering
    df['CreditUtilization'] = df['Balance'] / (df['CreditScore'] + 1)  # Avoid division by zero
    df['BalanceToSalaryRatio'] = df['Balance'] / (df['EstimatedSalary'] + 1)
    
    # Splitting Data
    X = df.drop(['Exited', 'RowNumber', 'CustomerId', 'Surname'], axis=1, errors='ignore')
    y = df['Exited']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Feature Scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Select Model
    model_option = st.selectbox("Select a Model", ['Random Forest', 'XGBoost'])

    # Train the Model
    if st.button("Train Model"):
        if model_option == 'Random Forest':
            model = RandomForestClassifier(random_state=42)
        else:
            model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Show Model Evaluation
        st.subheader("Model Performance")
        st.write(f"**Accuracy:** {accuracy_score(y_test, y_pred):.2f}")
        st.text("Classification Report:")
        st.text(classification_report(y_test, y_pred))

        # Store model and scaler for later use
        st.session_state['model'] = model
        st.session_state['scaler'] = scaler

    # Prediction Section
    st.subheader("🔮 Make Predictions")
    if 'model' in st.session_state:
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
        credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=600)
        balance = st.number_input("Balance", min_value=0, max_value=1000000, value=50000)
        estimated_salary = st.number_input("Estimated Salary", min_value=0, max_value=200000, value=50000)

        # Create feature array for prediction
        pred_input = np.array([age, credit_score, balance, estimated_salary]).reshape(1, -1)
        pred_input_scaled = st.session_state['scaler'].transform(pred_input)

        if st.button("Predict Churn"):
            prediction = st.session_state['model'].predict(pred_input_scaled)
            if prediction == 0:
                st.success("✅ This customer is likely to stay.")
            else:
                st.warning("⚠️ This customer is likely to churn.")
    else:
        st.warning("Please train the model first before making predictions.")
