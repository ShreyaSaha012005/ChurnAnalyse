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
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline as make_pipeline_imb

# Title of the app
st.title('Customer Churn Prediction Dashboard')

# File uploader to upload dataset
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
if uploaded_file is not None:
    # Read the dataset
    df = pd.read_csv(uploaded_file)
    st.write("Dataset Preview", df.head())

    # Display dataset info
    if st.checkbox("Show Dataset Info"):
        st.write(df.info())

    # Exploratory Data Analysis (EDA) Visualization
    if st.checkbox("Show EDA Visualizations"):
        st.subheader("Exploring Customer Demographics")
        
        # Age Distribution
        st.subheader("Age Distribution")
        fig, ax = plt.subplots()
        sns.histplot(df['Age'], kde=True, ax=ax)
        st.pyplot(fig)

        # Churn Distribution
        st.subheader("Churn Distribution")
        fig, ax = plt.subplots()
        output_counts = df['Exited'].value_counts()
        ax.pie(output_counts, labels=output_counts.index, autopct='%1.1f%%', startangle=140)
        ax.axis('equal')
        st.pyplot(fig)

    # Feature Engineering
    if st.checkbox("Perform Feature Engineering"):
        df['CreditUtilization'] = df['Balance'] / df['CreditScore']
        df['InteractionScore'] = df['NumOfProducts'] + df['HasCrCard'] + df['IsActiveMember']
        df['BalanceToSalaryRatio'] = df['Balance'] / df['EstimatedSalary']
        df['CreditScoreAgeInteraction'] = df['CreditScore'] * df['Age']
        st.write("Feature Engineering Complete", df.head())

    # Preprocessing and Model Training
    st.subheader("Model Training")

    cat_col = ['Geography', 'Gender', 'CreditScoreGroup']
    encoder = LabelEncoder()
    for column in cat_col:
        df[column] = encoder.fit_transform(df[column])

    X = df.drop(['Exited', 'RowNumber', 'CustomerId', 'Surname'], axis=1)
    y = df['Exited']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Scaling numerical features
    scaling_columns = ['Age', 'CreditScore', 'Balance', 'EstimatedSalary', 'CreditUtilization', 'BalanceToSalaryRatio', 'CreditScoreAgeInteraction']
    scaler = StandardScaler()
    X_train[scaling_columns] = scaler.fit_transform(X_train[scaling_columns])
    X_test[scaling_columns] = scaler.transform(X_test[scaling_columns])

    # Model options
    model_option = st.selectbox("Choose a model for prediction", ['Random Forest', 'XGBoost'])

    if model_option == 'Random Forest':
        model = RandomForestClassifier(random_state=42)
    elif model_option == 'XGBoost':
        model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    st.write("Model Evaluation")
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"Accuracy: {accuracy}")
    
    st.subheader("Classification Report")
    st.text(classification_report(y_test, y_pred))

    # Prediction Functionality
    st.subheader("Make Predictions")

    # Collect inputs for prediction
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=600)
    balance = st.number_input("Balance", min_value=0, max_value=1000000, value=50000)
    estimated_salary = st.number_input("Estimated Salary", min_value=0, max_value=200000, value=50000)

    # Create a prediction feature vector
    prediction_input = np.array([age, credit_score, balance, estimated_salary]).reshape(1, -1)
    prediction_input_scaled = scaler.transform(prediction_input)

    if st.button("Predict Churn"):
        prediction = model.predict(prediction_input_scaled)
        if prediction == 0:
            st.success("This customer is likely to stay.")
        else:
            st.warning("This customer is likely to churn.")

