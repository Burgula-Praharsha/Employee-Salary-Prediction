import streamlit as st
import pandas as pd
import joblib

# Load the model
model = joblib.load("best_model.pkl")

st.set_page_config(page_title="Employee Salary Classifier", page_icon="💼", layout="centered")
st.title("💼 Employee Salary Classification App")
st.markdown("Predict whether an employee earns **>50K** or **≤50K** based on input features.")

# Sidebar Inputs
st.sidebar.header("Enter Employee Details")

age = st.sidebar.slider("Age", 18, 65, 30)
educational_num = st.sidebar.slider("Education Level (numeric)", 1, 16, 10)
occupation = st.sidebar.selectbox("Occupation", [
    "Tech-support", "Craft-repair", "Other-service", "Sales", "Exec-managerial",
    "Prof-specialty", "Handlers-cleaners", "Machine-op-inspct", "Adm-clerical",
    "Farming-fishing", "Transport-moving", "Priv-house-serv", "Protective-serv",
    "Armed-Forces"
])
hours_per_week = st.sidebar.slider("Hours per Week", 1, 99, 40)
experience = st.sidebar.slider("Years of Experience", 0, 50, 5)

# Create input DataFrame with ALL required features
input_df = pd.DataFrame({
    'age': [age],
    'educational-num': [educational_num],
    'occupation': [occupation],
    'hours-per-week': [hours_per_week],
    'experience': [experience],
    'capital-gain': [0],     # Default dummy values
    'capital-loss': [0],
    'fnlwgt': [100000]       # Arbitrary default; adjust if needed
})

st.write("### 🔍 Input Data Preview")
st.write(input_df)

# Predict
if st.button("Predict Salary Class"):
    prediction = model.predict(input_df)
    st.success(f"✅ Prediction: Employee earns **{prediction[0]}**")

# Batch prediction
st.markdown("---")
st.markdown("### 📂 Batch Prediction")
uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

if uploaded_file is not None:
    batch_data = pd.read_csv(uploaded_file)

    # Fill missing required columns if necessary
    required_cols = ['age', 'educational-num', 'occupation', 'hours-per-week',
                     'experience', 'capital-gain', 'capital-loss', 'fnlwgt']
    for col in required_cols:
        if col not in batch_data.columns:
            if col in ['capital-gain', 'capital-loss']:
                batch_data[col] = 0
            elif col == 'fnlwgt':
                batch_data[col] = 100000
            else:
                st.error(f"Missing required column: {col}")
                st.stop()

    st.write("📄 Uploaded Data Preview:")
    st.write(batch_data.head())

    predictions = model.predict(batch_data)
    batch_data['Predicted_Salary_Class'] = predictions
    st.write("✅ Predicted Results:")
    st.write(batch_data.head())

    csv = batch_data.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Prediction Results", csv, "predicted_salary.csv", "text/csv")
