# %%
import pandas as pd

# %%
data = pd.read_csv(r'C:\Users\burgu\OneDrive\Desktop\Employee_Salary_Data.csv')
data

# %%
data.shape

# %%
data.head(6)

# %%
data.tail(7)

# %%
data.describe()

# %%
data.info()

# %%
# Null Values
data.isna()

# %%
data.isna().sum()

# %%
# Remove underrepresented occupation category
print(data.occupation.value_counts())

# %%
data.occupation.replace({'?': 'Others'}, inplace=True)
print(data['occupation'].value_counts())

# %%
print(data.gender.value_counts())

# %%
print(data.age.value_counts())

# %%
print(data.fnlwgt.value_counts())

# %%
print(data.workclass.value_counts())

# %%
data.workclass.replace({'?': 'NotListed'}, inplace=True)
print(data.workclass.value_counts())

# %%
data = data[data['workclass'] != 'Without-pay']
data = data[data['workclass'] != 'Never-worked']
print(data['workclass'].value_counts())

# %%
print(data.income.value_counts())

# %%
print(data['native-country'].value_counts())

# %%
print(data['hours-per-week'].value_counts())

# %%
print(data['capital-loss'].value_counts())

# %%
print(data['capital-gain'].value_counts())

# %%
print(data.race.value_counts())

# %%
print(data.relationship.value_counts())

# %%
print(data.education.value_counts())

# %%
data = data[data['education'] != 'Preschool']
data = data[data['education'] != '1st-4th']
data = data[data['education'] != '5th-6th']
print(data['education'].value_counts())

# %%
data.shape

# %%
import matplotlib.pyplot as plt
plt.boxplot(data['age'])
plt.show()

# %%
data = data[(data['age'] <= 75) & (data['age'] >= 17)]
plt.boxplot(data['age'])
plt.show()

# %%
data.shape

# %%
plt.boxplot(data['capital-gain'])
plt.show()

# %%
plt.boxplot(data['educational-num'])
plt.show()

# %%
data = data[(data['educational-num'] <= 16) & (data['educational-num'] >= 5)]
plt.boxplot(data['educational-num'])
plt.show()

# %%
plt.boxplot(data['hours-per-week'])
plt.show()

# %%
data.shape

# %%
data = data.drop(columns=['education'])

# %%
data

# %%
print(data.columns)

# %%
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
data['workclass'] = encoder.fit_transform(data['workclass'])
data['marital-status'] = encoder.fit_transform(data['marital-status'])
data['occupation'] = encoder.fit_transform(data['occupation'])
data['relationship'] = encoder.fit_transform(data['relationship'])
data['race'] = encoder.fit_transform(data['race'])
data['gender'] = encoder.fit_transform(data['gender'])
data['native-country'] = encoder.fit_transform(data['native-country'])

# %%
data

# %%
x = data.drop(columns=['income'])
y = data['income']

# %%
print(x)

# %%
print(y)

# %%
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X_train.select_dtypes(include=['object']).columns

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])

models = {
    "LogisticRegression": LogisticRegression(),
    "RandomForest": RandomForestClassifier(),
    "KNN": KNeighborsClassifier(),
    "SVM": SVC(),
    "GradientBoosting": GradientBoostingClassifier()
}

results = {}

for name, model in models.items():
    pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc
    print(f"\n{name} Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))

# %%
plt.bar(results.keys(), results.values(), color='purple')
plt.ylabel('Accuracy Score')
plt.title('Model Comparison')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()

# %%
import joblib

results = {}
best_pipeline = None
best_model_name = None
best_score = 0

for name, model in models.items():
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
    acc = accuracy_score(y_test, preds)
    results[name] = acc
    print(f"{name}: {acc:.4f}")
    
    if acc > best_score:
        best_pipeline = pipeline
        best_model_name = name
        best_score = acc

print(f"\n Best model: {best_model_name} with accuracy {best_score:.4f}")
joblib.dump(best_pipeline, "best_model.pkl")
print("Saved best model as best_model.pkl")


# %%
%%writefile app.py
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


# %%
!pip install streamlit pyngrok

# %%
!ngrok authtoken 1a2b3c4d5e6f7g8h9i0jKLMNOPqrstuVWXYZ1234gP

# %%
import os
import threading

def run_streamlit():
     os.system('streamlit run app.py --server.port 8503')

thread=threading.Thread(target=run_streamlit)
thread.start()


# %%
from pyngrok import ngrok
import time

#wait a few seconds to run streamlit port 8503
time.sleep(5)

#Create a tunnel to the streamlit port 8503
public_url=ngrok.connect(8503)
print("Your streamlit app is live here:", public_url)



