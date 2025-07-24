# Employee Salary Prediction

This project aims to predict whether an individual earns more or less than 50K annually based on their demographic and employment features using multiple machine learning algorithms.

## Overview

The dataset Employee_Salary_Data is used for this project which is having huge data in it. Various preprocessing techniques and machine learning models have been applied to classify employee salaries effectively.

## Features Used
- Age  
- Work Class  
- Education  
- Occupation  
- Hours per Week  
- Marital Status
- Gender
- etc.

## Technologies & Libraries
- **Python** - Language
- **Pandas** – Data handling  
- **NumPy** – Numerical operations  
- **Matplotlib** – Visualization  
- **Scikit-learn** – ML model building  
- **Streamlit** – Web App interface  
- **pyngrok** – Public URL tunneling  
- **Joblib** – Model saving/loading  

## 📊 Models Applied
- Logistic Regression  
- Random Forest  
- K-Nearest Neighbors  
- Support Vector Machine  
- Gradient Boosting

## 🎯 Best Model
The model with the highest accuracy was selected and saved using `joblib` as `best_model.pkl`.

## 🌐 Web App
A simple interactive UI is created using Streamlit and hosted with pyngrok.

## 💡 What I Learned
- Data preprocessing and label encoding  
- Applying and comparing multiple ML models  
- Evaluating models using accuracy, precision, recall, and F1-score  
- Creating an interactive app using Streamlit  
