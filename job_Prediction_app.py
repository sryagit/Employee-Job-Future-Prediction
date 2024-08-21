# streamlit_app.py

import streamlit as st
import pickle
import numpy as np

# Load the trained Naive Bayes model
with open('naive_bayes_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Title of the app
st.title('Employee Job Future Prediction')

# Collect user inputs for the model features with specified ranges and steps
satisfaction = st.slider('Satisfaction Level', 0.0, 1.0, 0.5, step=0.01)
evaluation = st.slider('Last Evaluation', 0.0, 1.0, 0.5, step=0.01)
number_project = st.number_input('Number of Projects', min_value=1, max_value=10, value=3, step=1)
monthly_hours = st.number_input('Monthly Hours', min_value=50, max_value=350, value=160, step=1)
tenure = st.number_input('Work Experiance (Year)', min_value=1, max_value=10, value=3, step=1)

# Use "Yes" or "No" options for binary features
work_accident = st.selectbox('Work Accident', ['No', 'Yes'])
promoted = st.selectbox('Promoted in Last 5 Years', ['No', 'Yes'])

# Define the salary level as a dropdown
salary = st.selectbox('Salary Level', ['low', 'medium', 'high'])

# Define the department as a dropdown
department = st.selectbox('Department', [
    'IT', 'RandD', 'accounting', 'hr', 'management', 'marketing',
    'product_mng', 'sales', 'support', 'technical'
])

# Encode user inputs into numerical format for the model
ordinal_salary = {'low': 0, 'medium': 1, 'high': 2}
department_mapping = {
    'IT': 0, 'RandD': 1, 'accounting': 2, 'hr': 3,
    'management': 4, 'marketing': 5, 'product_mng': 6,
    'sales': 7, 'support': 8, 'technical': 9
}

# Map "Yes" or "No" to 1 or 0 for binary features
work_accident_encoded = 1 if work_accident == 'Yes' else 0
promoted_encoded = 1 if promoted == 'Yes' else 0

# Create the feature vector for prediction
features = np.array([
    satisfaction, evaluation, number_project, monthly_hours,
    tenure, work_accident_encoded, promoted_encoded, ordinal_salary[salary]
])

# One-hot encode the department
department_encoded = np.zeros(len(department_mapping))
department_encoded[department_mapping[department]] = 1

# Combine numerical features and one-hot encoded department
features = np.concatenate([features, department_encoded])

# Make prediction when the user clicks the "Predict" button
if st.button('Predict'):
    prediction = model.predict([features])[0]
    if prediction == 1:
        st.success('The employee is likely to leave.')
    else:
        st.success('The employee is likely to stay.')
