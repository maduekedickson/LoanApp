import streamlit as st
import pandas as pd
import pickle

st.image('images.jpeg')
# Load the pickled model
loaded_pickle_model = pickle.load(open("random_forest_model.pkl", "rb"))

def predict_loan_approval(data):
    # Use the loaded model to make predictions
    prediction = loaded_pickle_model.predict(data)
    return prediction

def main():
    st.title("Loan Approval Machine Learning Model")

    # Input form for user to enter data
    st.header("Input Data")
    gender = st.selectbox("Gender", ["Male", "Female"])
    married = st.selectbox("Married", ["Yes", "No"])
    dependents = st.number_input("Dependents", min_value=0, max_value=10, value=0)
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self Employed", ["Yes", "No"])
    applicant_income = st.number_input("Applicant Income", value=0)
    coapplicant_income = st.number_input("Coapplicant Income", value=0)
    loan_amount = st.number_input("Loan Amount", value=0)
    loan_amount_term = st.number_input("Loan Amount Term", value=0)
    credit_history = st.selectbox("Credit History", [0.0, 1.0])
    property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

    # Mapping input values to numerical values
    gender_map = {'Male': 1, 'Female': 0}
    married_map = {'Yes': 1, 'No': 0}
    education_map = {'Graduate': 1, 'Not Graduate': 0}
    self_employed_map = {'Yes': 1, 'No': 0}
    property_area_map = {'Urban': 0, 'Semiurban': 1, 'Rural': 2}

    # Create a DataFrame from the input data
    new_data = pd.DataFrame({
        'Gender': [gender_map[gender]],
        'Married': [married_map[married]],
        'Dependents': [dependents],
        'Education': [education_map[education]],
        'Self_Employed': [self_employed_map[self_employed]],
        'ApplicantIncome': [applicant_income],
        'CoapplicantIncome': [coapplicant_income],
        'LoanAmount': [loan_amount],
        'Loan_Amount_Term': [loan_amount_term],
        'Credit_History': [credit_history],
        'Property_Area': [property_area_map[property_area]]
    })

    # Button to predict loan approval
    if st.button("Predict Loan Approval"):
        prediction = predict_loan_approval(new_data)
        if prediction[0] == 1:
            st.success("Loan is Approved 👍")
        else:
            st.error("Loan is Rejected 👎")

if __name__ == "__main__":
    main()