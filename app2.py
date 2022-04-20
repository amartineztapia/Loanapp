# -*- coding: utf-8 -*-
"""
Created on Sun Apr 17 19:48:37 2022

@author: alber
"""

import streamlit as st

st.title('Loan Application and Credit Spread Dashboard')

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression


Data = pd.read_csv('loan-train.csv')


primaryColor="#F63366"
backgroundColor="#FFFFFF"
secondaryBackgroundColor="#F0F2F6"
textColor="#262730"
font="sans serif"

## The model
df = Data
df = pd.get_dummies(df, columns=['Gender'])
df = df.drop(['Loan_ID', 'Gender_Male'], axis=1)

X = df[['ApplicantIncome', 'CoapplicantIncome','LoanAmount', 'Married','Dependents','Education','Self_Employed','Loan_Amount_Term','Credit_History','Property_Area','Gender_Female']]
y = df['Loan_Status']

# Split data into training and test sets
(X_train, X_test,
 y_train, y_test) = train_test_split(X, y, test_size=0.25)

## Inputs

col_names = df.columns
col_names = col_names.drop(['Loan_Status'])
col_names3 = col_names.drop(['ApplicantIncome', 'CoapplicantIncome','LoanAmount','Dependents','Education','Loan_Amount_Term','Credit_History','Property_Area'])

Binary_options = {
    0: "No",
    1: "Yes",
    }

for i in col_names3:
    globals()[f'{i}'] = st.radio(i, (0, 1), index = int(df[i].median()), format_func=lambda x: Binary_options.get(x))
    

Education = st.radio("College Completed", (0, 1), index = int(df["Education"].median()), format_func=lambda x: Binary_options.get(x))
Credit_History = st.radio("Credit History Available", (0, 1), index = int(df["Credit_History"].median()), format_func=lambda x: Binary_options.get(x))


ApplicantIncome = st.slider('Applicant Income ($/Mo)', int(df['ApplicantIncome'].min()), int(df['ApplicantIncome'].max()), int(df['ApplicantIncome'].median()))

CoapplicantIncome = st.slider('Coapplicant Income ($/Mo)', int(df['CoapplicantIncome'].min()), int(df['CoapplicantIncome'].max()), int(df['CoapplicantIncome'].median()))

LoanAmount = st.slider('Loan Amount (in thousands)', int(df['LoanAmount'].min()), int(df['LoanAmount'].max()), int(df['LoanAmount'].median()))
Loan_Amount_Term = st.slider('Loan Amount Term (in months)', int(df['Loan_Amount_Term'].min()), int(df['Loan_Amount_Term'].max()), int(df['Loan_Amount_Term'].median()))

Dependents = st.slider('Dependents', int(df['Dependents'].min()), int(df['Dependents'].max()), int(df['Dependents'].median()))



P_area = {
    0: "Rural",
    1: "Semi-Urban",
    2: "Urban"
    }


Property_Area = st.multiselect('Property Area (Select One)', (0, 1, 2), default = int(df["Property_Area"].median()), format_func=lambda x: P_area.get(x))
Property_Area = Property_Area[0]



## The regression

if st.button("Calculate"):
    log_loan = LogisticRegression()
    log_loan.fit(X_train, y_train)
    individual = [ApplicantIncome, CoapplicantIncome, LoanAmount, Married, Dependents, Education, Self_Employed, Loan_Amount_Term, Credit_History, Property_Area, Gender_Female]
    individual = np.array(individual)
    individual = individual.reshape(1, -1)   
    y_pred = log_loan.predict_proba(individual)
    pred = round(y_pred[0,1]*100,2)
    if pred < 50:
        st.metric(label="Probability of Acceptance", value=str(pred)+'%', delta="-Denied")
        st.metric(label="Credit Spread Direction (6mo)", value="Downwards", delta="-Spread will shrink")

    else:
        st.metric(label="Probability of Acceptance", value=str(pred)+'%', delta="Accepted")
        st.metric(label="Credit Spread Direction (6mo)", value="Downwards", delta="-Spread will shrink")
        st.balloons()


from PIL import Image
image = Image.open('Manus.jfif')

st.image(image, caption='Manus', width=128)















