# -*- coding: utf-8 -*-
"""
Created on Sun Apr 17 19:48:37 2022

@author: alber
"""

import streamlit as st

st.title('Credit Spreads Wizard')

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression

dr = r'C:\Users\alber\Documents\Spring 22\AI&ML\loan-train.csv'
Data = pd.read_csv(dr)

## The model
df = Data
df = pd.get_dummies(df, columns=['Gender'])
df = df.drop(['Loan_ID', 'Gender_Male'], axis=1)

X = df[['ApplicantIncome', 'CoapplicantIncome','LoanAmount', 'Married','Dependents','Education','Self_Employed','Loan_Amount_Term','Credit_History','Property_Area','Gender_Female']]
y = df['Loan_Status']

# Split data into training and test sets
(X_train, X_test,
 y_train, y_test) = train_test_split(X, y, test_size=0.25)

## Bunch of sliders

col_names = df.columns
Drop_Gender = ['Loan_Status']
col_names = col_names.drop(Drop_Gender)

for i in col_names:
    globals()[f'{i}'] = st.slider(i, int(df[i].min()), int(df[i].max()), int(df[i].median()))


## The regression

log_loan = LogisticRegression()
log_loan.fit(X_train, y_train)
individual = [ApplicantIncome, CoapplicantIncome, LoanAmount, Married, Dependents, Education, Self_Employed, Loan_Amount_Term, Credit_History, Property_Area, Gender_Female]
individual = np.array(individual)
individual = individual.reshape(1, -1)   
y_pred = log_loan.predict_proba(individual)
pred = round(y_pred[0,1]*100,2)

if pred < 50:
    st.metric(label="Probability of Acceptance", value=str(pred)+'%', delta="-Denied")
else:
    st.metric(label="Probability of Acceptance", value=str(pred)+'%', delta="Accepted")
    st.balloons()



















































