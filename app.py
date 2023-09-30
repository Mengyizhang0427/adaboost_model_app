# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 11:06:55 2023

@author: Starchild
"""

import streamlit as st
import pandas as pd
import pickle
import category_encoders as ce
import shap
import numpy as np



# Title
st.header("Adaboost+NearMiss ACLF death prediction model")

#input
Kidney=st.sidebar.selectbox("Whether Kidney failure",("YES","NO"))
st.write(f"Kidney:{Kidney}")
if Kidney=="YES":
    Kidney_0=1
    Kidney_1=0
else:
    Kidney_0=0
    Kidney_1=1
Brain=st.sidebar.selectbox("Whether Brain failure",("YES","NO"))
if Brain=="YES":
    Brain_0=0
    Brain_1=1
else:
    Brain_0=1
    Brain_1=0
Circulation=st.sidebar.selectbox("Whether Circulation failure",("YES","NO"))
if Circulation=="YES":
    Circulation_0=0
    Circulation_1=1
else:
    Circulation_0=1
    Circulation_1=0
Respiratory=st.sidebar.selectbox("Whether Respiratory failure",("YES","NO"))
if Respiratory=="YES":
    Respiratory_0=1
    Respiratory_1=0
else:
    Respiratory_0=0
    Respiratory_1=1
ast=st.sidebar.number_input("Enter ast")
alt=st.sidebar.number_input("Enter alt")
bilirubin=st.sidebar.number_input("Enter bilirubin")
INR=st.sidebar.number_input("Enter INR")
WBC=st.sidebar.number_input("Enter WBC")
platelet_count=st.sidebar.number_input("Enter platelet_count")
creatinine=st.sidebar.number_input("Enter creatinine")
sodium=st.sidebar.number_input("Enter sodium")
albumin=st.sidebar.number_input("Enter albumin")
heart_rate=st.sidebar.number_input("Enter heart_rate")
sbp=st.sidebar.number_input("Enter sbp")
dbp=st.sidebar.number_input("Enter dbp")
mbp=st.sidebar.number_input("Enter mbp")
resp_rate=st.sidebar.number_input("Enter resp_rate")
temperature=st.sidebar.number_input("Enter temperature")
spo2=st.sidebar.number_input("Enter spo2")
glucose=st.sidebar.number_input("Enter glucose")
gender=st.sidebar.selectbox("gender",("M","F"))
if gender=="YES":
    gender_0=1
    gender_1=0
else:
    gender_0=0
    gender_1=1  
age=st.sidebar.number_input("Enter age")



with open('/mount/src/adaboost_model_app/Adaboost+NearMiss.pkl', 'rb') as f:
    clf = pickle.load(f)
with open('/mount/src/adaboost_model_app/scaler_params.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('/mount/src/adaboost_model_app/explainer.pkl', 'rb') as f:
    explainer = pickle.load(f)


# If button is pressed
if st.button("Submit"):
    
    # Unpickle classifier
    # Store inputs into dataframe
    columns = ['Kidney_0','Kidney_1', 'Brain_0','Brain_1', 'Circulation_0','Circulation_1', 'Respiratory_0','Respiratory_1',
               'ast', 'alt',
           'bilirubin', 'INR', 'WBC', 'platelet_count', 'creatinine', 'sodium',
           'albumin', 'heart_rate', 'sbp', 'dbp', 'mbp', 'resp_rate',
           'temperature', 'spo2', 'glucose', 'gender_0','gender_1', 'age']
    X = pd.DataFrame([[Kidney_0,Kidney_1, Brain_0,Brain_1, Circulation_0,Circulation_1, 
                       Respiratory_0,Respiratory_1, ast, alt,
           bilirubin, INR, WBC, platelet_count, creatinine, sodium,
           albumin, heart_rate, sbp, dbp, mbp, resp_rate,
           temperature, spo2, glucose, gender_0,gender_1, age]], 
                     columns = ['Kidney_0','Kidney_1', 'Brain_0','Brain_1', 'Circulation_0','Circulation_1', 'Respiratory_0','Respiratory_1',
                                'ast', 'alt',
                            'bilirubin', 'INR', 'WBC', 'platelet_count', 'creatinine', 'sodium',
                            'albumin', 'heart_rate', 'sbp', 'dbp', 'mbp', 'resp_rate',
                            'temperature', 'spo2', 'glucose', 'gender_0','gender_1', 'age'])
    X = scaler.fit_transform(X)
    X= pd.DataFrame(X,columns=columns)
    st.dataframe(X)
    # Get prediction
    prediction = clf.predict(X)
    pred=clf.predict_proba(X)[0][1]
    shap_values2 = explainer(X)
    
    # Output prediction
    
    st.text(f"The probability of death of the patient is {pred}.")
    st.set_option('deprecation.showPyplotGlobalUse', False)
    fig=shap.plots.bar(shap_values2[0])
    st.pyplot(fig)
    
    
    
    
    
    
    
