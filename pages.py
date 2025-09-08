# pages.py

import streamlit as st

def show_home():
    st.title("Welcome to the Employee Attrition Prediction App")
    st.write("""
    This app helps analyze employee attrition data and make predictions about 
    potential employee churn. Use the Dashboard to explore visualizations, or 
    go to the Prediction page to try out the prediction model.
    """)

def show_dashboard():
    st.title("Dashboard")
    st.write("Here you can visualize employee attrition data.")

def show_prediction():
    st.title("Prediction")
    st.write("This page allows you to make predictions using the model.")
