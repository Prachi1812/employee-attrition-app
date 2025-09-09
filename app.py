import logging
logging.getLogger("streamlit.runtime.scriptrunner").setLevel(logging.ERROR)

import os
import streamlit as st
from streamlit_navigation_bar import st_navbar
import home
import dashboard
import prediction
import analysis

# ---------------------------
# Page configuration
# ---------------------------
st.set_page_config(
    page_title="Employee Attrition Prediction",
    page_icon="📊",
    layout="wide"
)

# ---------------------------
# Navbar setup
# ---------------------------
page_names = ["Home", "Analysis", "Dashboard", "Prediction"]

styles = {
    "nav": {
        "background-color": "#5a1e82",
        "justify-content": "center",
        "padding": "10px",
    },
    "span": {
        "color": "#ffffff",
        "padding": "14px",
        "font-size": "18px",
    },
    "active": {
        "background-color": "#5a1e82",
        "color": "#ffffff",
        "font-weight": "bold",
        "padding": "14px",
        "border-bottom": "3px solid #ffffff",
    },
    "hover": {
        "background-color": "rgba(255, 255, 255, 0.35)",
    },
}

selected_page = st_navbar(page_names, styles=styles)

# ---------------------------
# Page routing
# ---------------------------
pages = {
    "Home": home.show_home,
    "Analysis": analysis.show_analysis,
    "Dashboard": dashboard.show_dashboard,
    "Prediction": prediction.show_prediction
}

if selected_page in pages:
    try:
        pages[selected_page]()
    except FileNotFoundError as e:
        st.error(f"File not found: {e}")
    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.error("Page not found 🚨")
