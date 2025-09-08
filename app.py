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
        "background-color": "#5a1e82",   # Dark purple background
        "justify-content": "center",     # Center align items
        "padding": "10px",
    },
    "span": {
        "color": "#ffffff",              # White text
        "padding": "14px",
        "font-size": "18px",
    },
    "active": {
        "background-color": "#5a1e82",   # Same background for active
        "color": "#ffffff",              # White text
        "font-weight": "bold",           # Bold active page
        "padding": "14px",
        "border-bottom": "3px solid #ffffff",
    },
    "hover": {
        "background-color": "rgba(255, 255, 255, 0.35)",  # Hover effect
    },
}

# Create navbar
selected_page = st_navbar(page_names, styles=styles)

# ---------------------------
# Page routing
# ---------------------------
pages = {
    "Home": home.show_home,
    "Analysis": analysis.show_analysis,
    "Dashboard": dashboard.show_dashboard,
    "Prediction": prediction.show_prediction,
}

# Call the correct page
if selected_page in pages:
    pages[selected_page]()
else:
    st.error("Page not found 🚨")
