import streamlit as st
from PIL import Image
import streamlit.components.v1 as components



# Custom CSS to style buttons
def set_button_style():
    st.markdown("""
        <style>
        .stButton button {
            background-color: #41066f;  /* Dark purple */
            color: white;
            border-radius: 8px;
            padding: 0.5em 1em;
            font-size: 16px;
            border: 2px solid #793785;  /* Lighter purple for border */
            transition: background-color 0.3s ease;
        }

        .stButton button:hover {
            background-color: #5f0b9d;  /* Slightly lighter purple on hover */
        }
        </style>
    """, unsafe_allow_html=True)


def show_home():
    # Header
    st.markdown("<h2 style='text-align: center;'>Employee Attrition Prediction</h2>", unsafe_allow_html=True)
    
    # Load and center the image
    image = Image.open('static\img\Screenshot 2024-08-10 003741.png')
    col1, col2, col3 = st.columns([1, 1, 2])
    with col2:
        st.image(image, width=400)  # Adjust the width as needed

    # Welcome Section
    html_code1 = """
    <head>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.0.1/dist/css/bootstrap.min.css" rel="stylesheet" 
    integrity="sha384-+0n0xVW2eSR5OomGNYDnhzAbDsOXxcvSN1TPprVMTNDbiYZCxYbOOl7+AMvyTG2x" crossorigin="anonymous">

    <style>
        .hero-section {
            background-color: #f9f9f9;
            color: #0a0a0a;
            padding: 60px 0;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }

        .hero-text {
            padding: 20px;
            border-radius: 10px;
            font-family: Arial, sans-serif;
        }

        .btn-primary-custom {
            background-color: #41066f;
            border-color: #41066f;
        }
    </style>
    </head>

    <body>
        <!-- Hero Section -->
        <header class="hero-section text-center">
            <div class="hero-text">
                <h4 class="display-4">Welcome to our Employee Attrition Prediction app!</h4>
                <p class="lead">Leverage machine learning to understand and improve employee retention. Make informed
                    decisions with our advanced prediction model.</p>
            </div>
        </header>
    </body>
    """
    components.html(html_code1, height=250)
    # Key Features Section
    html_code = """
    <head>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.0.1/dist/css/bootstrap.min.css" rel="stylesheet" 
        integrity="sha384-+0n0xVW2eSR5OomGNYDnhzAbDsOXxcvSN1TPprVMTNDbiYZCxYbOOl7+AMvyTG2x" crossorigin="anonymous">
        
        <style>
            .features-icon {
                font-size: 3rem;
                color: #41066f;
            }

            .feature-box {
                border: 1px solid #793785;
                border-radius: 10px;
                padding: 20px;
                text-align: center;
                box-shadow: 0 4px 6px rgba(89, 36, 94, 0.1);
                transition: transform 0.3s;
                display: flex;
                flex-direction: column;
                justify-content: space-between;
            }

            .feature-box:hover {
                transform: scale(1.05);
            }

            .btn-primary-custom {
                background-color: #41066f;
                border-color: #41066f;
            }
        </style>
    </head>

    <body>
        <!-- Features Section -->
        <section class="container mt-5">
            <h2 class="text-center mb-4">Our Key Features</h2>
            <div class="row">
                <div class="col-md-4 d-flex mb-4">
                    <div class="feature-box flex-grow-1">
                        <div class="features-icon mb-3">
                            <i class="bi bi-bar-chart-line"></i>
                        </div>
                        <h4>Employee Attrition Prediction</h4>
                        <p>Accurately predict the likelihood of employees staying or leaving based on various factors.</p>
                    </div>
                </div>
                <div class="col-md-4 d-flex mb-4">
                    <div class="feature-box flex-grow-1">
                        <div class="features-icon mb-3">
                            <i class="bi bi-graph-up"></i>
                        </div>
                        <h4>Comprehensive Analysis</h4>
                        <p>Explore detailed analysis and insights into the factors affecting employee Attrition.</p>
                    </div>
                </div>
                <div class="col-md-4 d-flex mb-4">
                    <div class="feature-box flex-grow-1">
                        <div class="features-icon mb-3">
                            <i class="bi bi-gear"></i>
                        </div>
                        <h4>Interactive Tools</h4>
                        <p>Use our interactive tools to input data and receive predictions on employee Attrition.</p>
                    </div>
                </div>
            </div>
        </section>

        <!-- How It Works Section -->
        <section class="container mt-5">
            <h2 class="text-center mb-4">How It Works</h2>
            <p class="text-center">Our advanced machine learning models analyze factors like job satisfaction, 
               performance evaluations, and work conditions. Input employee data into our prediction tool to get insights 
               into their likelihood of staying or leaving.</p>
        </section>

        <!-- Call to Action -->
        <section class="container mt-5 text-center">
            <h2>Ready to Get Started?</h2>
            <p>Explore our prediction tool and see how it can help you improve employee retention in your organization.</p>
        
        </section>

    </body>
    """
    
    components.html(html_code, height=1500)
    