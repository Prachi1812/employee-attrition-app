import streamlit as st
import pickle



# Load the model
model = pickle.load(open('random_forest_model.pkl', 'rb'))

def predict(input_data):
    prediction = model.predict([input_data])
    if (prediction[0] == 0):
        return 'stay'
    else:
        return 'leave'


def classify():
    st.markdown("<h1 style='text-align: center;'>Employee Attrition Prediction</h1 >", unsafe_allow_html=True)  
    st.write(" ")
   
    # Create form to input employee details
    with st.form("prediction_form"):
        satisfaction_level = st.slider("Satisfaction Level", 0.0, 1.0, step=0.01)
        evaluation_score = st.slider("Last Evaluation", 0.0, 1.0, step=0.01)
        project_count = st.number_input("Number of Projects", 1, 10)
        monthly_hours = st.number_input("Average Monthly Hours", 0, 250)
        tenure = st.number_input("Time Spent at Company", 1, 10)
        had_accident = st.selectbox("Work Accident", ["No", "Yes"])
        had_promotion = st.selectbox("Promotion in Last 5 Years", ["No", "Yes"])
        salary_level = st.selectbox("Salary", ["Low", "Medium", "High"])
        
        # Map salary_level to numerical values
        salary_map = {"Low": 0, "Medium": 1, "High": 2}
        salary_level_num = salary_map[salary_level]

        # Map salary_level to numerical values
        accident_map = {"No": 0, "Yes": 1}
        had_accident_num = accident_map[had_accident]

        promotion_map = {"No": 0, "Yes": 1}
        had_promotion_num = promotion_map[had_promotion]

        # Calculate additional fields
        overtime = int(monthly_hours > 174)
        work_intensity = project_count * monthly_hours
        
        # Prevent division by zero for work-life balance
        work_life_balance = satisfaction_level / (monthly_hours * tenure) if tenure > 0 and monthly_hours > 0 else 0


        input_data = [satisfaction_level, evaluation_score, project_count, monthly_hours, tenure,
                          had_accident_num, had_promotion_num, salary_level_num, work_intensity, overtime, work_life_balance]

        Prediction = ''
        #button for prediction
        # Submit button
        # Button for prediction
        submitted = st.form_submit_button("Predict")
        if submitted:
            prediction = predict(input_data)
            
            # Display appropriate message and image based on prediction
            if prediction == 'stay':
                st.success("The employee is predicted to stay.")
                col1, col2, col3 = st.columns([1, 1, 1])
                with col2:
                    st.image("static\img\stay1.jpeg", caption="Employee likely to stay",width=300)
            else:
                st.warning("The employee is predicted to leave.")
                col1, col2, col3 = st.columns([1, 1, 1])
                with col2:
                    st.image("static\img\Leave.jpeg", caption="Employee likely to leave",width=300)
                



def show_prediction():
    classify()

