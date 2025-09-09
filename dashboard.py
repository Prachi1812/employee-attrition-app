import plotly.express as px

import streamlit as st
import pandas as pd
import plotly.graph_objects as go


def show_dashboard():

   
    def load_data():
        df = pd.read_csv('Employee.csv')

        return df

    df = load_data()
    
    # Define the mapping dictionaries for the categorical levels
    satisfaction_mapping = {1: 'Low', 2: 'Medium', 3: 'High', 4: 'Very High'}
    #performance_mapping = {1: 'Low', 2: 'Good', 3: 'Excellent', 4: 'Outstanding'}
    work_life_balance_mapping = {1: 'Bad', 2: 'Good', 3: 'Better', 4: 'Best'}

    # Apply the mappings to the corresponding columns
    df['Environment Satisfaction'] = df['Environment Satisfaction'].map(satisfaction_mapping)
    df['Job Involvement'] = df['Job Involvement'].map(satisfaction_mapping)
    df['Job Satisfaction'] = df['Job Satisfaction'].map(satisfaction_mapping)
    #df['Performance Rating'] = df['Performance Rating'].map(performance_mapping)
    df['Relationship Satisfaction'] = df['Relationship Satisfaction'].map(satisfaction_mapping)
    df['Work Life Balance'] = df['Work Life Balance'].map(work_life_balance_mapping)

    # Sidebar with employee filters
    with st.sidebar.expander("Filters", expanded=False):
        st.sidebar.header("Choose your filter: ")

    section= st.sidebar.selectbox("Select Options:",['HR Analysis','Attrition Analysis'])
  
    # Department filter
    department_filter = st.sidebar.multiselect(
        "Select Department(s):", 
        options=df["Department"].unique(), 
        default=df["Department"].unique()
    )

    # Job Role filter
    jobrole_filter = st.sidebar.multiselect(
        "Select Job Role(s):", 
        options=df["Job Role"].unique(), 
        default=df["Job Role"].unique()
    )

    # Gender filter
    gender_filter = st.sidebar.multiselect(
        "Select Gender(s):", 
        options=df["Gender"].unique(), 
        default=df["Gender"].unique()
    )

    # Age filter using 'CF_age band'
    age_band_filter = st.sidebar.multiselect(
        "Select Age Band(s):", 
        options=df["CF_age band"].unique(), 
        default=df["CF_age band"].unique()
    )

    # Apply filters to the dataset
    filtered_df = df[(df["Department"].isin(department_filter)) & 
                    (df["Job Role"].isin(jobrole_filter)) & 
                    (df["Gender"].isin(gender_filter)) &
                    (df["CF_age band"].isin(age_band_filter))]

    # Calculate metrics for KPIs based on filtered data
    employee_count = filtered_df['Employee Number'].nunique()
    attrition_count = filtered_df[filtered_df['Attrition'] == 'Yes']['Attrition'].count()
    attrition_rate = (attrition_count / employee_count) * 100 if employee_count != 0 else 0
    active_employees = employee_count - attrition_count
    avg_age = filtered_df['Age'].mean()

    # Data for department-wise attrition
    department_attrition = filtered_df[filtered_df['Attrition'] == 'Yes'].groupby('Department').size().reset_index(name='Attrition Count')
    if attrition_count != 0:
        department_attrition['Percentage'] = (department_attrition['Attrition Count'] / attrition_count) * 100

    # Data for number of employees by age band
    employee_age_band = filtered_df.groupby('CF_age band').size().reset_index(name='No. of Employees')

    # Data for education field-wise attrition
    education_attrition = filtered_df[filtered_df['Attrition'] == 'Yes'].groupby('Education Field').size().reset_index(name='Attrition Count')

    # Data for attrition rate by gender for different age bands
    attrition_gender_age_band = filtered_df[filtered_df['Attrition'] == 'Yes'].groupby(['CF_age band', 'Gender']).size().unstack().reset_index()

    # Job satisfaction ratings by job role
    job_satisfaction = filtered_df.groupby('Job Role')['Job Satisfaction'].value_counts().unstack().fillna(0).astype(int)

    # New Visualizations Data
    monthly_income_dist = filtered_df['Monthly Income']
    years_at_company_attrition = filtered_df[['Years At Company', 'Attrition']]
    total_working_years_attrition = filtered_df[['Total Working Years', 'Attrition']]
    marital_status_attrition = filtered_df[filtered_df['Attrition'] == 'Yes'].groupby('Marital Status').size().reset_index(name='Attrition Count')

    
    # Dashboard Title
    st.title("Analysis Dashboard")
   
    # Row 1: KPIs
    st.subheader("Key Metrics")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Employee Count", employee_count)
    col2.metric("Attrition Count", attrition_count)
    col3.metric("Attrition Rate", f"{attrition_rate:.2f}%")
    col4.metric("Active Employees", active_employees)
    col5.metric("Average Age", f"{avg_age:.0f}")
    
    
    #tabs0=st.tabs(['HR Analysis','Attrition Analysis'])
    if section == 'HR Analysis':
        st.write(' ')
        st.subheader("Employee Demographics")
        tabs = st.tabs(["Gender Breakdown", "Age Distribution","Education Field Distribution","Education Distribution","Marital Status"])
    
        # Dataset Information
        with tabs[0]:

            gender_breakdown = filtered_df['Gender'].value_counts().reset_index()
            gender_breakdown.columns = ['Gender', 'Count']
            fig_gender = px.pie(gender_breakdown, names='Gender', values='Count', 
                                title="Gender Distribution", 
                                color_discrete_sequence=px.colors.qualitative.Set2, hole=0.4)
            fig_gender.update_traces(textinfo='percent+label')
            st.write(fig_gender)

        with tabs[1]:
            # Row 3: No. of Employees by Age Band
            if not employee_age_band.empty:
                fig2 = px.bar(employee_age_band, x='CF_age band', y='No. of Employees',
                            title='No. of Employees by Age Band',
                            color_discrete_sequence=px.colors.qualitative.Set2,
                            labels={'No. of Employees':'Employees'},
                            height=400)
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.write("No data available for the selected filters.")
        
        with tabs[2]:
            # 2. Education Level Distribution

            fig_education_field_dist = px.bar(filtered_df['Education Field'].value_counts().reset_index(),
                                    x='index', y='Education Field', 
                                    title="Distribution of Education Fields",
                                    labels={'index': 'Education Field', 'Education Field': 'Count'},
                                    color_discrete_sequence=px.colors.qualitative.Set2)
            st.write(fig_education_field_dist)
        
        with tabs[3]:
            education_distribution = filtered_df['Education'].value_counts().reset_index()
            education_distribution.columns = ['Education', 'Count']
            fig_education = px.bar(education_distribution, x='Education', y='Count', 
                                title="Distribution of Education",
                                labels={'Education Field': 'Education', 'Count': 'Count'},
                                color_discrete_sequence=px.colors.qualitative.Set2)
            st.write(fig_education)

        with tabs[4]:
            marital_status_distribution = filtered_df['Marital Status'].value_counts().reset_index()
            marital_status_distribution.columns = ['Marital Status', 'Count']
            fig_marital_status = px.pie(marital_status_distribution, values='Count', names='Marital Status',
                                        title="Distribution of Marital Status",
                                        color_discrete_sequence=px.colors.qualitative.Set2, hole=0.4)
            fig_marital_status.update_traces(textinfo='percent+label')
            st.write(fig_marital_status)
    
        st.write('')
        st.subheader('Employee Satisfaction')
        tabs1= st.tabs(['Overall Job Satisfaction Distribution','Job Satisfaction Distribution by Job Role','Job Satisfaction by Gender '])
        
        with tabs1[0]:
            # Overall Job Satisfaction Count (Histogram)
            st.subheader("Overall Job Satisfaction Distribution")
            fig = px.histogram(filtered_df, x='Job Satisfaction', color='Job Satisfaction',
                            title="Overall Job Satisfaction Count",
                            color_discrete_sequence=px.colors.qualitative.Set2,
                            labels={'Job Satisfaction': 'Job Satisfaction Level', 'count': 'Number of Employees'},
                            category_orders={"Job Satisfaction": ['Low', 'Medium', 'High', 'Very High']})
            fig.update_layout(xaxis_title="Job Satisfaction Level", yaxis_title="Count")
            st.plotly_chart(fig)


        with tabs1[1]:
            # Job Satisfaction by Job Role
            job_satisfaction_by_role = filtered_df.groupby('Job Role')['Job Satisfaction'].value_counts().unstack().fillna(0)

            st.subheader("Job Satisfaction by Job Role")
            fig = px.bar(job_satisfaction_by_role, 
                        title="Job Satisfaction Distribution by Job Role", 
                        labels={'value': 'Number of Employees', 'Job Satisfaction': 'Job Satisfaction Level'})
            fig.update_layout(barmode='group', xaxis_title="Job Role", yaxis_title="Count", legend_title="Satisfaction Level")
            st.plotly_chart(fig)
        with  tabs1[2]:
            # Job Satisfaction by Gender
            job_satisfaction_by_gender = filtered_df.groupby(['Gender', 'Job Satisfaction']).size().unstack(fill_value=0)

            st.subheader("Job Satisfaction by Gender")
            fig = go.Figure()

            # Add traces for each satisfaction level
            for level in ['Low', 'Medium', 'High', 'Very High']:
                fig.add_trace(go.Bar(
                    x=job_satisfaction_by_gender.index,
                    y=job_satisfaction_by_gender[level],
                    name=level
                ))

            fig.update_layout(barmode='stack', title="Job Satisfaction by Gender", 
                            xaxis_title="Gender", yaxis_title="Count", legend_title="Satisfaction Level")
            st.plotly_chart(fig)


    
        st.write('')
        st.subheader('Job Role')
        tabs3=st.tabs(['Distribution of Job Role','Business Travel Frequency by Job Role','Work-Life Balance by Job Role'])
        with tabs3[0]:
            job_distribution = filtered_df['Job Role'].value_counts().reset_index()
            job_distribution.columns = ['Job Role', 'Count']
            fig_education = px.bar(job_distribution, x='Job Role', y='Count', 
                                title="Distribution of Job Role",
                                labels={'Job Role': 'Job Role', 'Count': 'Count'},
                                color_discrete_sequence=px.colors.qualitative.Set2 )
            st.write(fig_education)
            

        with tabs3[1]:
            travel_job_role = filtered_df.groupby('Job Role')['Business Travel'].value_counts().unstack().fillna(0)

            fig_travel_job_role = px.bar(travel_job_role, barmode='stack',title='Business Travel Frequency by Job Role')
            st.plotly_chart(fig_travel_job_role)

        with tabs3[2]:
           # Grouping by Job Role and Work Life Balance
            work_life_by_jobrole = filtered_df.groupby('Job Role')['Work Life Balance'].value_counts().unstack().fillna(0)

            # Creating a stacked bar chart for Work Life Balance by Job Role
            fig_jobrole = px.bar(work_life_by_jobrole, 
                                title='Work Life Balance by Job Role',
                                labels={'value': 'Number of Employees', 'Work Life Balanceed': 'Work Life Balance'}
                                )

            st.plotly_chart(fig_jobrole)
            
        st.write('')
        st.subheader('Department')
        tabs4=st.tabs(['Department Distribution','Monthly Income by Department','Overtime by  Department'])

        with tabs4[0]:
            department_distribution = filtered_df['Department'].value_counts().reset_index()
            department_distribution.columns = ['Department', 'Count']
            fig_department = px.bar(department_distribution, x='Department', y='Count',
                                    title="Department Distribution",   
                                    labels={'Department': 'Department', 'Count': 'Count'},
                                    color_discrete_sequence=px.colors.qualitative.Set2)
            st.write(fig_department)

        with tabs4[1]:
            income_dept = filtered_df.groupby('Department')['Monthly Income'].mean().reset_index()

            fig_income_dept = px.bar(income_dept, x='Department', y='Monthly Income',
                                    title='Average Monthly Income by Department',color_discrete_sequence=px.colors.qualitative.Set2, labels={'Monthly Income': 'Average Income'})
            st.plotly_chart(fig_income_dept)


        with tabs4[2]:
            department_overtime = filtered_df.groupby(['Department', 'Over Time']).size().reset_index(name='Count')
            fig_dept_overtime = px.bar(department_overtime, x='Department', y='Count', color='Over Time', 
                                       title="Department vs Overtime",
                                       barmode='group')
            st.write(fig_dept_overtime)

        st.write('')
        st.subheader('Performance Rating')
        # Map Performance Rating to Categories
        performance_mapping = {1: 'Low', 2: 'Good', 3: 'Excellent', 4: 'Outstanding'}
        filtered_df['Performance Category'] = filtered_df['Performance Rating'].map(performance_mapping)

        tabs5=st.tabs(['Performance Rating Distribution','Percent Salary Hike by Performance Rating'])
        with tabs5[0]:
            performance_dist = filtered_df['Performance Category'].value_counts().reset_index()
            performance_dist.columns = ['Performance Category', 'Count']

            fig_performance_dist = px.bar(performance_dist, x='Performance Category', y='Count', title='Performance Rating Distribution',
                                        labels={'Count': 'Number of Employees'}, color='Performance Category')
            st.plotly_chart(fig_performance_dist)

        with tabs5[1]:
            # Percent Salary Hike by Performance Rating (Boxplot)

            fig_hike_performance = px.box(filtered_df, x='Performance Category', y='Percent Salary Hike', 
                                        title='Percent Salary Hike by Performance Rating',
                                        labels={'Performance Category': 'Performance Rating', 'Percent Salary Hike': 'Salary Hike (%)'},
                                        color='Performance Category')
            st.plotly_chart(fig_hike_performance)

        # Employee Overview (Card View)
        st.write("### Employee Overview")

        emp_id = st.selectbox("Select Employee ID", filtered_df['Employee ID'].unique())

        employee_details = filtered_df[filtered_df['Employee ID'] == emp_id]

        if not employee_details.empty:
            st.write(f"**Name:** {employee_details['Employee First Name'].values[0]} {employee_details['Employee Last Name'].values[0]}")
            st.write(f"**Department:** {employee_details['Department'].values[0]}")
            st.write(f"**Job Role:** {employee_details['Job Role'].values[0]}")
            st.write(f"**Monthly Income:** ${employee_details['Monthly Income'].values[0]}")
            st.write(f"**Job Satisfaction:** {employee_details['Job Satisfaction'].values[0]}")
            st.write(f"**Performance Rating:** {employee_details['Performance Category'].values[0]}")




           

      
    if section == 'Attrition Analysis' :
        st.subheader('Attrition Analysis')
        st.write('')
        tabs6=st.tabs(['Department-wise Attrition','Attrition by Age Band','Attrition by Job Role','Attrition vs. Overtime'])
        
        with tabs6[0]:
            st.write('')
            # Row 2: Department-wise Attrition
            st.subheader("Department-wise Attrition")
            if not department_attrition.empty:
                fig1 = px.pie(department_attrition, values='Attrition Count', names='Department', 
                            title='Department-wise Attrition',
                            hole=0.3)
                st.plotly_chart(fig1, use_container_width=True)
            else:
                st.write("No data available for the selected filters.")

        with tabs6[1]:
            attrition_age_band = filtered_df[filtered_df['Attrition'] == 'Yes'].groupby('CF_age band').size().reset_index(name='Count')

            fig_attrition_age_band = px.bar(attrition_age_band, x='CF_age band', y='Count', title='Attrition by Age Band',
                                            labels={'Count': 'Number of Employees'}, color='CF_age band')
            st.plotly_chart(fig_attrition_age_band)
        
        with tabs6[2]:
            attrition_job_role = filtered_df[filtered_df['Attrition'] == 'Yes'].groupby('Job Role').size().reset_index(name='Count')

            fig_attrition_job_role = px.bar(attrition_job_role, x='Job Role', y='Count', title='Attrition by Job Role',
                                            labels={'Count': 'Number of Employees'}, color='Job Role')
            st.plotly_chart(fig_attrition_job_role)

        with tabs6[3]: 
            attrition_overtime = filtered_df.groupby(['Over Time', 'Attrition']).size().unstack().fillna(0)

            fig_attrition_overtime = px.bar(attrition_overtime, barmode='stack', title='Attrition vs. Overtime',
                                            labels={'value': 'Number of Employees'})
            st.plotly_chart(fig_attrition_overtime)
        
        tabs7=st.tabs(['Education Field-wise Attrition','Attrition Rate by Gender for Different Age Bands','Attrition by Marital Status'])
        with tabs7[0]:
            # Row 4: Education Field-wise Attrition
            st.subheader("Education Field-wise Attrition")
            if not education_attrition.empty:
                fig3 = px.bar(education_attrition, x='Attrition Count', y='Education Field', orientation='h',
                            title='Education Field-wise Attrition')
                st.plotly_chart(fig3, use_container_width=True)
            else:
                st.write("No data available for the selected filters.")
        with tabs7[1]:

            # Row 5: Attrition Rate by Gender for Different Age Bands
            st.subheader("Attrition Rate by Gender for Different Age Bands")
            if not attrition_gender_age_band.empty:
                fig4 = go.Figure()
                fig4.add_trace(go.Bar(x=attrition_gender_age_band['CF_age band'], y=attrition_gender_age_band['Male'],
                                    name='Male', marker_color='blue'))
                fig4.add_trace(go.Bar(x=attrition_gender_age_band['CF_age band'], y=attrition_gender_age_band['Female'],
                                    name='Female', marker_color='orange'))
                fig4.update_layout(barmode='group', title='Attrition Rate by Gender for Different Age Bands')
                st.plotly_chart(fig4, use_container_width=True)
            else:
                st.write("No data available for the selected filters.")

        with tabs7[2]:

            # Row 10: Attrition by Marital Status
            st.subheader("Attrition by Marital Status")
            if not marital_status_attrition.empty:
                fig9 = px.pie(marital_status_attrition, values='Attrition Count', names='Marital Status', 
                            title='Attrition by Marital Status', hole=0.3)
                st.plotly_chart(fig9, use_container_width=True)
            else:
                st.write("No data available for the selected filters.")
        
        # Display a data table for the filtered employees
        st.write("## Filtered Employees")
        st.dataframe(filtered_df[['Employee ID', 'Employee First Name', 'Employee Last Name', 'Job Role', 'Department', 'Attrition', 'Age', 'Gender']])

