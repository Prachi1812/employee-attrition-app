# Importing ToolKits
import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
import warnings
import io 
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, confusion_matrix, 
    roc_curve, roc_auc_score, precision_recall_curve, ConfusionMatrixDisplay, 
    RocCurveDisplay, PrecisionRecallDisplay
)
import matplotlib.pyplot as plt

# Function to load models from pickle files
def load_model(filename):
    with open(filename, 'rb') as file:
        model = pickle.load(file)
    return model

# Function to plot the confusion matrix
def plot_confusion_matrix(y_test, y_pred, title):
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap='Blues', linewidths=2.5, cbar=False)
    plt.xlabel('Predicted Labels')
    plt.ylabel('Actual Labels')
    plt.title(f'Confusion Matrix - {title}')
    plt.xticks([0.5, 1.5], ['Predicted Stay', 'Predicted Leave'])
    plt.yticks([0.5, 1.5], ['Actual Stay', 'Actual Leave'])
    st.pyplot(plt.gcf())

# Function to plot the ROC AUC curve
def plot_roc_auc(y_test, pred_prob, title):
    roc_auc = roc_auc_score(y_test, pred_prob[:, 1])
    fpr, tpr, _ = roc_curve(y_test, pred_prob[:, 1])
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic (ROC) - {title}')
    plt.legend(loc='lower right')
    st.pyplot(plt.gcf())


# Load test data (X_test and Y_test)
# Example:
X_test = pickle.load(open('X_test.pkl', 'rb'))
Y_test = pickle.load(open('Y_test (1).pkl', 'rb'))

def custome_layout(fig, title_size=28, hover_font_size=18, showlegend=False):
    fig.update_layout(
        showlegend=showlegend,
        title={
            "font": {
                "size": title_size,
                "family": "tahoma"
            }
        },
        hoverlabel={
            "bgcolor": "#000",
            "font_size": hover_font_size,
            "font_family": "arial"
        }
    )


def box_plot(the_df, column):
    fig = px.box(
        data_frame=the_df,
        x=column,
        title=f'{column.title().replace("_", " ")} Distribution & 5-Summary',
        template="plotly_dark",
        labels={column: column.title().replace("_", " ")},
        height=600,
        color_discrete_sequence=['#17B794']
    )
    custome_layout(fig, showlegend=False)
    return fig


def bar_plot(the_df, column, orientation="v", top_10=False):
    dep = the_df[column].value_counts()
    if top_10:
        dep = the_df[column].value_counts().nlargest(10)

    fig = px.bar(data_frame=dep,
                 x=dep.index,
                 y=dep / sum(dep) * 100,
                 orientation=orientation,
                 color=dep.index.astype(str),
                 title=f'Distribution of  {column.title().replace("_", " ")}',
                 color_discrete_sequence=["#17B794"],
                 labels={column: column.title().replace("_", " "),
                         "y": "Employees Frequency in PCT(%)"},
                 template="plotly_dark",
                 text=dep.apply(lambda x: f"{x / sum(dep) * 100:0.0f}%"),
                 height=650)

    fig.update_traces(
        textfont={
            "size": 20,
            "family": "consolas",
            "color": "#000"
        },
        hovertemplate="X Axis: %{x}<br>Y Axis: %{y:0.1f}%",
    )

    if orientation == "h":
        fig = px.bar(data_frame=dep,
                     y=dep.index[::-1],
                     x=dep[::-1] / sum(dep) * 100,
                     orientation=orientation,
                     color=dep.index.astype(str),
                     title=f'Distribution of {column.title().replace("_", " ")}',
                     color_discrete_sequence=["#17B394"],
                     labels={"y": column.title().replace("_", " "),
                             "x": "Employees Frequency in PCT(%)"},
                     template="plotly_dark",
                     text=dep[::-
                              1].apply(lambda x: f"{x / sum(dep) * 100:0.0f}%"),
                     height=650)

        fig.update_traces(
            textfont={
                "size": 20,
                "family": "consolas",
                "color": "#000"
            },
            hovertemplate="X Axis: %{y}<br>Y Axis: %{x:0.1f}%",
        )
    custome_layout(fig, title_size=28)
    return fig


def pie_chart(the_df, column):
    counts = the_df[column].value_counts()

    fig = px.pie(data_frame=counts,
                 names=counts.index,
                 values=counts,
                 title=f'Distribution of {column.title().replace("_", " ")}',
                 color_discrete_sequence=["#17B794", "#EEB76B", "#9C3D54"],
                 template="plotly_dark",
                 height=650
                 )

    custome_layout(fig, showlegend=True, title_size=28)
    pulls = np.zeros(len(counts))
    pulls[-1] = 0.1

    fig.update_traces(
        textfont={
            "size": 16,
            "family": "arial",
            "color": "#fff"
        },
        hovertemplate="Label:%{label}<br>Frequency: %{value:0.4s}<br>Percentage: %{percent}",
        marker=dict(line=dict(color='#000000', width=0.5)),
        pull=pulls,
    )

    return fig


# Main Visualization Function
def create_visualization(the_df, viz_type="box", data_type="number"):
    """
        This Function Take 3 Parameters [data_frame, viz_type, data_type]
        and return 3 
        1• [array of all created figures].
        2• df columns.
        3• target column Index according to dtype
    """
    figs = []
    num_columns = list(the_df.select_dtypes(include=data_type).columns)
    cols_index = []

    if viz_type == "box":
        for i in range(len(num_columns)):
            if the_df[num_columns[i]].nunique() > 10:
                figs.append(box_plot(
                    the_df, num_columns[i]))
                cols_index.append(i)

    if viz_type == "bar":
        for i in range(len(num_columns)):
            if the_df[num_columns[i]].nunique() < 8:
                figs.append(bar_plot(
                    the_df, num_columns[i]))
                cols_index.append(i)
            elif the_df[num_columns[i]].nunique() >= 8 and the_df[num_columns[i]].nunique() < 15:
                figs.append(bar_plot(
                    the_df, num_columns[i], "h"))
                cols_index.append(i)
            if the_df[num_columns[i]].nunique() >= 15:
                figs.append(bar_plot(
                    the_df, num_columns[i], "h", top_10=True))
                cols_index.append(i)

    if viz_type == "pie":
        num_columns = list(the_df.columns)
        for i in range(len(num_columns)):
            if the_df[num_columns[i]].nunique() <= 4:
                figs.append(pie_chart(
                    the_df, num_columns[i]))
                cols_index.append(i)

    if len(cols_index) > 0:
        tabs = st.tabs(
            [str(num_columns[i]).title().replace("_", " ") for i in cols_index])

        for i in range(len(cols_index)):
            tabs[i].plotly_chart(figs[i], use_container_width=True)

    # return figs, num_columns, cols_index


def create_heat_map(the_df):
    correlation = the_df.corr(numeric_only=True)

    fig = px.imshow(
        correlation,
        template="plotly_dark",
        text_auto="0.2f",
        aspect=1,
        color_continuous_scale="greens",
        title="Correlation Heatmap of Data",
        height=650,
    )
    fig.update_traces(
        textfont={
            "size": 16,
            "family": "consolas"
        },
    )
    fig.update_layout(
        title={
            "font": {
                "size": 30,
                "family": "tahoma"
            }
        },
        hoverlabel={
            "bgcolor": "#111",
            "font_size": 15,
            "font_family": "consolas"
        }
    )
    return fig




def show_analysis():

     # Call show_analysis() function with your dataframe
    df = pd.read_csv('Employee.csv')
    #Renaming some columns
    column_name_mapping = {
            'last_evaluation': 'evaluation_score',
            'number_project': 'project_count',
            'average_monthly_hours': 'monthly_hours',
            'work_accident': 'had_accident',
            'promotion_last_5years': 'had_promotion',
            'salary': 'salary_level'
        }

    df.rename(columns=column_name_mapping, inplace=True)

    df['work_intensity'] = df['project_count'] * df['monthly_hours']
    df['overtime'] = (df['monthly_hours'] > 174).astype(int)
    df['work_life_balance'] = df['satisfaction_level'] / (df['monthly_hours'] * df['tenure'])

    st.write(" ")
    st.markdown("<h2 style='text-align: center;'>Employee Attrition Analysis 📉</h2>", unsafe_allow_html=True)  
    st.write(" ")
    st.write(" ")
    

    st.subheader("Data Summary Overview 🧐")
    
    
    # Tabbed interface
    tabs = st.tabs(["Dataset Information", "Tabular Data", "Features", "Statistical Summary of Data","Categorical Data Summary"])

    # Dataset Information
    with tabs[0]:
        st.header("Dataset Information")
        # Define the data as a dictionary
        data = {
    "Column Name": [
        "satisfaction_level",
        "last_evaluation",
        "number_project",
        "average_monthly_hours",
        "time_spend_company",
        "work_accident",
        "left",
        "promotion_last_5years",
        "department",
        "salary"
    ],
    "Description": [
        "Employee's reported job satisfaction level, ranging from 0 to 1.",
        "The score from the employee's last performance review, ranging from 0 to 1.",
        "The total number of projects an employee is involved in.",
        "Average number of hours worked by the employee per month.",
        "The number of years the employee has spent with the company.",
        "Indicates if the employee experienced a work-related accident (0 = No, 1 = Yes).",
        "Indicates whether the employee left the company (0 = No, 1 = Yes).",
        "Shows if the employee was promoted in the last five years (0 = No, 1 = Yes).",
        "The department in which the employee works.",
        "The employee's salary level (low, medium, high)."
    ]
}

# Create a DataFrame
        dfsample = pd.DataFrame(data)

        st.write('''The dataset used for this analysis focuses on employee retention and contains key variables related to job performance, work environment, and employee satisfaction. Here’s a brief overview of the variables:''')


# Display the DataFrame as a table
        st.table(dfsample)

    
        
    # Tabular Data
    with tabs[1]:
        st.subheader('Top 10 records :')
        st.table(df.head(10))

    # Features (Column names)
    with tabs[2]:
        st.write("Number of rows:", df.shape[0])
        st.write("Number of columns:", df.shape[1])
        st.write("Columns and data types:")
        st.write(df.dtypes)

    # Statistical Summary of Data
    with tabs[3]:
        st.header("Statistical Summary of Data")
        st.write(df.describe())

    len_numerical_data = df.select_dtypes( include="number").shape[1]
    len_string_data = df.select_dtypes(include="object").shape[1]

    with tabs[4]:   
        if len_string_data > 0:
            st.subheader("String Data [𝓗]")

        data_stats = df.select_dtypes(
            include="object").describe().T
        st.table(data_stats)

    st.write('')

    st.subheader("Model Evaluation 📉🚀")
    st.write("Choose a classifier and evaluate its performance.")
    
    Classifier = st.selectbox("Select Classifier", ("Logistic Regression", "Random Forest", "Decision Tree", "Support Vector Machine (SVM)"))
    
    # Logistic Regression
    if Classifier == 'Logistic Regression':
        lr = load_model('logistic_regression.pkl')
        y_pred = lr.predict(X_test)
        pred_prob = lr.predict_proba(X_test)
        
        st.write("Logistic Regression :")
        st.write("Accuracy Score:", accuracy_score(Y_test, y_pred).round(2))
        st.write("Precision: ", precision_score(Y_test, y_pred).round(2))
        st.write("Recall: ", recall_score(Y_test, y_pred).round(2))
        
        metrics = st.multiselect("Select Metrics :", ('Confusion Matrix', 'ROC Curve'))
        if 'Confusion Matrix' in metrics:
            plot_confusion_matrix(Y_test, y_pred, "Logistic Regression")
        if 'ROC Curve' in metrics:
            plot_roc_auc(Y_test, pred_prob, "Logistic Regression")

    # Random Forest
    if Classifier == 'Random Forest':
        rf = load_model('random_forest.pkl')
        y_pred = rf.predict(X_test)
        pred_prob = rf.predict_proba(X_test)
        
        st.write("Random Forest :")
        st.write("Accuracy Score:", accuracy_score(Y_test, y_pred).round(2))
        st.write("Precision: ", precision_score(Y_test, y_pred).round(2))
        st.write("Recall: ", recall_score(Y_test, y_pred).round(2))
        
        metrics = st.multiselect("Select Metrics :", ('Confusion Matrix', 'ROC Curve'))
        if 'Confusion Matrix' in metrics:
            plot_confusion_matrix(Y_test, y_pred, "Random Forest")
        if 'ROC Curve' in metrics:
            plot_roc_auc(Y_test, pred_prob, "Random Forest")

    # Decision Tree
    if Classifier == 'Decision Tree':
        dt = load_model('decision_tree.pkl')
        y_pred = dt.predict(X_test)
        pred_prob = dt.predict_proba(X_test)
        
        st.write("Decision Tree :")
        st.write("Accuracy Score:", accuracy_score(Y_test, y_pred).round(2))
        st.write("Precision: ", precision_score(Y_test, y_pred).round(2))
        st.write("Recall: ", recall_score(Y_test, y_pred).round(2))
        
        metrics = st.multiselect("Select Metrics :", ('Confusion Matrix', 'ROC Curve'))
        if 'Confusion Matrix' in metrics:
            plot_confusion_matrix(Y_test, y_pred, "Decision Tree")
        if 'ROC Curve' in metrics:
            plot_roc_auc(Y_test, pred_prob, "Decision Tree")

    if Classifier == 'Support Vector Machine (SVM)':
        svm = load_model('svm_model (1).pkl')
        y_pred = svm.predict(X_test)

        st.write("Support Vector Machine (SVM) :")
        st.write("Accuracy Score:", accuracy_score(Y_test, y_pred).round(2))
        st.write("Precision: ", precision_score(Y_test, y_pred).round(2))
        st.write("Recall: ", recall_score(Y_test, y_pred).round(2))
        
        metrics = st.multiselect("Select Metrics :", ('Confusion Matrix', 'ROC Curve'))
        
        # Initialize pred_prob to None
        pred_prob = None

        # Plot Confusion Matrix
        if 'Confusion Matrix' in metrics:
            plot_confusion_matrix(Y_test, y_pred, "Support Vector Machine")
        
        # Check if the model has probability estimates enabled
        if 'ROC Curve' in metrics:
            try:
                pred_prob = svm.predict_proba(X_test)
                plot_roc_auc(Y_test, pred_prob, "Support Vector Machine")
            except AttributeError:
                st.write("ROC Curve not available as probability=True was not set during model training.")
            except Exception as e:
                st.write(f"An error occurred: {e}")

    