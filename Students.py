import streamlit as st
import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression

# Load the trained model
model_path = r'C:\Users\ezer2\Desktop\jupiter\Student performance\students_model.pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)


# Function to map categorical features
def map_features(data):
    encoding_mappings = {
        'Parental_Involvement_encoded': {'High': 0, 'Low': 1, 'Medium': 2},
        'Access_to_Resources_encoded': {'High': 0, 'Low': 1, 'Medium': 2},
        'Extracurricular_Activities_encoded': {'No': 0, 'Yes': 1},
        'Motivation_Level_encoded': {'High': 0, 'Low': 1, 'Medium': 2},
        'Internet_Access_encoded': {'No': 0, 'Yes': 1},
        'Family_Income_encoded': {'High': 0, 'Low': 1, 'Medium': 2},
        'Teacher_Quality_encoded': {'High': 0, 'Low': 1, 'Medium': 2},
        'School_Type_encoded': {'Private': 0, 'Public': 1},
        'Peer_Influence_encoded': {'Negative': 0, 'Neutral': 1, 'Positive': 2},
        'Learning_Disabilities_encoded': {'No': 0, 'Yes': 1},
        'Parental_Education_Level_encoded': {'College': 0, 'High School': 1, 'Postgraduate': 2},
        'Distance_from_Home_encoded': {'Far': 0, 'Moderate': 1, 'Near': 2},
        'Gender_encoded': {'Female': 0, 'Male': 1},
    }
    for col, mapping in encoding_mappings.items():
        data[col] = data[col].map(mapping)
    return data


# Function to predict Exam Score
def predict_exam_score(data):
    # Create the DataFrame and map categorical values
    df = pd.DataFrame(data)
    df = map_features(df)

    # Predict the score for the input data
    prediction = model.predict(df)

    return prediction


# Streamlit Interface
st.title("Student Exam Score Prediction")

# Input fields for the model
st.sidebar.header("Input Parameters")
Hours_Studied = st.sidebar.number_input("Hours Studied", min_value=0, max_value=24, value=5)
Attendance = st.sidebar.number_input("Attendance", min_value=0, max_value=100, value=85)
Sleep_Hours = st.sidebar.number_input("Sleep Hours", min_value=0, max_value=24, value=7)
Previous_Scores = st.sidebar.number_input("Previous Scores", min_value=0, max_value=100, value=70)
Tutoring_Sessions = st.sidebar.number_input("Tutoring Sessions", min_value=0, max_value=20, value=2)
Physical_Activity = st.sidebar.number_input("Physical Activity (hrs/week)", min_value=0, max_value=20, value=3)
Parental_Involvement_encoded = st.sidebar.selectbox("Parental Involvement", ["High", "Medium", "Low"])
Access_to_Resources_encoded = st.sidebar.selectbox("Access to Resources", ["High", "Medium", "Low"])
Extracurricular_Activities_encoded = st.sidebar.selectbox("Extracurricular Activities", ["No", "Yes"])
Motivation_Level_encoded = st.sidebar.selectbox("Motivation Level", ["High", "Medium", "Low"])
Internet_Access_encoded = st.sidebar.selectbox("Internet Access", ["No", "Yes"])
Family_Income_encoded = st.sidebar.selectbox("Family Income", ["High", "Medium", "Low"])
Teacher_Quality_encoded = st.sidebar.selectbox("Teacher Quality", ["High", "Medium", "Low"])
School_Type_encoded = st.sidebar.selectbox("School Type", ["Private", "Public"])
Peer_Influence_encoded = st.sidebar.selectbox("Peer Influence", ["Negative", "Neutral", "Positive"])
Learning_Disabilities_encoded = st.sidebar.selectbox("Learning Disabilities", ["No", "Yes"])
Parental_Education_Level_encoded = st.sidebar.selectbox("Parental Education Level",
                                                        ["College", "High School", "Postgraduate"])
Distance_from_Home_encoded = st.sidebar.selectbox("Distance from Home", ["Far", "Moderate", "Near"])
Gender_encoded = st.sidebar.selectbox("Gender", ["Female", "Male"])

# Create the input dataframe
input_data = {
    'Hours_Studied': [Hours_Studied],
    'Attendance': [Attendance],
    'Sleep_Hours': [Sleep_Hours],
    'Previous_Scores': [Previous_Scores],
    'Tutoring_Sessions': [Tutoring_Sessions],
    'Physical_Activity': [Physical_Activity],
    'Parental_Involvement_encoded': [Parental_Involvement_encoded],
    'Access_to_Resources_encoded': [Access_to_Resources_encoded],
    'Extracurricular_Activities_encoded': [Extracurricular_Activities_encoded],
    'Motivation_Level_encoded': [Motivation_Level_encoded],
    'Internet_Access_encoded': [Internet_Access_encoded],
    'Family_Income_encoded': [Family_Income_encoded],
    'Teacher_Quality_encoded': [Teacher_Quality_encoded],
    'School_Type_encoded': [School_Type_encoded],
    'Peer_Influence_encoded': [Peer_Influence_encoded],
    'Learning_Disabilities_encoded': [Learning_Disabilities_encoded],
    'Parental_Education_Level_encoded': [Parental_Education_Level_encoded],
    'Distance_from_Home_encoded': [Distance_from_Home_encoded],
    'Gender_encoded': [Gender_encoded]
}

# Display user inputs
st.subheader('User Input Parameters')
st.write(pd.DataFrame(input_data))

# Predict Exam Score
if st.button('Predict Exam Score'):
    result = predict_exam_score(input_data)
    st.subheader('Predicted Exam Score')
    st.write(result[0])
