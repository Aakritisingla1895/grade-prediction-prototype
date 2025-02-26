import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings("ignore")
# Load models and test datasets
subjects = ["English", "Physics", "Mathematics", "Computer Science"]
file_paths = {
    "English": "Modified_English_dataset.csv",
    "Physics": "Modified_Physics_dataset.csv",
    "Mathematics": "Modified_Maths_dataset.csv",
    "Computer Science": "Modified_Computer_Science_dataset.csv"
}
models = {}
test_data = {}

for subject in subjects:
    try:
        models[subject] = joblib.load(f"{subject}_updated_random_forest_model.pkl")
        df = pd.read_csv(file_paths[subject])

        # Define feature columns
        feature_columns = [
            "student_id", "Average_Grade", "Max_Score", "G3", "G2", "G1", "studytime", 
            "Medu", "going_out", "traveltime", "activities"
        ]
        target_column = "G4"
        df = df[feature_columns + [target_column]]
        
        # Split the data (20% test set)
        _, X_test, _, y_test = train_test_split(df[feature_columns], df[target_column], test_size=0.2, random_state=42)
        X_test["G4"] = y_test.values
        
        # Store test data
        test_data[subject] = X_test
    except FileNotFoundError:
        st.error(f"Model file for {subject} not found!")

# Default values for inputs
default_values = {
    "Average_Grade": 10.0, "Max_Score": 100.0, "G3": 10.0, "G2": 18.0, "G1": 16.0,
    "studytime": 2.0, "Medu": 3.0, "going_out": 2.0, "traveltime": 1.0, "activities": 1.0
}

st.set_page_config(layout="wide")
st.title("📊 G4 Score Predictor")

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(subjects)

for i, subject in enumerate(subjects):
    with [tab1, tab2, tab3, tab4][i]:
        st.subheader(f"🎯 Predict G4 Score for {subject}")
        user_input = {}
        col1, col2 = st.columns(2)
        
        for idx, feature in enumerate(default_values):
            with (col1 if idx % 2 == 0 else col2):
                user_input[feature] = st.number_input(
                    feature, min_value=0.0, format="%.2f", value=default_values[feature],
                    key=f"{subject}_{feature}"
                )
        
        predict_button = st.button(f"🚀 Predict G4 for {subject}", key=f"predict_{subject}")
        if predict_button:
            model = models.get(subject)
            if model:
                input_array = np.array([list(user_input.values())]).reshape(1, -1)
                prediction = model.predict(input_array)[0]
                
                # Show Model Performance (R² Score)
                X_test = test_data[subject].drop(columns=["G4", "student_id"])
                y_test = test_data[subject]["G4"]
                y_pred = model.predict(X_test)
                r2 = r2_score(y_test, y_pred)
                
                st.success(f"🎯 **Predicted G4 score for {subject}: {prediction:.2f}**")
                st.metric(label="📈 Model Accuracy (R² Score)", value=f"{r2*100:.2f}%")
        
        # Test Data Form
        st.subheader("📝 Test Data Prediction")
        if subject in test_data:
            X_test = test_data[subject]
            test_ids = X_test["student_id"].tolist()
            selected_id = st.selectbox("Select Test Data ID", test_ids, key=f"{subject}_test_id")
            
            test_row = X_test[X_test["student_id"] == selected_id].iloc[0]
            test_col1, test_col2 = st.columns(2)
            
            test_user_input = {}
            for idx, feature in enumerate(default_values):
                with (test_col1 if idx % 2 == 0 else test_col2):
                    test_user_input[feature] = st.number_input(
                        feature, min_value=0.0, format="%.2f", value=float(test_row[feature]),
                        key=f"{subject}_test_{feature}"
                    )
            
            test_predict_button = st.button(f"📌 Predict G4 for Test Data - {subject}", key=f"test_predict_{subject}")
            if test_predict_button:
                model = models.get(subject)
                if model:
                    test_input_array = np.array([list(test_user_input.values())]).reshape(1, -1)
                    test_prediction = model.predict(test_input_array)[0]
                    
                    # Show Model Performance (R² Score)
                    r2 = r2_score(y_test, model.predict(X_test.drop(columns=["G4", "student_id"])))
                    
                    st.success(
                        f"🔍 **Predicted G4 score for test data in {subject}: {test_prediction:.2f}**\n\n"
                        f"🎓 **Student ID:** {selected_id}\n\n"
                        f"✅ **Actual G4 Score:** {test_row['G4']:.2f}"
                    )
                    st.metric(label="📈 Model Accuracy (R² Score)", value=f"{r2*100:.2f}%")
