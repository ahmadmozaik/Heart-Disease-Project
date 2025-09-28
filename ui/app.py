import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# --- LOAD THE SAVED MODEL, SCALER, AND FINAL COLUMNS ---
try:
    current_dir = Path(__file__).parent
    
    model_path = current_dir.parent / "models" / "final_model.pkl"
    model = joblib.load(model_path)
    
    scaler_path = current_dir.parent / "models" / "scaler.pkl"
    scaler = joblib.load(scaler_path)
    
    final_cols_path = current_dir.parent / "data" / "final_feature_dataset.csv"
    final_model_columns = pd.read_csv(final_cols_path).drop('target', axis=1).columns.tolist()

except Exception as e:
    st.error(f"Error loading necessary files: {e}")
    st.stop()


# --- APP TITLE AND DESCRIPTION ---
st.title('❤️ Heart Disease Prediction App')
st.write("This app predicts whether a patient has heart disease based on their medical data. Please enter the patient's information in the sidebar.")


# --- SIDEBAR FOR USER INPUT ---
st.sidebar.header('Patient Data Input')

def user_input_features():
    age = st.sidebar.slider('Age', 29, 77, 60)
    sex = st.sidebar.selectbox('Sex', ('Male', 'Female'), index=0)
    cp = st.sidebar.selectbox('Chest Pain Type (CP)', (0, 1, 2, 3), index=2)
    trestbps = st.sidebar.slider('Resting Blood Pressure (trestbps)', 94, 200, 150)
    chol = st.sidebar.slider('Serum Cholestoral in mg/dl (chol)', 126, 564, 250)
    fbs = st.sidebar.selectbox('Fasting Blood Sugar > 120 mg/dl (fbs)', (0, 1))
    restecg = st.sidebar.selectbox('Resting Electrocardiographic Results (restecg)', (0, 1, 2))
    thalach = st.sidebar.slider('Maximum Heart Rate Achieved (thalach)', 71, 202, 120)
    exang = st.sidebar.selectbox('Exercise Induced Angina (exang)', (0, 1), index=1)
    oldpeak = st.sidebar.slider('ST depression induced by exercise (oldpeak)', 0.0, 6.2, 2.5)
    slope = st.sidebar.selectbox('Slope of the peak exercise ST segment (slope)', (0, 1, 2), index=2)
    ca = st.sidebar.selectbox('Number of major vessels colored by flourosopy (ca)', (0, 1, 2, 3, 4), index=3)
    thal = st.sidebar.selectbox('Thalassemia (thal)', (0, 1, 2, 3), index=2)

    sex_val = 1 if sex == 'Male' else 0

    raw_data = {
        'age': age, 'sex': sex_val, 'cp': cp, 'trestbps': trestbps, 'chol': chol,
        'fbs': fbs, 'restecg': restecg, 'thalach': thalach, 'exang': exang,
        'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal
    }
    
    input_df = pd.DataFrame(raw_data, index=[0])

    numerical_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])
    
    categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
    input_df = pd.get_dummies(input_df, columns=categorical_cols)
    
    input_df = input_df.reindex(columns=final_model_columns, fill_value=0)
    
    return input_df

# --- GET AND DISPLAY USER INPUT ---
input_df = user_input_features()

st.subheader('Patient Input Data (Processed for Model)')
st.write(input_df)

# --- PREDICTION ---
if st.button('**Predict**', type="primary"):
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)

    st.subheader('Prediction Result')
    if prediction[0] == 1:
        st.error("**Result: The model predicts that the patient HAS heart disease.**")
    else:
        st.success("**Result: The model predicts that the patient DOES NOT have heart disease.**")

    st.subheader('Prediction Probability')
    prob_df = pd.DataFrame({
        'Probability of No Disease': [f"{prediction_proba[0][0]:.2f}"],
        'Probability of Disease': [f"{prediction_proba[0][1]:.2f}"]
    })
    st.table(prob_df)