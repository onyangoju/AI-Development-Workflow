from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# Load trained model and scaler
model = joblib.load("models/model.joblib")
scaler = joblib.load("models/scaler.joblib")

# Create FastAPI app
app = FastAPI(title="Patient Readmission Predictor")

# Define the input schema
class PatientInput(BaseModel):
    age: int
    gender: str
    bp: float
    clinic: str
    diagnosis: str
    prev_visits: int

# Preprocess incoming input
def preprocess_input(data: PatientInput):
    df = pd.DataFrame([data.dict()])

    # One-hot encode categorical variables
    df = pd.get_dummies(df, columns=['gender', 'clinic', 'diagnosis'])

    # Define expected columns from training
    expected_cols = [
        'age', 'bp', 'prev_visits',
        'gender_Male',
        'clinic_General', 'clinic_Maternity', 'clinic_Surgery',
        'diagnosis_Flu', 'diagnosis_Fracture', 'diagnosis_Infection', 'diagnosis_Pregnancy'
    ]

    # Add any missing expected columns and set them to 0
    for col in expected_cols:
        if col not in df.columns:
            df[col] = 0

    # Ensure the DataFrame has only the expected columns in the correct order
    df = df[expected_cols]

    # Normalize numeric features
    df[['age', 'bp', 'prev_visits']] = scaler.transform(df[['age', 'bp', 'prev_visits']])

    return df

# Define prediction endpoint
@app.post("/predict")
def predict_readmission(patient: PatientInput):
    X = preprocess_input(patient)
    prediction = model.predict(X)[0]
    return {"readmit_prediction": int(prediction)}
