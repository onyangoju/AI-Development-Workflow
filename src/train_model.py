import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os

# Ensure models/ folder exists
os.makedirs("models", exist_ok=True)

# Load data
df = pd.read_csv("data/patients.csv")

# Preprocess data
df = df.dropna(subset=['readmit'])
df['age'] = df['age'].fillna(df['age'].median())
df['bp'] = df['bp'].fillna(df['bp'].mean())
df = pd.get_dummies(df, columns=['gender', 'clinic', 'diagnosis'], drop_first=True)

# Split features and target
X = df.drop(columns=['patient_id', 'readmit'])
y = df['readmit']

# Normalize numeric features
scaler = StandardScaler()
X[['age', 'bp', 'prev_visits']] = scaler.fit_transform(X[['age', 'bp', 'prev_visits']])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

# Save model and scaler
joblib.dump(model, "models/model.joblib")
joblib.dump(scaler, "models/scaler.joblib")

print("✅ Model trained and saved to models/model.joblib")
