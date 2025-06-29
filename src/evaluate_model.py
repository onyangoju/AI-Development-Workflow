import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# Load data
df = pd.read_csv("data/patients.csv")

# Preprocess
df = df.dropna(subset=['readmit'])
df['age'] = df['age'].fillna(df['age'].median())
df['bp'] = df['bp'].fillna(df['bp'].mean())
df = pd.get_dummies(df, columns=['gender', 'clinic', 'diagnosis'], drop_first=True)

# Prepare features and labels
X = df.drop(columns=['patient_id', 'readmit'])
y = df['readmit']

# Load saved scaler and model
scaler = joblib.load("models/scaler.joblib")
model = joblib.load("models/model.joblib")

# Apply same scaling
X[['age', 'bp', 'prev_visits']] = scaler.transform(X[['age', 'bp', 'prev_visits']])

# Split
_, X_test, _, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Predict
y_pred = model.predict(X_test)

# Evaluate
print("✅ Classification Report:\n", classification_report(y_test, y_pred))
print("🧮 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
