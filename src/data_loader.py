import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(filepath):
    df = pd.read_csv(filepath)
    return df

def preprocess_data(df):
    # Drop rows with missing target
    df = df.dropna(subset=['readmit'])

    # Fill missing values
    df['age'] = df['age'].fillna(df['age'].median())
    df['bp'] = df['bp'].fillna(df['bp'].mean())

    # Encode categorical
    df = pd.get_dummies(df, columns=['clinic', 'gender'])

    # Normalize features
    scaler = StandardScaler()
    df[['age', 'bp']] = scaler.fit_transform(df[['age', 'bp']])

    return df
