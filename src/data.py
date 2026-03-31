import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def engineer_features(df):
    service_cols = [
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
        'TechSupport', 'StreamingTV', 'StreamingMovies'
    ]

    df['service_count'] = (df[service_cols] == 'Yes').sum(axis=1)

    df['tenure_group'] = pd.cut(
        df['tenure'],
        bins=[0, 6, 12, 24, 73],
        labels=['new_0-6m', 'engaged_6-12m', 'loyal_1-2y', 'very_loyal_2y+']
    )

    df['family_size'] = (
        (df['Partner'] == 'Yes').astype(int) +
        (df['Dependents'] == 'Yes').astype(int) + 1
    )

    df['alone_senior'] = (
        (df['SeniorCitizen'] == 1) &
        (df['Partner'] == 'No') &
        (df['Dependents'] == 'No')
    ).astype(int)

    df['avg_monthly_charge'] = df['TotalCharges'] / (df['tenure'] + 1)
    df['monthly_charge_ratio'] = df['MonthlyCharges'] / (df['avg_monthly_charge'] + 1)

    df['at_risk'] = (
        (df['tenure'] < 12) &
        (df['Contract'] == 'Month-to-month')
    ).astype(int)

    df['risky_payment'] = (df['PaymentMethod'] == 'Electronic check').astype(int)
    df['month_to_month'] = (df['Contract'] == 'Month-to-month').astype(int)

    return df

def load_data(path="data/churn.csv"):
    df = pd.read_csv(path)

    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    df = engineer_features(df)

    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    df = pd.get_dummies(df, drop_first=True)

    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    return train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )