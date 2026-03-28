import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(path="../data/churn.csv"):
    """
    Load and preprocess Telco Customer Churn dataset.
    Returns train-test split.
    """
    df = pd.read_csv(path)

    # Drop irrelevant column
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    # Fix TotalCharges (string → numeric)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

    # Encode target
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    # One-hot encode categorical features
    df = pd.get_dummies(df, drop_first=True)

    # Split
    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    return train_test_split(X, y, test_size=0.2, random_state=42)