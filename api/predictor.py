import mlflow.xgboost
import mlflow.sklearn
import pandas as pd
from mlflow.tracking import MlflowClient

MLFLOW_TRACKING_URI = "./logs/mlruns"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Load models from registry
xgb_model = mlflow.xgboost.load_model("models:/XGBoostChurnModel/Production")
lr_model = mlflow.sklearn.load_model("models:/LogisticRegressionChurnModel/Production")

# Load threshold dynamically
client = MlflowClient()

def get_model_threshold(model_name):
    latest = client.get_latest_versions(model_name, stages=["Production"])[0]
    run_id = latest.run_id

    run = client.get_run(run_id)
    return float(run.data.params.get("decision_threshold", 0.5))


xgb_threshold = get_model_threshold("XGBoostChurnModel")


def predict_xgb(data: dict):
    df = pd.DataFrame([data])

    proba = xgb_model.predict(df)[:, 1][0]
    prediction = (proba >= xgb_threshold).astype(int)

    return {
        "model": "xgboost",
        "prediction": prediction,
        "probability": float(proba),
        "threshold": float(xgb_threshold)
    }


def predict_lr(data: dict):
    df = pd.DataFrame([data])

    prediction = lr_model.predict(df)[0]

    return {
        "model": "logistic_regression",
        "prediction": int(prediction)
    }