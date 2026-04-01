from fastapi import FastAPI
from api.schemas import CustomerData
from api.predictor import predict_xgb, predict_lr

app = FastAPI(title="Churn Prediction API")


@app.post("/predict/xgb")
def predict_xgb_api(data: CustomerData):
    return predict_xgb(data.features)


@app.post("/predict/lr")
def predict_lr_api(data: CustomerData):
    return predict_lr(data.features)