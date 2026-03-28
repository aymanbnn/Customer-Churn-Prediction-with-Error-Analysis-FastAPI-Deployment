from fastapi import FastAPI
import pickle

app = FastAPI()

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

@app.post("/predict")
def predict(features: list):
    """
    Expects a list of feature values in the same order as training.
    Returns 0 (no churn) or 1 (churn).
    """
    prediction = model.predict([features])
    return {"prediction": int(prediction[0])}