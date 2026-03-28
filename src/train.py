import os
import random
import numpy as np
import pickle
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from data import load_data

seed = 42
random.seed(seed)
np.random.seed(seed)

X_train, X_test, y_train, y_test = load_data()

lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

print("=== Logistic Regression ===")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_lr))
print("\nClassification Report:\n", classification_report(y_test, y_pred_lr))

model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    random_state=seed,
    eval_metric="logloss"
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("=== XGBoost ===")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

fn = X_test[(y_test == 1) & (y_pred == 0)]
print("\nSample False Negatives:\n", fn.head())

os.makedirs("../models", exist_ok=True)
with open("../models/model.pkl", "wb") as f:
    pickle.dump(model, f)