import os
import random
import numpy as np
from sklearn.discriminant_analysis import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_recall_curve, precision_recall_curve, precision_score, recall_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from data import load_data
import mlflow

seed = 42
random.seed(seed)
np.random.seed(seed)

X_train, X_test, y_train, y_test = load_data()

os.makedirs("logs/mlruns", exist_ok=True)
mlflow.set_tracking_uri("./logs/mlruns")

mlflow.set_experiment("churn_prediction_autolog")

mlflow.sklearn.autolog(registered_model_name="LogisticRegressionChurnModel")
mlflow.xgboost.autolog(registered_model_name="XGBoostChurnModel")

with mlflow.start_run(run_name="logistic_regression"):
    lr_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000, random_state=seed))
    ])

    lr_pipeline.fit(X_train, y_train)
    y_pred_lr = lr_pipeline.predict(X_test)

    # Logistic Regression false negatives
    fn_lr = X_test[(y_test == 1) & (y_pred_lr == 0)]
    mlflow.log_text(fn_lr.to_csv(index=False), "false_negatives_lr.csv")
 
    print("=== Logistic Regression ===")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_lr))
    print("\nClassification Report:\n", classification_report(y_test, y_pred_lr))
mlflow.end_run()


with mlflow.start_run(run_name="xgboost_churn"):
    model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=len(y_train[y_train == 0]) / len(y_train[y_train == 1]),  # Handle class imbalance
        random_state=seed,
        eval_metric=["logloss", "auc"],
        early_stopping_rounds=20
    )
    
    model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], verbose=False)

    y_proba = model.predict_proba(X_test)[:, 1]

    # compute curve
    precision, recall, thresholds = precision_recall_curve(y_test, y_proba)

    # compute F1 for each threshold (align indices: thresholds has n-1 elements)
    f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-8)

    # find best threshold
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]

    print("Best threshold:", best_threshold)
    print("Best F1:", f1_scores[best_idx])
    print("Precision:", precision[best_idx])
    print("Recall:", recall[best_idx])
    y_pred = (y_proba >= best_threshold).astype(int)
    
    # Metrics
    f1_xgb = f1_score(y_test, y_pred)
    precision_xgb = precision_score(y_test, y_pred)
    recall_xgb = recall_score(y_test, y_pred)
    accuracy_xgb = accuracy_score(y_test, y_pred)
    roc_auc_xgb = roc_auc_score(y_test, y_proba)
    mlflow.log_metric("f1_score", float(f1_xgb))
    mlflow.log_metric("precision", float(precision_xgb))
    mlflow.log_metric("recall", float(recall_xgb))
    mlflow.log_metric("accuracy", float(accuracy_xgb))
    mlflow.log_metric("roc_auc", float(roc_auc_xgb))
    mlflow.log_metric("best_threshold", float(best_threshold))
    mlflow.log_metric("best_f1", float(f1_xgb))

    # Error analysis: false negatives
    fn = X_test[(y_test == 1) & (y_pred == 0)]
    mlflow.log_text(fn.to_csv(index=False), "false_negatives_xgb.csv")

    print("=== XGBoost ===")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
mlflow.end_run()