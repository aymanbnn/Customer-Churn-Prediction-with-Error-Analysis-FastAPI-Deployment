import os
import random
import numpy as np
from sklearn.discriminant_analysis import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_recall_curve, precision_recall_curve, precision_score, recall_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
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

mlflow.set_experiment("churn_prediction_skfold")

mlflow.sklearn.autolog(registered_model_name="LogisticRegressionChurnModel")
mlflow.xgboost.autolog(registered_model_name="XGBoostChurnModel")

def run_cross_validation(model, X, y):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    scores = cross_validate(
        model,
        X,
        y,
        cv=skf,
        scoring=["f1", "precision", "recall", "roc_auc"],
        return_train_score=False
    )

    return {k: np.mean(v) for k, v in scores.items() if "test" in k}

def run_cv_with_threshold(model, X, y):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    f1_scores, precisions, recalls, aucs = [], [], [], []
    thresholds = []
    best_iterations = []

    for train_idx, val_idx in skf.split(X, y):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model_clone = model.__class__(**model.get_params())

        model_clone.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            verbose=False
        )

        y_proba = model_clone.predict_proba(X_val)[:, 1]

        best_t, _, _, _ = find_best_threshold(y_val, y_proba)

        y_pred = (y_proba >= best_t).astype(int)

        f1_scores.append(f1_score(y_val, y_pred))
        precisions.append(precision_score(y_val, y_pred))
        recalls.append(recall_score(y_val, y_pred))
        aucs.append(roc_auc_score(y_val, y_proba))
        thresholds.append(best_t)
        best_iterations.append(model_clone.best_iteration)

    return {
        "f1_mean": np.mean(f1_scores),
        "f1_std": np.std(f1_scores),
        "precision_mean": np.mean(precisions),
        "precision_std": np.std(precisions),
        "recall_mean": np.mean(recalls),
        "recall_std": np.std(recalls),
        "roc_auc_mean": np.mean(aucs),
        "roc_auc_std": np.std(aucs),
        "threshold_mean": np.mean(thresholds),
        "threshold_std": np.std(thresholds),
        "best_n_estimators": int(np.mean(best_iterations))
    }

def find_best_threshold(y_true, y_proba):
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)

    f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-8)

    best_idx = np.argmax(f1_scores)
    return thresholds[best_idx], f1_scores[best_idx], precision[best_idx], recall[best_idx]


with mlflow.start_run(run_name="logistic_regression"):
    lr_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000, random_state=seed))
    ])

    cv_metrics = run_cross_validation(lr_pipeline, X_train, y_train)

    for k, v in cv_metrics.items():
        mlflow.log_metric(k, v)

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
        scale_pos_weight=len(y_train[y_train == 0]) / len(y_train[y_train == 1]),
        random_state=seed,
        eval_metric=["logloss", "auc"],
        early_stopping_rounds=20
    )

    cv_metrics = run_cv_with_threshold(model, X_train, y_train)

    for k, v in cv_metrics.items():
        mlflow.log_metric(k, v)
    
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train, random_state=seed
    )
    
    model.fit(
        X_tr, y_tr, 
        eval_set=[(X_val, y_val)], 
        verbose=False
    )

    y_proba = model.predict_proba(X_test)[:, 1]

    y_pred = (y_proba >= cv_metrics["threshold_mean"]).astype(int)
    
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

    # Error analysis: false negatives
    fn = X_test[(y_test == 1) & (y_pred == 0)]
    mlflow.log_text(fn.to_csv(index=False), "false_negatives_xgb.csv")

    print("=== XGBoost ===")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
mlflow.end_run()