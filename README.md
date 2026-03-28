# Churn Prediction ML Project

## Problem

Predict customer churn to reduce retention loss.

## Models

- Logistic Regression (baseline)
- XGBoost (main)

## Results

- F1-score: X
- Key issue: false negatives

## Error Analysis

- Model struggled with minority class
- Fixed using class weighting

## Deployment

- FastAPI endpoint: /predict

## Reproducibility

- Seed = 42
