"""
trains and compares three classifiers on the fraud detection task:
logistic regression, random forest, and XGBoost
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
)
from xgboost import XGBClassifier

RANDOM_STATE = 313


def load_features():
    df = pd.read_csv("data/creditcard.csv")
    X = df.drop(columns=["Class"])
    y = df["Class"]
    return X, y


def evaluate(name, y_true, y_pred, y_proba):
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_proba)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    print(f"\n--- {name} ---")
    print(f"Precision: {precision:.3f}  (of transactions flagged as fraud, how many actually were)")
    print(f"Recall:    {recall:.3f}  (of actual fraud cases, how many were caught)")
    print(f"F1:        {f1:.3f}")
    print(f"ROC-AUC:   {auc:.3f}")
    print(f"Confusion matrix -> TN: {tn}  FP: {fp}  FN: {fn}  TP: {tp}")
    return {"model": name, "precision": precision, "recall": recall, "f1": f1, "auc": auc,
            "false_negatives": fn, "false_positives": fp}


if __name__ == "__main__":
    X, y = load_features()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = []

    log_model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE)
    log_model.fit(X_train_scaled, y_train)
    log_pred = log_model.predict(X_test_scaled)
    log_proba = log_model.predict_proba(X_test_scaled)[:, 1]
    results.append(evaluate("Logistic Regression", y_test, log_pred, log_proba))

    # Random forest
    rf_model = RandomForestClassifier(
        n_estimators=100, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_proba = rf_model.predict_proba(X_test)[:, 1]
    results.append(evaluate("Random Forest", y_test, rf_pred, rf_proba))

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    xgb_model = XGBClassifier(
        n_estimators=100, scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE, eval_metric="logloss"
    )
    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict(X_test)
    xgb_proba = xgb_model.predict_proba(X_test)[:, 1]
    results.append(evaluate("XGBoost", y_test, xgb_pred, xgb_proba))

    results_df = pd.DataFrame(results)
    results_df.to_csv("data/model_comparison.csv", index=False)
    print("\n\n=== Summary ===")
    print(results_df.to_string(index=False))
