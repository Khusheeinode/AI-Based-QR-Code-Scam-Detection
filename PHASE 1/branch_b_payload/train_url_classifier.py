"""
train_url_classifier.py

Trains and compares Logistic Regression, Random Forest, and XGBoost on
URL features to classify phishing vs legitimate URLs.

Expects a CSV at data/urls.csv with columns: url, label (1 = malicious, 0 = legitimate)
Build that CSV from PhishTank (malicious) + Tranco top domains (legitimate) -
see README for the data-collection steps.

Run (from project root):
    python branch_b_payload/train_url_classifier.py
"""

import os
import sys
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier

sys.path.append(os.path.dirname(__file__))
from url_features import extract_url_features  # noqa: E402

DATA_CSV = os.path.join(os.path.dirname(__file__), "..", "data", "urls.csv")
MODEL_OUT = os.path.join(os.path.dirname(__file__), "url_model.joblib")
FEATURE_ORDER_OUT = os.path.join(os.path.dirname(__file__), "url_feature_order.joblib")


def build_feature_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = [extract_url_features(u) for u in df["url"]]
    return pd.DataFrame(rows)


def evaluate(model, X_test, y_test, name):
    preds = model.predict(X_test)
    print(f"\n--- {name} ---")
    print(f"Accuracy : {accuracy_score(y_test, preds):.4f}")
    print(f"Precision: {precision_score(y_test, preds):.4f}")
    print(f"Recall   : {recall_score(y_test, preds):.4f}")
    print(f"F1       : {f1_score(y_test, preds):.4f}")
    return f1_score(y_test, preds)


def main():
    df = pd.read_csv(DATA_CSV)  # columns: url, label
    X = build_feature_table(df)
    y = df["label"].values
    feature_order = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42),
        "XGBoost": XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.1,
            eval_metric="logloss", random_state=42,
        ),
    }

    best_model, best_name, best_f1 = None, None, -1
    for name, model in models.items():
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="f1")
        print(f"{name} 5-fold CV F1: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        model.fit(X_train, y_train)
        f1 = evaluate(model, X_test, y_test, name)
        if f1 > best_f1:
            best_model, best_name, best_f1 = model, name, f1

    print(f"\nBest model: {best_name} (F1={best_f1:.4f}) - saving.")
    joblib.dump(best_model, MODEL_OUT)
    joblib.dump(feature_order, FEATURE_ORDER_OUT)


if __name__ == "__main__":
    main()
