"""
train_model.py
---------------
Standalone training script for the Heart Disease Prediction model.
Mirrors the logic in Assignment-10.ipynb (Tasks 1 & 2), so it can be
run directly from the command line to (re)produce model.pkl.

Usage:
    python train_model.py

Expects heart.csv to be present in the same directory (see README.md
for how to download it via kagglehub or the Kaggle CLI).
"""

import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

SEED = 42
DATA_PATH = "heart.csv"
MODEL_PATH = "model.pkl"
FEATURE_ORDER_PATH = "feature_order.json"
TARGET_COL = "target"


def main():
    # ---- Task 1: Data Understanding & Preprocessing ----
    df = pd.read_csv(DATA_PATH)
    print("Dataset shape:", df.shape)
    print(df.head())

    missing = df.isnull().sum().sum()
    print(f"Total missing values: {missing}")

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=SEED, stratify=y
    )

    # ---- Task 2: Model Development ----
    model = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=SEED)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy: {accuracy:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=["No Disease", "Disease"]))

    # Save the trained model
    joblib.dump(model, MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")

    # Save the exact feature column order — app.py needs this to build
    # correctly-ordered rows from incoming JSON payloads.
    with open(FEATURE_ORDER_PATH, "w") as f:
        json.dump(list(X.columns), f)
    print(f"Feature order saved to {FEATURE_ORDER_PATH}")


if __name__ == "__main__":
    main()
