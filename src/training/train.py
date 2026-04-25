import os

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
)

FEATURES = ["temperature", "humidity", "noise_level"]
TARGET = "failure"
MODEL_PATH = "model/wind_turbine_model.joblib"
TRAIN_PATH = "data/cleaned/train.csv"
TEST_PATH  = "data/cleaned/test.csv"


def train_and_save() -> None:
    train_df = pd.read_csv(TRAIN_PATH)
    test_df  = pd.read_csv(TEST_PATH)

    X_train, y_train = train_df[FEATURES], train_df[TARGET]
    X_test,  y_test  = test_df[FEATURES],  test_df[TARGET]

    # Random Forest chosen because:
    # - Score-based labeling creates non-linear decision boundaries trees handle naturally
    # - predict_proba gives calibrated failure probabilities without post-processing
    # - class_weight='balanced' corrects the ~19/81 class split without resampling
    # - Feature importance supports future interpretability work
    model = RandomForestClassifier(class_weight="balanced", random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print(f"  Accuracy  : {accuracy_score(y_test, y_pred):.4f}")
    print(f"  Precision : {precision_score(y_test, y_pred):.4f}")
    print(f"  Recall    : {recall_score(y_test, y_pred):.4f}  <- primary")
    print(f"  F1        : {f1_score(y_test, y_pred):.4f}")
    print(f"  ROC-AUC   : {roc_auc_score(y_test, y_prob):.4f}  <- primary")

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"  Model saved  {MODEL_PATH}")
