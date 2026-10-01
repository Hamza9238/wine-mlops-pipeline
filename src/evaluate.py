"""
evaluate.py - Champion model evaluation on the held-out test set.

Loads WineClassifier@champion from the MLflow Model Registry,
runs inference on the test split, and prints final metrics.
"""

import mlflow
import mlflow.sklearn
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    classification_report,
)

from src.data import load_data, validate_data, split_data

REGISTERED_MODEL_NAME = "WineClassifier"


def evaluate_champion():
    """
    Load the champion model and evaluate it on the test set.

    Returns
    -------
    metrics : dict  with keys accuracy, macro_f1, log_loss
    """
    # -- load champion -----------------------------------------──
    model_uri = f"models:/{REGISTERED_MODEL_NAME}@champion"
    print(f"Loading model from: {model_uri}")
    model = mlflow.sklearn.load_model(model_uri)

    # -- prepare test data -------------------------------------
    X, y = load_data()
    validate_data(X, y)
    _, X_test, _, y_test = split_data(X, y)

    # -- inference ---------------------------------------------────
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    # -- metrics -----------------------------------------------────
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")
    ll = log_loss(y_test, y_proba)

    metrics = {
        "accuracy": round(acc, 4),
        "macro_f1": round(f1, 4),
        "log_loss": round(ll, 4),
    }

    print("\n-- Champion Model - Test-Set Evaluation --\n")
    print(f"  Accuracy   : {acc:.4f}")
    print(f"  Macro F1   : {f1:.4f}")
    print(f"  Log Loss   : {ll:.4f}")
    print("\n-- Classification Report --\n")
    print(classification_report(y_test, y_pred,
                                target_names=["class_0",
                                              "class_1",
                                              "class_2"]))

    return metrics


if __name__ == "__main__":
    evaluate_champion()
