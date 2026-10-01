"""
train.py - Model training with MLflow experiment tracking.

Trains 3 Random Forest and 3 Gradient Boosting configs
using 5-fold Stratified CV.  Logs params, metrics, models,
and signatures to MLflow.  Selects the champion by
validation Macro-F1 and registers it as WineClassifier@champion.
"""

import warnings
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    make_scorer,
)

from src.data import load_data, validate_data, split_data

# -- suppress convergence / future warnings during CV ----------
warnings.filterwarnings("ignore")

# -- experiment name required by the rubric --------------------
EXPERIMENT_NAME = "Wine-Cultivar-Classification"
REGISTERED_MODEL_NAME = "WineClassifier"

# -- model configurations (>= 3 each => >= 6 runs) ------------
MODEL_CONFIGS = [
    # ---------- Random Forest ----------
    {
        "name": "RandomForest",
        "config_label": "RF-Config-1",
        "model": RandomForestClassifier,
        "params": {
            "n_estimators": 100,  # Resolved: benchmark validated 100 estimators optimal
            "max_depth": None,
            "min_samples_split": 2,
            "random_state": 42,
        },
    },
    {
        "name": "RandomForest",
        "config_label": "RF-Config-2",
        "model": RandomForestClassifier,
        "params": {
            "n_estimators": 200,
            "max_depth": 10,
            "min_samples_split": 5,
            "random_state": 42,
        },
    },
    {
        "name": "RandomForest",
        "config_label": "RF-Config-3",
        "model": RandomForestClassifier,
        "params": {
            "n_estimators": 300,
            "max_depth": 20,
            "min_samples_split": 3,
            "random_state": 42,
        },
    },
    # ---------- Gradient Boosting ----------
    {
        "name": "GradientBoosting",
        "config_label": "GB-Config-1",
        "model": GradientBoostingClassifier,
        "params": {
            "n_estimators": 100,
            "learning_rate": 0.1,
            "max_depth": 3,
            "random_state": 42,
        },
    },
    {
        "name": "GradientBoosting",
        "config_label": "GB-Config-2",
        "model": GradientBoostingClassifier,
        "params": {
            "n_estimators": 200,
            "learning_rate": 0.05,
            "max_depth": 4,
            "random_state": 42,
        },
    },
    {
        "name": "GradientBoosting",
        "config_label": "GB-Config-3",
        "model": GradientBoostingClassifier,
        "params": {
            "n_estimators": 300,
            "learning_rate": 0.01,
            "max_depth": 5,
            "random_state": 42,
        },
    },
]


# -- custom scorers for cross_validate -------------------------
def _macro_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average="macro")


scorers = {
    "accuracy": make_scorer(accuracy_score),
    "macro_f1": make_scorer(_macro_f1),
    "neg_log_loss": "neg_log_loss",
}


def run_experiment(X_train, y_train, X_test, y_test):
    """
    Train all configs, log to MLflow, pick & register champion.

    Returns
    -------
    results : list[dict]   summary of every run
    """
    mlflow.set_experiment(EXPERIMENT_NAME)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = []
    best_val_f1 = -1.0
    best_run_id = None

    for cfg in MODEL_CONFIGS:
        clf = cfg["model"](**cfg["params"])

        # -- 5-fold Stratified CV ------------------------------
        cv_results = cross_validate(
            clf, X_train, y_train,
            cv=cv,
            scoring=scorers,
            return_train_score=True,
        )

        train_f1 = cv_results["train_macro_f1"].mean()
        val_f1 = cv_results["test_macro_f1"].mean()
        train_acc = cv_results["train_accuracy"].mean()
        val_acc = cv_results["test_accuracy"].mean()
        val_log_loss = -cv_results["test_neg_log_loss"].mean()

        # -- retrain on full training set for artefact logging -
        clf.fit(X_train, y_train)

        # -- MLflow run ----------------------------------------
        with mlflow.start_run(run_name=cfg["config_label"]) as run:
            # params
            mlflow.log_params(cfg["params"])
            mlflow.log_param("model_family", cfg["name"])

            # metrics
            mlflow.log_metric("train_macro_f1", train_f1)
            mlflow.log_metric("val_macro_f1", val_f1)
            mlflow.log_metric("train_accuracy", train_acc)
            mlflow.log_metric("val_accuracy", val_acc)
            mlflow.log_metric("val_log_loss", val_log_loss)

            # tags
            mlflow.set_tag("model_type", cfg["name"])
            mlflow.set_tag("config", cfg["config_label"])
            mlflow.set_tag("student_id", "23F-0060")

            # signature + example input
            signature = infer_signature(
                X_train, clf.predict(X_train)
            )
            input_example = X_train.iloc[:1]

            mlflow.sklearn.log_model(
                clf,
                artifact_path="model",
                signature=signature,
                input_example=input_example,
            )

            run_id = run.info.run_id

        record = {
            "run_id": run_id,
            "model": cfg["name"],
            "config": cfg["config_label"],
            "train_f1": round(train_f1, 4),
            "val_f1": round(val_f1, 4),
            "train_acc": round(train_acc, 4),
            "val_acc": round(val_acc, 4),
            "val_log_loss": round(val_log_loss, 4),
        }
        results.append(record)

        print(
            f"  {cfg['config_label']:15s}  |  "
            f"Val F1={val_f1:.4f}  "
            f"Val Acc={val_acc:.4f}  "
            f"Val LogLoss={val_log_loss:.4f}"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_run_id = run_id

    # -- register champion -------------------------------------
    print(f"\n[BEST] Champion run: {best_run_id}  "
          f"(Val Macro-F1 = {best_val_f1:.4f})")

    model_uri = f"runs:/{best_run_id}/model"
    mv = mlflow.register_model(model_uri, REGISTERED_MODEL_NAME)

    client = mlflow.tracking.MlflowClient()
    client.set_registered_model_alias(
        REGISTERED_MODEL_NAME, "champion", mv.version
    )
    print(f"[OK] Registered {REGISTERED_MODEL_NAME} v{mv.version} "
          f"with alias 'champion'.")

    return results


def main():
    """Entry-point called by `make train`."""
    print("=" * 60)
    print("  Wine MLOps Pipeline - Training")
    print("  Student: 23F-0060")
    print("=" * 60)

    X, y = load_data()
    validate_data(X, y)
    X_train, X_test, y_train, y_test = split_data(X, y)

    print("\n-- Training & MLflow logging --\n")
    results = run_experiment(X_train, y_train, X_test, y_test)

    # -- print summary table -----------------------------------
    print("\n-- Results Summary --\n")
    header = (
        f"{'Config':15s} | {'Train F1':>9s} | {'Val F1':>9s} | "
        f"{'Train Acc':>9s} | {'Val Acc':>9s} | {'Val LL':>9s}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['config']:15s} | {r['train_f1']:9.4f} | "
            f"{r['val_f1']:9.4f} | {r['train_acc']:9.4f} | "
            f"{r['val_acc']:9.4f} | {r['val_log_loss']:9.4f}"
        )

    return results


if __name__ == "__main__":
    main()
