# Wine MLOps Pipeline 🍷

**Student:** M Yousaf Rehan — **Roll Number:** 23F-0060

An end-to-end MLOps pipeline for the UCI Wine dataset using
scikit-learn, MLflow, pytest, and GitHub Actions.

---

## Project Structure

```text
wine-mlops-pipeline/
│
├── .github/workflows/ci.yml   # GitHub Actions CI
├── data/.gitkeep               # Data directory placeholder
│
├── src/
│   ├── __init__.py
│   ├── data.py                 # Load, validate & split Wine dataset
│   ├── train.py                # Train RF + GB with MLflow tracking
│   └── evaluate.py             # Evaluate champion on test set
│
├── tests/
│   ├── __init__.py
│   ├── test_data.py            # Data validation tests
│   └── test_model_gate.py      # Quality-gate tests (F1, latency, classes)
│
├── .gitignore
├── Makefile
├── requirements.txt
└── README.md
```

---

## Quick Start

```bash
# 1. Create & activate virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 2. Install dependencies
make install

# 3. Lint the code
make lint

# 4. Train all models + register champion in MLflow
make train

# 5. Evaluate champion model on held-out test set
make evaluate

# 6. Run all tests (data + quality gates)
make test

# 7. Launch MLflow UI to inspect experiments
mlflow ui
```

---

## Models & Configurations

| # | Model             | Key Hyperparameters                                   |
|---|-------------------|-------------------------------------------------------|
| 1 | Random Forest     | n_estimators=100, max_depth=None, min_samples_split=2 |
| 2 | Random Forest     | n_estimators=200, max_depth=10, min_samples_split=5   |
| 3 | Random Forest     | n_estimators=300, max_depth=20, min_samples_split=3   |
| 4 | Gradient Boosting | n_estimators=100, lr=0.10, max_depth=3                |
| 5 | Gradient Boosting | n_estimators=200, lr=0.05, max_depth=4                |
| 6 | Gradient Boosting | n_estimators=300, lr=0.01, max_depth=5                |

All configurations use **5-fold Stratified Cross-Validation** on the
training data. The champion is selected by **Validation Macro-F1** and
registered in MLflow as `WineClassifier@champion`.

---

## Quality Gates (CI)

| Gate                | Threshold |
|---------------------|-----------|
| Macro F1            | ≥ 0.88    |
| Batch inference     | ≤ 30 ms   |
| Prediction classes  | {0, 1, 2} |

---

## Dataset

[UCI Wine](https://scikit-learn.org/stable/datasets/toy_dataset.html#wine-dataset)
— 178 samples, 13 features, 3 classes. Loaded via
`sklearn.datasets.load_wine`.

---

## License

This project is submitted as coursework for the MLOps course.
