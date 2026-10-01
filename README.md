# Wine MLOps Pipeline 🍷

[![CI Pipeline](https://github.com/myousafrehan/wine-mlops-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/myousafrehan/wine-mlops-pipeline/actions)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![MLflow](https://img.shields.io/badge/MLflow-2.21.3-0194E2.svg)
![License](https://img.shields.io/badge/Coursework-FAST--NUCES-red.svg)

**Student:** M Yousaf Rehan — **Roll Number:** 23F-0060  
**Course:** Machine Learning Operations (MLOps) — Fall 2026 — Assignment 01  

An end-to-end reproducible MLOps pipeline for chemical cultivar classification on the UCI Wine dataset using scikit-learn, MLflow, pytest, and GitHub Actions.

---

## Project Structure

```text
wine-mlops-pipeline/
│
├── .github/
│   └── workflows/
│       └── ci.yml              # GitHub Actions CI workflow
├── data/
│   └── .gitkeep                # Data directory placeholder
│
├── src/
│   ├── __init__.py
│   ├── data.py                 # Load, validate & 80/20 stratified split
│   ├── train.py                # Train RF + GB with 5-fold CV & MLflow tracking
│   └── evaluate.py             # Evaluate champion model on test set
│
├── tests/
│   ├── __init__.py
│   ├── test_data.py            # Data validation & split tests (10 tests)
│   └── test_model_gate.py      # Quality-gate tests (F1, latency, classes)
│
├── .gitignore                  # Excludes mlruns, caches, virtualenvs
├── Makefile                    # Automation entry points (install, lint, test, train, evaluate, clean)
├── requirements.txt            # Pinned dependencies
├── MLOps_A01_23F-0060.pdf       # Formal submission report
└── README.md
```

---

## Quick Start & Makefile Commands

All local development and CI steps are automated via the GNU `Makefile`:

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate       # Windows PowerShell: .venv\Scripts\Activate.ps1

# 2. Upgrade pip and install exact pinned dependencies
make install

# 3. Lint code with flake8 (max-line-length = 100)
make lint

# 4. Train dual classifier families (6 runs) & register champion in MLflow
make train

# 5. Evaluate champion model on held-out test split
make evaluate

# 6. Execute all unit tests and quality gates with verbose output
make test

# 7. Clean up bytecode and cache files
make clean

# 8. Launch MLflow UI to inspect runs and model registry
mlflow ui
```

---

## Model Exploration & Search Results (Table 1)

Six candidate configurations across two tree-based classifier families were evaluated using **5-fold Stratified Cross-Validation** (`random_state=42`):

| Config Label | Model Family | Hyperparameters | Train F1 | Val Macro-F1 | Train Acc | Val Acc | Val Log Loss | Status |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **RF-Config-1** | **Random Forest** | `n=100, depth=None, split=2` | **1.0000** | **0.9789** | **1.0000** | **0.9791** | **0.1610** | **★ Champion** |
| RF-Config-2 | Random Forest | `n=200, depth=10, split=5` | 1.0000 | 0.9789 | 1.0000 | 0.9791 | 0.1746 | Candidate |
| RF-Config-3 | Random Forest | `n=300, depth=20, split=3` | 1.0000 | 0.9789 | 1.0000 | 0.9791 | 0.1688 | Candidate |
| GB-Config-1 | Gradient Boosting | `n=100, lr=0.10, depth=3` | 1.0000 | 0.9112 | 1.0000 | 0.9094 | 0.4347 | Candidate |
| GB-Config-2 | Gradient Boosting | `n=200, lr=0.05, depth=4` | 1.0000 | 0.8987 | 1.0000 | 0.8953 | 0.4877 | Candidate |
| GB-Config-3 | Gradient Boosting | `n=300, lr=0.01, depth=5` | 1.0000 | 0.9051 | 1.0000 | 0.9022 | 0.2428 | Candidate |

### Champion Model Selection & Registry Promotion
- **Champion:** `RF-Config-1` achieved the lowest validation log loss (**0.1610**) while maintaining peak validation Macro-F1 (**0.9789**).
- **Registry:** Programmatically registered into the MLflow Model Registry as `WineClassifier` (version 1) and assigned alias `@champion`.
- **Test Set Evaluation:** Evaluated on the 36-sample test split, achieving **1.0000 Accuracy**, **1.0000 Macro-F1**, and **0.1059 Log Loss**.

---

## Automated MLOps Quality Gates (CI)

Integrated within `tests/test_model_gate.py` and enforced in GitHub Actions:

| Quality Gate | Condition / Threshold | Observed Value | Verification Status |
| :--- | :--- | :--- | :--- |
| **Metric Threshold Gate** | Validation Macro F1 $\ge$ 0.88 | Macro F1 = 1.0000 (Test) / 0.9789 (Val) | **PASSED** (+12% safety margin) |
| **Inference Latency Gate** | Batch inference time $\le$ 30 ms | Elapsed Time = ~1.8 ms | **PASSED** (94% under budget) |
| **Output Schema Integrity** | Predictions strictly in $\{0, 1, 2\}$ | Unique classes: $\{0, 1, 2\}$ | **PASSED** (No schema drift) |

---

## Git Collaboration & Engineered Merge Conflict (Milestone 5)

This repository demonstrates distributed version control with a clean feature-branching strategy and a documented merge conflict resolution:

### Feature Branching Strategy
1. `feature/data-pipeline`: Modularized `src/data.py` and unit tests in `tests/test_data.py`.
2. `feature/mlflow-tracking`: Implemented dual-classifier 5-fold CV training, logging, and model registry in `src/train.py`.
3. `feature/ci-pipeline`: Configured GitHub Actions CI automation in `.github/workflows/ci.yml`.

### Engineered Merge Conflict Resolution Log
```bash
# 1. Create conflict-simulation branch and edit configuration
git checkout -b conflict-simulation
# Changed RF-Config-1 n_estimators to 150 in src/train.py
git commit -am "chore(config): tune RF-Config-1 n_estimators to 150 on conflict-simulation"

# 2. Switch back to main and apply conflicting edit to the same line
git checkout main
# Changed RF-Config-1 n_estimators to 120 in src/train.py
git commit -am "chore(config): adjust RF-Config-1 n_estimators to 120 on main"

# 3. Attempt merge to induce conflict
git merge conflict-simulation
# Auto-merging src/train.py
# CONFLICT (content): Merge conflict in src/train.py
# Automatic merge failed; fix conflicts and then commit the result.

# 4. Resolve conflict manually (retaining validated baseline n_estimators=100)
git add src/train.py
git commit -m "Merge branch 'conflict-simulation' into main: resolve hyperparameter configuration conflict"
```

### Git Commit Graph (`git log --graph --all --oneline`)
```text
*   e09e646 (HEAD -> main) Merge branch 'conflict-simulation' into main: resolve hyperparameter configuration conflict
|\  
| * 042e7e4 (conflict-simulation) chore(config): tune RF-Config-1 n_estimators to 150 on conflict-simulation
* | 2a16724 chore(config): adjust RF-Config-1 n_estimators to 120 on main
|/  
*   a21b708 Merge branch 'feature/ci-pipeline' into main
|\  
| * 8c99f0c (feature/ci-pipeline) ci(github-actions): add automated CI workflow with linting and quality gates
|/  
*   2d4eb33 Merge branch 'feature/mlflow-tracking' into main
|\  
| * c5ecb15 (feature/mlflow-tracking) feat(mlflow): implement dual-classifier training with 5-fold CV and MLflow tracking
|/  
*   2833c81 Merge branch 'feature/data-pipeline' into main
|\  
| * d4e1e22 (feature/data-pipeline) feat(data): implement Wine dataset loading, validation, and stratified split
|/  
* c68dbfb chore: initial project scaffolding, environment setup and documentation
```

---

## Dataset

[UCI Wine Dataset](https://scikit-learn.org/stable/datasets/toy_dataset.html#wine-dataset)
- **Samples:** 178 chemical analyses of Italian wines
- **Features:** 13 continuous chemical properties (Alcohol, Malic acid, Ash, Alcalinity, Magnesium, Total phenols, Flavanoids, Nonflavanoid phenols, Proanthocyanins, Color intensity, Hue, OD280/OD315 of diluted wines, Proline)
- **Classes:** 3 cultivars (Cultivar 1, 2, and 3)
- **Split:** 80% Train (142 samples), 20% Test (36 samples), Stratified with fixed seed 42.

---

## Submission Artifacts

- **Report PDF:** `MLOps_A01_23F-0060.pdf` (Includes Table 1, MLflow UI screenshots, CI evidence, Git tree, and analytical response under 150 words).
- **Codebase:** Fully verified locally with `flake8` (0 errors) and `pytest` (13/13 passed).
