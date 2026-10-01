"""
data.py - Wine dataset loading and preprocessing.

Loads the Wine dataset from sklearn, validates it,
and provides an 80/20 stratified train-test split.
"""

import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split


def load_data():
    """
    Load the Wine dataset and return it as a pandas DataFrame
    plus the target Series.

    Returns
    -------
    X : pd.DataFrame  (178 × 13)
    y : pd.Series      (178,)
    """
    wine = load_wine()
    X = pd.DataFrame(wine.data, columns=wine.feature_names)
    y = pd.Series(wine.target, name="target")
    return X, y


def validate_data(X, y):
    """
    Run sanity checks required by the assignment rubric.

    Checks
    ------
    1. No null / NaN values in features or target.
    2. Exactly 13 features.
    3. Target contains only classes {0, 1, 2}.

    Raises
    ------
    ValueError  if any check fails.
    """
    if X.isnull().any().any():
        raise ValueError("Feature matrix contains null values.")
    if y.isnull().any():
        raise ValueError("Target vector contains null values.")
    if X.shape[1] != 13:
        raise ValueError(
            f"Expected 13 features, got {X.shape[1]}."
        )
    if set(y.unique()) != {0, 1, 2}:
        raise ValueError(
            f"Unexpected target classes: {sorted(y.unique())}"
        )
    print("[OK] Data validation passed - no nulls, 13 features, "
          "classes {0,1,2}.")


def split_data(X, y, test_size=0.20, random_state=42):
    """
    Stratified 80/20 train-test split.

    Parameters
    ----------
    X : pd.DataFrame
    y : pd.Series
    test_size : float   (default 0.20)
    random_state : int  (default 42)

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )
    print(f"[OK] Split complete - train: {len(X_train)}, "
          f"test: {len(X_test)}")
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    X, y = load_data()
    validate_data(X, y)
    X_train, X_test, y_train, y_test = split_data(X, y)
    print(f"\nDataset shape : {X.shape}")
    print(f"Train shape   : {X_train.shape}")
    print(f"Test shape    : {X_test.shape}")
