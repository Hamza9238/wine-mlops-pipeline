"""
test_data.py — Unit tests for data loading and validation.
"""

import pandas as pd
import pytest
from src.data import load_data, validate_data, split_data


class TestLoadData:
    """Tests for load_data()."""

    def test_returns_dataframe_and_series(self):
        X, y = load_data()
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)

    def test_shape(self):
        X, y = load_data()
        assert X.shape == (178, 13), (
            f"Expected (178, 13), got {X.shape}"
        )
        assert y.shape == (178,), (
            f"Expected (178,), got {y.shape}"
        )

    def test_no_nulls(self):
        X, y = load_data()
        assert not X.isnull().any().any()
        assert not y.isnull().any()

    def test_target_classes(self):
        _, y = load_data()
        assert set(y.unique()) == {0, 1, 2}


class TestValidateData:
    """Tests for validate_data()."""

    def test_valid_data_passes(self):
        X, y = load_data()
        # Should not raise
        validate_data(X, y)

    def test_null_features_raises(self):
        X, y = load_data()
        X.iloc[0, 0] = None
        with pytest.raises(ValueError, match="null"):
            validate_data(X, y)

    def test_wrong_feature_count_raises(self):
        X, y = load_data()
        X_bad = X.drop(columns=[X.columns[0]])
        with pytest.raises(ValueError, match="13 features"):
            validate_data(X_bad, y)


class TestSplitData:
    """Tests for split_data()."""

    def test_split_sizes(self):
        X, y = load_data()
        X_train, X_test, y_train, y_test = split_data(X, y)
        assert len(X_train) + len(X_test) == 178
        # ~80 / 20
        assert len(X_test) == pytest.approx(
            178 * 0.20, abs=2
        )

    def test_stratification(self):
        """Each class should be roughly proportional in both splits."""
        X, y = load_data()
        _, _, y_train, y_test = split_data(X, y)
        for cls in [0, 1, 2]:
            train_frac = (y_train == cls).mean()
            test_frac = (y_test == cls).mean()
            assert abs(train_frac - test_frac) < 0.05, (
                f"Class {cls} not stratified: "
                f"train={train_frac:.2f}, test={test_frac:.2f}"
            )

    def test_reproducibility(self):
        X, y = load_data()
        split1 = split_data(X, y)
        split2 = split_data(X, y)
        assert split1[0].equals(split2[0]), (
            "Splits are not reproducible with same random_state"
        )
