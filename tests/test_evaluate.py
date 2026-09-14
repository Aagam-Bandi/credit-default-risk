import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
import pandas as pd

from src.evaluate import (
    assess_calibration,
    best_f1_threshold,
    cost_sensitive_threshold,
    cross_validate_auc,
)


@pytest.fixture
def imbalanced():
    X, y = make_classification(
        n_samples=800, n_features=8, n_informative=5,
        weights=[0.8, 0.2], random_state=42,
    )
    return pd.DataFrame(X, columns=[f"f{i}" for i in range(8)]), pd.Series(y)


class TestThresholds:
    def test_f1_threshold_in_range(self, imbalanced):
        X, y = imbalanced
        model = LogisticRegression(max_iter=1000).fit(X, y)
        thr = best_f1_threshold(y, model.predict_proba(X)[:, 1])
        assert 0.0 < thr < 1.0

    def test_expensive_false_negatives_lower_the_threshold(self, imbalanced):
        # The core claim of the cost function: if missing a default is
        # expensive, the model should flag more people, not fewer.
        X, y = imbalanced
        prob = LogisticRegression(max_iter=1000).fit(X, y).predict_proba(X)[:, 1]
        cheap_fn, _ = cost_sensitive_threshold(y, prob, 1.0, 1.0)
        costly_fn, _ = cost_sensitive_threshold(y, prob, 20.0, 1.0)
        assert costly_fn < cheap_fn

    def test_cost_is_non_negative(self, imbalanced):
        X, y = imbalanced
        prob = LogisticRegression(max_iter=1000).fit(X, y).predict_proba(X)[:, 1]
        _, cost = cost_sensitive_threshold(y, prob, 5.0, 1.0)
        assert cost >= 0


class TestCrossValidation:
    def test_returns_one_auc_per_fold(self, imbalanced):
        X, y = imbalanced
        result = cross_validate_auc(lambda: LogisticRegression(max_iter=1000), X, y)
        assert len(result.fold_aucs) == 5
        assert 0.5 < result.mean_auc <= 1.0

    def test_confidence_interval_brackets_mean(self, imbalanced):
        X, y = imbalanced
        result = cross_validate_auc(lambda: LogisticRegression(max_iter=1000), X, y)
        lo, hi = result.ci95
        assert lo <= result.mean_auc <= hi

    def test_factory_is_called_fresh_each_fold(self, imbalanced):
        X, y = imbalanced
        calls = []

        def factory():
            calls.append(1)
            return LogisticRegression(max_iter=1000)

        cross_validate_auc(factory, X, y)
        assert len(calls) == 5


class TestCalibration:
    def test_perfect_probabilities_are_well_calibrated(self):
        rng = np.random.default_rng(0)
        prob = rng.uniform(0, 1, 4000)
        y = (rng.uniform(0, 1, 4000) < prob).astype(int)
        assert assess_calibration(y, prob).is_well_calibrated

    def test_inflated_probabilities_are_flagged(self):
        rng = np.random.default_rng(0)
        true_p = rng.uniform(0, 0.3, 4000)
        y = (rng.uniform(0, 1, 4000) < true_p).astype(int)
        inflated = np.clip(true_p * 3, 0, 1)
        assert not assess_calibration(y, inflated).is_well_calibrated

    def test_brier_is_bounded(self):
        rng = np.random.default_rng(0)
        prob = rng.uniform(0, 1, 500)
        y = rng.integers(0, 2, 500)
        assert 0.0 <= assess_calibration(y, prob).brier_score <= 1.0
