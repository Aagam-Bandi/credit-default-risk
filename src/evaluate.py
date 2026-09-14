"""Evaluation: cross-validated metrics, calibration, and cost-sensitive thresholds.

Separated from training because these answer different questions. `train.py`
asks "which model?"; this module asks "can I trust the number, and where should
the cutoff actually sit?"
"""

import logging
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, precision_recall_curve, roc_auc_score
from sklearn.model_selection import StratifiedKFold

logger = logging.getLogger(__name__)

N_SPLITS = 5
RANDOM_STATE = 42


@dataclass
class CVResult:
    """Cross-validated AUC with a spread, so two models can be compared honestly."""
    mean_auc: float
    std_auc: float
    fold_aucs: list[float]

    @property
    def ci95(self) -> tuple[float, float]:
        """Approximate 95% interval on the mean across folds."""
        margin = 1.96 * self.std_auc / np.sqrt(len(self.fold_aucs))
        return (self.mean_auc - margin, self.mean_auc + margin)

    def __str__(self) -> str:
        lo, hi = self.ci95
        return f"AUC {self.mean_auc:.4f} ± {self.std_auc:.4f}  (95% CI {lo:.4f}–{hi:.4f})"


def cross_validate_auc(model_factory, X: pd.DataFrame, y: pd.Series,
                       resample=None) -> CVResult:
    """Stratified k-fold AUC.

    `model_factory` is a callable returning a fresh unfitted model — reusing a
    fitted instance across folds leaks the previous fold's fit.

    `resample` is applied inside each fold, on the training portion only. This
    is the whole reason the function exists: calling SMOTE outside the loop and
    then cross-validating is a standard way to produce an AUC that is two or
    three points too high.
    """
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    aucs: list[float] = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_te, y_te = X.iloc[test_idx], y.iloc[test_idx]

        if resample is not None:
            X_tr, y_tr = resample(X_tr, y_tr)

        model = model_factory()
        model.fit(X_tr, y_tr)
        auc = roc_auc_score(y_te, model.predict_proba(X_te)[:, 1])
        aucs.append(auc)
        logger.debug("fold %d AUC %.4f", fold, auc)

    return CVResult(float(np.mean(aucs)), float(np.std(aucs)), aucs)


def cost_sensitive_threshold(
    y_true,
    y_prob,
    cost_false_negative: float,
    cost_false_positive: float,
) -> tuple[float, float]:
    """Pick the cutoff minimising expected cost, not maximising F1.

    F1 weights precision and recall equally, which is a statement about the
    metric rather than about the business. A missed default costs the lender
    the outstanding balance; a false positive costs a declined customer's
    margin. Those are rarely equal and often differ by an order of magnitude.

    Returns (threshold, expected_cost_per_customer).
    """
    thresholds = np.linspace(0.01, 0.99, 197)
    y_true = np.asarray(y_true)
    best_thr, best_cost = 0.5, float("inf")

    for thr in thresholds:
        y_pred = (y_prob >= thr).astype(int)
        fn = int(((y_true == 1) & (y_pred == 0)).sum())
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        cost = (fn * cost_false_negative + fp * cost_false_positive) / len(y_true)
        if cost < best_cost:
            best_thr, best_cost = float(thr), float(cost)

    return best_thr, best_cost


def best_f1_threshold(y_true, y_prob) -> float:
    """Cutoff maximising F1 on the default class."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    return float(thresholds[int(np.argmax(f1[:-1]))])


@dataclass
class CalibrationReport:
    """Whether predicted probabilities behave like probabilities.

    Threshold selection assumes they do. If the model says 0.30 for a group of
    customers, roughly 30% of them should default — otherwise a cutoff of 0.38
    doesn't mean what the threshold search thinks it means.
    """
    brier_score: float
    max_deviation: float
    bin_predicted: list[float]
    bin_observed: list[float]

    @property
    def is_well_calibrated(self) -> bool:
        return self.max_deviation < 0.10

    def to_dict(self) -> dict:
        return asdict(self) | {"is_well_calibrated": self.is_well_calibrated}

    def __str__(self) -> str:
        verdict = "well calibrated" if self.is_well_calibrated else "MISCALIBRATED"
        lines = [
            f"Brier score        {self.brier_score:.4f}",
            f"Max deviation      {self.max_deviation:.4f}  ({verdict})",
            "",
            "  predicted   observed",
        ]
        for p, o in zip(self.bin_predicted, self.bin_observed):
            lines.append(f"     {p:.3f}      {o:.3f}")
        return "\n".join(lines)


def assess_calibration(y_true, y_prob, n_bins: int = 10) -> CalibrationReport:
    observed, predicted = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="quantile")
    deviations = np.abs(np.asarray(predicted) - np.asarray(observed))
    return CalibrationReport(
        brier_score=float(brier_score_loss(y_true, y_prob)),
        max_deviation=float(deviations.max()),
        bin_predicted=[float(x) for x in predicted],
        bin_observed=[float(x) for x in observed],
    )
