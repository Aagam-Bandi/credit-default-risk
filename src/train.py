"""Train, tune and compare default-prediction models under class imbalance."""

import argparse
import json
import logging

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from .evaluate import (
    assess_calibration,
    best_f1_threshold,
    cost_sensitive_threshold,
    cross_validate_auc,
)
from .features import prepare

logger = logging.getLogger(__name__)

RANDOM_STATE = 42
TEST_SIZE = 0.2

# Illustrative unit costs for the cost-sensitive threshold. A missed default
# costs the outstanding balance; a false positive costs the margin on a
# customer wrongly declined. The 5:1 ratio is a placeholder — real figures
# would come from the lender's own book.
COST_FALSE_NEGATIVE = 5.0
COST_FALSE_POSITIVE = 1.0

XGB_SEARCH_SPACE = {
    "n_estimators": [100, 200, 300, 500],
    "max_depth": [3, 4, 5, 6, 8],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "subsample": [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
    "min_child_weight": [1, 3, 5, 7],
    "gamma": [0, 0.1, 0.3, 0.5],
}


def _smote(X, y):
    return SMOTE(random_state=RANDOM_STATE).fit_resample(X, y)


def load_and_split(train_path: str):
    """Load, split, then derive every statistic from the training fold only."""
    df = pd.read_csv(train_path).drop(columns=["Customer_ID"], errors="ignore")
    df = df[df["next_month_default"].notnull()]
    df["next_month_default"] = df["next_month_default"].astype(int)

    y = df["next_month_default"]
    X_raw = df.drop(columns=["next_month_default"])

    X_tr_raw, X_te_raw, y_tr, y_te = train_test_split(
        X_raw, y, stratify=y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    age_median = X_tr_raw["age"].median()
    logger.info("age median from training split: %.1f", age_median)

    return prepare(X_tr_raw, age_median), prepare(X_te_raw, age_median), y_tr, y_te


def tune_xgboost(X, y, n_iter: int = 40) -> XGBClassifier:
    """Randomised search over the XGBoost space, scored by AUC.

    Search runs on the unresampled training fold with `scale_pos_weight` doing
    the imbalance correction instead of SMOTE. Running a search over
    SMOTE-augmented data tunes the model to fit synthetic points, which is not
    the objective.
    """
    ratio = float((y == 0).sum() / (y == 1).sum())
    logger.info("tuning XGBoost (%d candidates, scale_pos_weight=%.2f)", n_iter, ratio)

    search = RandomizedSearchCV(
        XGBClassifier(
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            scale_pos_weight=ratio,
            n_jobs=-1,
        ),
        param_distributions=XGB_SEARCH_SPACE,
        n_iter=n_iter,
        scoring="roc_auc",
        cv=3,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=0,
    )
    search.fit(X, y)
    logger.info("best CV AUC %.4f with %s", search.best_score_, search.best_params_)
    return search.best_estimator_


def evaluate(name, model, X_test, y_test, threshold=None) -> dict:
    y_prob = model.predict_proba(X_test)[:, 1]
    thr = threshold if threshold is not None else best_f1_threshold(y_test, y_prob)
    y_pred = (y_prob >= thr).astype(int)

    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    auc = roc_auc_score(y_test, y_prob)

    print(f"\n===== {name} =====")
    print(f"threshold: {thr:.3f}")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred, zero_division=0))
    print(f"AUC-ROC: {auc:.4f}")

    return {
        "model": name,
        "threshold": round(thr, 4),
        "auc_roc": round(auc, 4),
        "precision_default": round(report["1"]["precision"], 4),
        "recall_default": round(report["1"]["recall"], 4),
        "f1_default": round(report["1"]["f1-score"], 4),
    }


def run(train_path: str, out_path: str | None = None, n_iter: int = 40) -> dict:
    X_train, X_test, y_train, y_test = load_and_split(train_path)

    X_bal, y_bal = _smote(X_train, y_train)
    logger.info("training rows after SMOTE: %d (from %d)", len(X_bal), len(X_train))

    results = []

    baselines = [
        ("Logistic Regression", make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        )),
        ("Decision Tree", DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE)),
        ("XGBoost (defaults)", XGBClassifier(eval_metric="logloss", random_state=RANDOM_STATE)),
    ]
    for name, model in baselines:
        model.fit(X_bal, y_bal)
        results.append(evaluate(name, model, X_test, y_test))

    tuned = tune_xgboost(X_train, y_train, n_iter=n_iter)
    results.append(evaluate("XGBoost (tuned)", tuned, X_test, y_test))

    # scale_pos_weight fixes the imbalance but distorts the probability scale:
    # the raw tuned model reports ~0.68 for groups that default at ~0.34.
    # Threshold selection is meaningless on top of that, so the ranking model
    # is wrapped in an isotonic calibrator fitted by internal CV. Ranking (AUC)
    # is unchanged; the probabilities become usable.
    logger.info("calibrating tuned model (isotonic, 5-fold internal CV)")
    best = CalibratedClassifierCV(tuned, method="isotonic", cv=5)
    best.fit(X_train, y_train)
    results.append(evaluate("XGBoost (tuned + calibrated)", best, X_test, y_test))

    # Cross-validated AUC with resampling inside each fold — the only number
    # here that supports a claim about one model beating another.
    print("\n===== Cross-validated AUC (5-fold, SMOTE inside each fold) =====")
    cv_results = {}
    for name, factory in [
        ("Logistic Regression", lambda: make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))),
        ("XGBoost (defaults)", lambda: XGBClassifier(
            eval_metric="logloss", random_state=RANDOM_STATE)),
        ("XGBoost (tuned)", lambda: XGBClassifier(**tuned.get_params())),
    ]:
        cv = cross_validate_auc(factory, X_train, y_train, resample=_smote)
        print(f"{name:<24} {cv}")
        cv_results[name] = {
            "mean_auc": round(cv.mean_auc, 4),
            "std_auc": round(cv.std_auc, 4),
            "ci95": [round(v, 4) for v in cv.ci95],
        }

    # Calibration of the model actually being shipped.
    y_prob = best.predict_proba(X_test)[:, 1]
    calib = assess_calibration(y_test, y_prob)
    raw_calib = assess_calibration(y_test, tuned.predict_proba(X_test)[:, 1])
    print("\n===== Calibration: before vs after =====")
    print(f"raw tuned model    Brier {raw_calib.brier_score:.4f}  "
          f"max deviation {raw_calib.max_deviation:.4f}")
    print(f"calibrated model   Brier {calib.brier_score:.4f}  "
          f"max deviation {calib.max_deviation:.4f}")
    print()
    print(calib)

    cost_thr, cost = cost_sensitive_threshold(
        y_test, y_prob, COST_FALSE_NEGATIVE, COST_FALSE_POSITIVE
    )
    f1_thr = best_f1_threshold(y_test, y_prob)
    print("\n===== Threshold selection (calibrated model) =====")
    print(f"F1-optimal          {f1_thr:.3f}")
    print(f"Cost-optimal ({COST_FALSE_NEGATIVE:.0f}:{COST_FALSE_POSITIVE:.0f})  "
          f"{cost_thr:.3f}   expected cost {cost:.4f} per customer")
    cost_metrics = evaluate("XGBoost (calibrated, cost threshold)", best, X_test, y_test, cost_thr)
    results.append(cost_metrics)

    payload = {
        "holdout": results,
        "cross_validated_auc": cv_results,
        "calibration": calib.to_dict(),
        "thresholds": {
            "f1_optimal": round(f1_thr, 4),
            "cost_optimal": round(cost_thr, 4),
            "cost_ratio_fn_fp": f"{COST_FALSE_NEGATIVE:.0f}:{COST_FALSE_POSITIVE:.0f}",
        },
        "calibration_before": raw_calib.to_dict(),
        "best_params": {k: v for k, v in tuned.get_params().items()
                        if k in XGB_SEARCH_SPACE},
    }

    if out_path:
        with open(out_path, "w") as fh:
            json.dump(payload, fh, indent=2, default=str)
        print(f"\nwrote {out_path}")

    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Train credit default models.")
    parser.add_argument("--train", default="data/train.csv")
    parser.add_argument("--out", default="results.json")
    parser.add_argument("--n-iter", type=int, default=40, help="search candidates")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s  %(levelname)-7s %(message)s",
    )
    run(args.train, args.out, args.n_iter)


if __name__ == "__main__":
    main()
