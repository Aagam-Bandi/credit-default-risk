# Credit Default Risk Model

Predicting next-month credit card default from six months of billing and repayment history. 25,247 customers, 19% default rate.

The modelling is the easy part. The work is in not being fooled — by accuracy on an imbalanced target, by leakage that inflates every metric, and by probabilities that don't mean what the threshold search assumes they mean.

```bash
pip install -r requirements.txt
python -m src.train --train data/train.csv
pytest
```

---

## Results

Held-out 20% split, metrics for the **default class specifically**:

| Model | AUC-ROC | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic Regression | 0.718 | 0.41 | 0.54 | 0.46 |
| Decision Tree | 0.721 | 0.45 | 0.52 | 0.48 |
| XGBoost (defaults) | 0.754 | 0.50 | 0.49 | 0.49 |
| XGBoost (tuned) | 0.791 | 0.50 | 0.57 | 0.53 |
| **XGBoost (tuned + calibrated)** | **0.789** | **0.51** | **0.56** | **0.53** |

Cross-validated, with SMOTE applied inside each fold:

| Model | AUC | 95% CI |
|---|---|---|
| Logistic Regression | 0.716 ± 0.007 | 0.710 – 0.722 |
| XGBoost (defaults) | 0.738 ± 0.007 | 0.732 – 0.743 |
| XGBoost (tuned) | 0.751 ± 0.005 | 0.746 – 0.755 |

The intervals don't overlap, so the tuned model's advantage is real rather than a favourable split.

**Why default-class metrics and not accuracy.** The tuned model's overall accuracy is 0.81 and its weighted F1 is 0.82 — both look much healthier than the table above, and both are meaningless here. A model predicting "no default" for everyone scores 0.81 accuracy on this dataset. Any metric a constant function can win is not measuring the thing you care about.

---

## Three ways to get this wrong

### Leakage

Two statistics are computed from data, and both must see the training fold only:

- **Age imputation.** The median must come from training rows. Computing it over train and test combined leaks the test distribution. `impute_age` takes the median as a parameter rather than computing it internally, which makes the correct usage the only usage.
- **SMOTE.** It synthesises minority points by interpolating between neighbours. Applied before the split, synthetic neighbours of test rows end up in training — the model has effectively seen the test set. Resampling happens after the split, and inside each CV fold.

Both were present in the original notebook version of this analysis. Fixing them moved AUC down before tuning moved it back up.

### Threshold

The 0.5 default is wrong at a 19% base rate. The model is well-calibrated toward the majority class, so at 0.5 it barely predicts default at all.

Two selection strategies are implemented:

| Strategy | Threshold | Precision | Recall |
|---|---|---|---|
| F1-optimal | 0.257 | 0.51 | 0.56 |
| Cost-optimal (5:1 FN:FP) | 0.175 | 0.40 | 0.67 |

F1 weights precision and recall equally, which is a statement about the metric, not about lending. A missed default costs the outstanding balance; a false positive costs the margin on a wrongly declined customer. At a 5:1 ratio the optimal cutoff drops and recall rises to 0.67 — the model flags more people because missing one is expensive. The 5:1 figure is a placeholder; real numbers come from the lender's book, and `cost_sensitive_threshold` takes them as arguments.

### Calibration

Tuning used `scale_pos_weight` to handle imbalance, which fixes the ranking and wrecks the probability scale. Threshold selection sits on top of those probabilities, so this isn't cosmetic.

| | Brier score | Max deviation |
|---|---|---|
| Raw tuned model | 0.174 | **0.342** |
| After isotonic calibration | **0.119** | **0.035** |

Before calibration the model said 0.68 for a group that defaulted at 0.34. After, predicted and observed track within 3.5 points across all ten quantile bins:

```
  predicted   observed
     0.031      0.040
     0.057      0.040
     0.078      0.074
     0.092      0.102
     0.111      0.117
     0.137      0.137
     0.165      0.162
     0.222      0.212
     0.366      0.339
     0.654      0.689
```

AUC is essentially unchanged (0.791 → 0.789), exactly as expected — isotonic regression is monotonic, so it rewrites the probability scale without touching the ranking.

---

## Feature engineering

Six features, each a hypothesis about repayment behaviour:

| Feature | Hypothesis |
|---|---|
| `credit_util` | Average bill ÷ limit. Customers at their ceiling have no buffer for a shock. |
| `total_pay_amt` | Absolute repayment volume — separates servicing debt from merely carrying it. |
| `repayment_ratio` | Repayment relative to amount owed. Large payments against larger bills is a different risk from small payments against small bills. |
| `num_delays` | Count of delayed months. Blunt, strongly monotonic with default rate. |
| `overdue_count` | Months where payment fell short of the prior bill. Catches habitual partial payment — which `num_delays` misses, since a partial payment isn't formally a delay. |
| `delinq_streak` | Longest *consecutive* delinquency run. Three scattered missed months and three consecutive ones give identical `num_delays` and very different trajectories. |

The twelve raw monthly columns are dropped once these exist — keeping both gives the model twelve noisy views of a signal the aggregates express cleanly.

`education` contains undocumented codes 0, 5 and 6, and `marriage` contains 0. They affect a small number of rows, so they're folded into the modal category rather than dropped.

---

## Tuning

Randomised search, 40 candidates, 3-fold CV, scored by AUC. Search runs on unresampled data with `scale_pos_weight` handling the imbalance — tuning over SMOTE-augmented data optimises the model to fit synthetic points, which isn't the objective.

Selected: `n_estimators=500`, `max_depth=5`, `learning_rate=0.01`, `subsample=0.8`, `colsample_bytree=0.7`, `min_child_weight=7`, `gamma=0`.

Low learning rate with many shallow trees — the search converged on heavy regularisation, which is what you'd expect on 20k rows and 15 features.

---

## Layout

```
src/
  features.py   cleaning, imputation, feature engineering
  train.py      splitting, tuning, calibration, model comparison
  evaluate.py   cross-validation, calibration assessment, cost-sensitive thresholds
tests/          23 tests
notebooks/      original EDA
data/           train.csv, validate.csv
```

```bash
python -m src.train --n-iter 100     # wider search
python -m src.train -v               # split stats, tuning progress
pytest                               # 23 tests, ~1s
```

**As a library**

```python
from src import prepare, cost_sensitive_threshold, assess_calibration

X = prepare(new_customers, age_median=34.0)          # training-derived median
thr, cost = cost_sensitive_threshold(y, probs, cost_false_negative=8.0,
                                     cost_false_positive=1.0)
```

---

## Limitations

- **Precision is around 0.51 at the F1 threshold.** Half the flagged customers don't default. That's a property of this feature set rather than a fixable bug — six months of history, no income, employment or bureau signal.
- **The cost ratio is illustrative.** The machinery takes real costs; the dataset doesn't include them.

---

## Stack

Python · pandas · scikit-learn · XGBoost · imbalanced-learn · pytest

Originally built for the FinClub open project, IIT Roorkee. Restructured from notebook to tested package, with leakage in the imputation and resampling steps fixed, and hyperparameter search plus probability calibration added.

## License

MIT
