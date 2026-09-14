"""Feature engineering for credit default prediction.

Every feature here is a hypothesis about repayment behaviour, not a mechanical
transform. The comments state the hypothesis, because a feature whose rationale
isn't written down is a feature nobody can evaluate later.
"""

import pandas as pd

PAY_COLS = ["pay_0", "pay_2", "pay_3", "pay_4", "pay_5", "pay_6"]
BILL_COLS = [f"Bill_amt{i}" for i in range(1, 7)]
PAY_AMT_COLS = [f"pay_amt{i}" for i in range(1, 7)]

# Source data uses out-of-range codes the schema does not define.
# Folding them into the modal category preserves the rows; dropping them
# would discard several hundred customers for a data-entry artefact.
EDUCATION_MAP = {0: 2, 1: 1, 2: 2, 3: 3, 4: 4, 5: 2, 6: 2}
MARRIAGE_MAP = {0: 2, 1: 1, 2: 2, 3: 3}


def _max_consecutive(row) -> int:
    """Longest run of consecutive delayed months."""
    streak = best = 0
    for value in row:
        if value > 0:
            streak += 1
            best = max(best, streak)
        else:
            streak = 0
    return best


def clean_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["education"] = df["education"].map(EDUCATION_MAP)
    df["marriage"] = df["marriage"].map(MARRIAGE_MAP)
    return df


def impute_age(df: pd.DataFrame, median: float) -> pd.DataFrame:
    """Fill missing ages with a median supplied by the caller.

    The median is a parameter rather than computed here on purpose: computing
    it over train+test combined leaks test distribution into training. It must
    be derived from the training split alone and then applied to both.
    """
    df = df.copy()
    df["age"] = df["age"].fillna(median)
    return df


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Add behavioural features and drop the raw monthly columns they replace."""
    df = df.copy()

    # How much of the available limit is habitually in use. High utilisation
    # is a classic distress signal — the customer is living at their ceiling.
    df["credit_util"] = df["AVG_Bill_amt"] / (df["LIMIT_BAL"] + 1)

    # Absolute repayment volume over the window. Separates customers who are
    # actively servicing debt from those merely carrying it.
    df["total_pay_amt"] = df[PAY_AMT_COLS].sum(axis=1)
    df["total_bill_amt"] = df[BILL_COLS].sum(axis=1)

    # Repayment relative to what was owed. A customer paying large absolute
    # amounts against even larger bills is not the same risk as one paying
    # small amounts against small bills.
    df["repayment_ratio"] = df["total_pay_amt"] / (df["AVG_Bill_amt"] * 6 + 1)

    # Count of delayed months. Blunt but strongly monotonic with default rate.
    df["num_delays"] = (df[PAY_COLS] > 0).sum(axis=1)

    # Months where payment fell short of the previous month's bill — captures
    # habitual partial payment, which "num_delays" misses because a partial
    # payment is not formally a delay.
    overdue = pd.Series(0, index=df.index)
    for i in range(2, 7):
        overdue += (df[f"Bill_amt{i-1}"] > df[f"pay_amt{i}"]).astype(int)
    df["overdue_count"] = overdue

    # Longest consecutive delinquency run. Distinguishes a customer who missed
    # three scattered months from one sliding continuously — same "num_delays",
    # very different trajectory.
    df["delinq_streak"] = df[PAY_COLS].apply(_max_consecutive, axis=1)

    return df.drop(columns=BILL_COLS + PAY_AMT_COLS)


def prepare(df: pd.DataFrame, age_median: float) -> pd.DataFrame:
    """Full transform: clean, impute, engineer."""
    return engineer(impute_age(clean_categoricals(df), age_median))
