import numpy as np
import pandas as pd
import pytest

from src.features import (
    EDUCATION_MAP,
    MARRIAGE_MAP,
    _max_consecutive,
    clean_categoricals,
    engineer,
    impute_age,
    prepare,
)


@pytest.fixture
def sample():
    return pd.DataFrame({
        "marriage": [1, 2, 0, 3],
        "sex": [1, 2, 1, 2],
        "education": [1, 5, 0, 6],
        "LIMIT_BAL": [100_000, 50_000, 200_000, 20_000],
        "age": [30.0, np.nan, 45.0, 28.0],
        "pay_0": [0, 1, 0, 2], "pay_2": [0, 1, 0, 2], "pay_3": [0, 1, 2, 0],
        "pay_4": [1, 0, 0, 2], "pay_5": [0, 0, 1, 2], "pay_6": [0, 1, 0, 0],
        "Bill_amt1": [5000, 4000, 9000, 1000], "Bill_amt2": [5200, 4100, 9100, 1100],
        "Bill_amt3": [5400, 4200, 9200, 1200], "Bill_amt4": [5600, 4300, 9300, 1300],
        "Bill_amt5": [5800, 4400, 9400, 1400], "Bill_amt6": [6000, 4500, 9500, 1500],
        "pay_amt1": [500, 100, 9000, 50], "pay_amt2": [500, 100, 9000, 50],
        "pay_amt3": [500, 100, 9000, 50], "pay_amt4": [500, 100, 9000, 50],
        "pay_amt5": [500, 100, 9000, 50], "pay_amt6": [500, 100, 9000, 50],
        "AVG_Bill_amt": [5500, 4250, 9250, 1250],
        "PAY_TO_BILL_ratio": [0.09, 0.02, 0.97, 0.04],
    })


class TestMaxConsecutive:
    def test_no_delays(self):
        assert _max_consecutive([0, 0, 0, 0, 0, 0]) == 0

    def test_all_delayed(self):
        assert _max_consecutive([1, 1, 1, 1, 1, 1]) == 6

    def test_scattered_beats_nothing_but_loses_to_run(self):
        # Same total delays, different trajectory — this is the whole point
        # of the feature.
        scattered = _max_consecutive([1, 0, 1, 0, 1, 0])
        consecutive = _max_consecutive([1, 1, 1, 0, 0, 0])
        assert scattered == 1
        assert consecutive == 3

    def test_takes_longest_not_last(self):
        assert _max_consecutive([1, 1, 1, 0, 1, 0]) == 3


class TestCleanCategoricals:
    def test_out_of_range_codes_folded(self, sample):
        out = clean_categoricals(sample)
        assert set(out["education"].unique()) <= set(EDUCATION_MAP.values())
        assert set(out["marriage"].unique()) <= set(MARRIAGE_MAP.values())

    def test_no_rows_dropped(self, sample):
        assert len(clean_categoricals(sample)) == len(sample)

    def test_does_not_mutate_input(self, sample):
        before = sample["education"].tolist()
        clean_categoricals(sample)
        assert sample["education"].tolist() == before


class TestImputeAge:
    def test_fills_missing(self, sample):
        assert impute_age(sample, 33.0)["age"].isna().sum() == 0

    def test_uses_supplied_median_not_computed(self, sample):
        # Guards the leakage fix: the median must come from the caller, so a
        # training-fold value can be applied to test data unchanged.
        out = impute_age(sample, 99.0)
        assert out.loc[1, "age"] == 99.0


class TestEngineer:
    def test_expected_columns_added(self, sample):
        out = engineer(clean_categoricals(impute_age(sample, 33.0)))
        for col in ["credit_util", "total_pay_amt", "repayment_ratio",
                    "num_delays", "overdue_count", "delinq_streak"]:
            assert col in out.columns

    def test_raw_monthly_columns_dropped(self, sample):
        out = engineer(clean_categoricals(impute_age(sample, 33.0)))
        assert not [c for c in out.columns if c.startswith(("Bill_amt", "pay_amt"))]

    def test_no_division_by_zero(self):
        df = pd.DataFrame({
            "LIMIT_BAL": [0], "AVG_Bill_amt": [0], "age": [30.0],
            "marriage": [1], "sex": [1], "education": [1],
            **{f"pay_{i}": [0] for i in [0, 2, 3, 4, 5, 6]},
            **{f"Bill_amt{i}": [0] for i in range(1, 7)},
            **{f"pay_amt{i}": [0] for i in range(1, 7)},
            "PAY_TO_BILL_ratio": [0.0],
        })
        out = engineer(df)
        assert np.isfinite(out["credit_util"]).all()
        assert np.isfinite(out["repayment_ratio"]).all()

    def test_num_delays_counts_positive_only(self, sample):
        out = engineer(clean_categoricals(impute_age(sample, 33.0)))
        assert out.loc[0, "num_delays"] == 1   # only pay_4 > 0
        assert out.loc[3, "num_delays"] == 4


class TestPrepare:
    def test_train_and_test_produce_identical_schema(self, sample):
        train, test = sample.iloc[:2], sample.iloc[2:]
        a = prepare(train, 33.0)
        b = prepare(test, 33.0)
        assert list(a.columns) == list(b.columns)
