"""Known-answer tests for effect sizes and test selection."""
import numpy as np
import pandas as pd
import pytest

from src.stats import cohens_dz, friedman, paired_test, rank_biserial


def test_rank_biserial_extremes_and_balance():
    assert rank_biserial(np.array([1.0, 2.0, 3.0])) == 1.0
    assert rank_biserial(np.array([-1.0, -2.0])) == -1.0
    assert rank_biserial(np.array([1.0, -1.0, 2.0, -2.0])) == 0.0


def test_cohens_dz():
    assert cohens_dz(np.array([1.0, 2.0, 3.0])) == pytest.approx(2.0)  # mean 2, sd 1


def test_paired_test_detects_shift_and_picks_test_by_normality():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 40)
    normal = paired_test(a, a + 1 + rng.normal(0, 0.5, 40))
    assert normal["test"] == "paired t" and normal["p"] < 1e-6 and normal["diff_ci_lo"] > 0
    skewed = paired_test(a, a + rng.exponential(1, 40) ** 3)
    assert skewed["test"] == "Wilcoxon" and skewed["effect_size"] > 0.8


def test_friedman_kendall_w_is_one_for_identical_rankings():
    wide = pd.DataFrame({"a": [1, 2, 3, 4.0], "b": [5, 6, 7, 8.0], "c": [9, 10, 11, 12.0]})
    assert friedman(wide)["kendall_w"] == pytest.approx(1.0)
