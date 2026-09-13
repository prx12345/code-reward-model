"""Split construction and the leakage guards -- the easiest place to fake a result."""

import pytest

from src.data import LeakageError
from src.dataset import (Splits, assert_no_code_overlap, assert_no_leakage,
                         assert_no_problem_overlap, build_rows, class_weights,
                         split_by_problem)


def make_rows(n_problems=40, n_candidates=8):
    rows = []
    for p in range(n_problems):
        for c in range(n_candidates):
            rows.append({
                "candidate_id": f"mbpp/{p}#g#{c}",
                "problem_id": f"mbpp/{p}",
                "benchmark": "mbpp", "generator": "g",
                "problem_text": f"problem {p}",
                "code": f"def f_{p}_{c}():\n    return {c}",
                "label": int(c % 2 == 0),
                "outcome": "pass" if c % 2 == 0 else "assertion_failure",
                "error_type": None, "multiplicity": 1,
            })
    return rows


def test_all_candidates_for_a_problem_stay_together():
    splits = split_by_problem(make_rows())
    for name_a, group_a in (("train", splits.train), ("val", splits.val), ("test", splits.test)):
        ids_a = {r["problem_id"] for r in group_a}
        for name_b, group_b in (("train", splits.train), ("val", splits.val), ("test", splits.test)):
            if name_a >= name_b:
                continue
            assert not (ids_a & {r["problem_id"] for r in group_b})


def test_splits_are_deterministic_across_calls():
    rows = make_rows()
    first = split_by_problem(rows)
    second = split_by_problem(rows)
    assert [r["candidate_id"] for r in first.test] == [r["candidate_id"] for r in second.test]


def test_every_row_lands_in_exactly_one_split():
    rows = make_rows()
    splits = split_by_problem(rows)
    total = len(splits.train) + len(splits.val) + len(splits.test)
    assert total == len(rows)


def test_problem_overlap_guard_fires():
    shared = [{"problem_id": "mbpp/1", "code": "a"}]
    with pytest.raises(LeakageError, match="problem-id overlap"):
        assert_no_problem_overlap(shared, shared)


def test_row_level_split_is_caught_by_the_guard():
    """Simulates the exact mistake the guard exists to prevent."""
    rows = make_rows(n_problems=10)
    bad = Splits(train=rows[::2], val=[], test=rows[1::2])
    with pytest.raises(LeakageError):
        assert_no_leakage(bad)


def test_code_overlap_guard_fires_on_identical_snippets():
    a = [{"problem_id": "mbpp/1", "code": "def f():\n    return 1"}]
    b = [{"problem_id": "mbpp/2", "code": "def f():\n    return 1"}]
    with pytest.raises(LeakageError, match="identical code"):
        assert_no_code_overlap(a, b)


def test_code_overlap_ignores_trailing_whitespace_differences():
    a = [{"problem_id": "mbpp/1", "code": "def f():\n    return 1  "}]
    b = [{"problem_id": "mbpp/2", "code": "def f():\n    return 1"}]
    with pytest.raises(LeakageError):
        assert_no_code_overlap(a, b)


def test_humaneval_in_train_is_rejected():
    rows = make_rows(n_problems=4)
    for row in rows:
        row["problem_id"] = "humaneval/HumanEval_1"
    with pytest.raises(LeakageError):
        assert_no_leakage(Splits(train=rows, val=[], test=[]))


def test_build_rows_drops_harness_errors_rather_than_labeling_them_zero():
    problems = [{"problem_id": "mbpp/1", "classifier_prompt": "p"}]
    labeled = [
        {"candidate_id": "a", "problem_id": "mbpp/1", "benchmark": "mbpp",
         "generator": "g", "code": "def f(): pass", "label": 0,
         "outcome": "harness_error", "error_type": "X"},
        {"candidate_id": "b", "problem_id": "mbpp/1", "benchmark": "mbpp",
         "generator": "g", "code": "def g(): pass", "label": 1,
         "outcome": "pass", "error_type": None},
    ]
    rows = build_rows(labeled, problems)
    assert [r["candidate_id"] for r in rows] == ["b"]


def test_build_rows_deduplicates_and_counts_multiplicity():
    problems = [{"problem_id": "mbpp/1", "classifier_prompt": "p"}]
    labeled = [
        {"candidate_id": f"c{i}", "problem_id": "mbpp/1", "benchmark": "mbpp",
         "generator": "g", "code": "def f():\n    return 1", "label": 1,
         "outcome": "pass", "error_type": None}
        for i in range(3)
    ]
    rows = build_rows(labeled, problems, dedup=True)
    assert len(rows) == 1
    assert rows[0]["multiplicity"] == 3
    assert len(build_rows(labeled, problems, dedup=False)) == 3


def test_class_weights_are_computed_and_degenerate_sets_rejected():
    weights = class_weights(make_rows(n_problems=4))
    assert weights["pos_weight"] == pytest.approx(1.0)
    with pytest.raises(ValueError):
        class_weights([{"label": 1}, {"label": 1}])
