"""Data pipeline: normalization, entry-point inference, and the HumanEval guard."""

import pytest

from src.data import (LeakageError, assert_humaneval_never_trained,
                      dataset_stats, infer_entry_point, normalize_humaneval,
                      normalize_mbpp)
from src.label import assemble_solution, humaneval_preamble

RAW_MBPP = [
    {"task_id": 3, "text": "few-shot example", "code": "def a(): pass",
     "test_setup_code": "", "test_list": ["assert a() is None"], "challenge_test_list": []},
    {"task_id": 101, "text": "Find the kth element.",
     "code": "def kth_element(arr, n, k):\r\n  return arr[k-1]",
     "test_setup_code": "", "challenge_test_list": [],
     "test_list": ["assert kth_element([12,3,5], 3, 2) == 3",
                   "assert kth_element([1,2], 2, 1) == 1"]},
]

RAW_HUMANEVAL = [
    {"task_id": "HumanEval/0", "entry_point": "has_close",
     "prompt": "from typing import List\n\n\ndef has_close(x: List[float]) -> bool:\n    \"\"\"doc\"\"\"\n",
     "canonical_solution": "    return False\n",
     "test": "def check(candidate):\n    assert candidate([1.0]) is False\n"},
]


def test_mbpp_few_shot_ids_are_excluded():
    problems = normalize_mbpp(RAW_MBPP)
    assert [p["problem_id"] for p in problems] == ["mbpp/101"]


def test_mbpp_crlf_is_normalized():
    problem = normalize_mbpp(RAW_MBPP)[0]
    assert "\r" not in problem["reference_solution"]


def test_generation_prompt_contains_tests_and_classifier_prompt_does_not():
    """The core anti-leak decision: the label source must not reach the model."""
    problem = normalize_mbpp(RAW_MBPP)[0]
    assert "assert kth_element" in problem["generation_prompt"]
    assert "assert" not in problem["classifier_prompt"]
    assert problem["classifier_prompt"] == "Find the kth element."


@pytest.mark.parametrize("statement,expected", [
    ("assert kth_element([1], 1, 1) == 1", "kth_element"),
    ("assert math.isclose(area_circle(2), 12.56, rel_tol=0.001)", "area_circle"),
    ("assert set(unique([1,1,2])) == {1,2}", "unique"),
    ("assert len(flatten([[1]])) == 1", "flatten"),
])
def test_entry_point_inference_sees_through_wrappers(statement, expected):
    assert infer_entry_point([statement]) == expected


def test_entry_point_inference_survives_unparseable_tests():
    assert infer_entry_point(["assert ((("]) is None


def test_humaneval_normalization_shape():
    problem = normalize_humaneval(RAW_HUMANEVAL)[0]
    assert problem["problem_id"] == "humaneval/HumanEval_0"
    assert problem["tests"] == ["check(has_close)"]
    assert "def check" in problem["test_defs"]
    assert problem["prompt_kind"] == "signature"


def test_humaneval_preamble_captures_imports_only():
    stub = RAW_HUMANEVAL[0]["prompt"]
    preamble = humaneval_preamble(stub)
    assert "from typing import List" in preamble
    assert "def has_close" not in preamble


def test_assemble_solution_restores_missing_imports_for_humaneval():
    problem = normalize_humaneval(RAW_HUMANEVAL)[0]
    candidate = "def has_close(x: List[float]) -> bool:\n    return False"
    assembled = assemble_solution(problem, candidate)
    assert "from typing import List" in assembled
    assert assembled.count("def has_close") == 1


def test_assemble_solution_leaves_mbpp_candidates_untouched():
    problem = normalize_mbpp(RAW_MBPP)[0]
    assert assemble_solution(problem, "def f(): pass") == "def f(): pass"


def test_humaneval_guard_raises_on_any_leak():
    with pytest.raises(LeakageError):
        assert_humaneval_never_trained(["mbpp/1", "humaneval/HumanEval_0"])


def test_humaneval_guard_passes_on_clean_ids():
    assert_humaneval_never_trained(["mbpp/1", "mbpp/2"])


def test_dataset_stats_on_empty_input():
    assert dataset_stats([])["n_problems"] == 0
