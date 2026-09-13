"""Static features, code extraction from model replies, and long-code truncation."""

import pytest

from src.classifier import head_tail_truncate
from src.evaluate import best_f1_threshold, compute_metrics, expected_calibration_error
from src.features import FEATURE_NAMES, feature_vector, static_features
from src.generate import extract_code


# --------------------------------------------------------------- features

def test_feature_vector_is_fixed_length_even_for_garbage():
    assert len(feature_vector("def f( syntax error")) == len(FEATURE_NAMES)
    assert len(feature_vector("")) == len(FEATURE_NAMES)


def test_parses_flag_separates_valid_from_invalid_code():
    assert static_features("def f():\n    return 1")["parses"] == 1.0
    assert static_features("def f(: return")["parses"] == 0.0


def test_structural_counts():
    code = ("def f(n):\n"
            "    if n > 0:\n"
            "        for i in range(n):\n"
            "            pass\n"
            "        return f(n - 1)\n"
            "    return 0\n")
    feats = static_features(code)
    assert feats["n_branches"] == 1
    assert feats["n_loops"] == 1
    assert feats["n_returns"] == 2
    assert feats["is_recursive"] == 1.0


def test_features_do_not_crash_on_regex_heavy_code():
    """Candidate code is full of stray backslashes; SyntaxWarning must not escape."""
    assert static_features("import re\ndef f(s):\n    return re.match('\\w+', s)")["parses"] == 1.0


# --------------------------------------------------------- code extraction

def test_extracts_fenced_python_block():
    reply = "Here you go:\n```python\ndef f():\n    return 1\n```\nHope that helps!"
    assert extract_code(reply) == "def f():\n    return 1"


def test_takes_the_last_block_when_the_model_sketches_first():
    reply = "```python\nwrong\n```\nActually:\n```python\ndef f():\n    return 2\n```"
    assert extract_code(reply) == "def f():\n    return 2"


def test_recovers_from_a_fence_truncated_by_max_new_tokens():
    reply = "Sure.\n```python\ndef f():\n    return 1"
    assert extract_code(reply) == "def f():\n    return 1"


def test_handles_a_reply_with_no_fence_at_all():
    reply = "def f():\n    return 1\nThis returns one."
    assert extract_code(reply).startswith("def f():")
    assert "This returns one." not in extract_code(reply)


def test_prose_only_reply_is_returned_as_is_and_will_be_labeled_syntax_error():
    """Deliberate: 'the model produced no code' is a real failure, not a crash."""
    assert extract_code("I am not sure how to do this.").strip()


# ------------------------------------------------------------- truncation

def test_head_tail_keeps_both_ends():
    ids = list(range(100))
    kept, truncated = head_tail_truncate(ids, budget=10, head_fraction=0.6)
    assert truncated is True
    assert len(kept) == 10
    assert kept[:6] == [0, 1, 2, 3, 4, 5]
    assert kept[6:] == [96, 97, 98, 99]


def test_short_sequences_are_untouched():
    ids = [1, 2, 3]
    kept, truncated = head_tail_truncate(ids, budget=10)
    assert kept == ids and truncated is False


# ---------------------------------------------------------------- metrics

def test_perfect_scores_give_auc_one_and_constant_scores_give_half():
    perfect = compute_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9], 0.5)
    assert perfect["auc"] == 1.0
    constant = compute_metrics([0, 0, 1, 1], [0.5, 0.5, 0.5, 0.5], 0.5)
    assert constant["auc"] == 0.5


def test_confusion_counts_add_up():
    metrics = compute_metrics([0, 0, 1, 1], [0.9, 0.1, 0.9, 0.1], 0.5)
    confusion = metrics["confusion"]
    assert sum(confusion.values()) == 4
    assert confusion["fp"] == 1 and confusion["fn"] == 1


def test_single_class_split_does_not_crash():
    metrics = compute_metrics([1, 1, 1], [0.2, 0.5, 0.9], 0.5)
    assert metrics["auc"] != metrics["auc"]  # NaN


def test_threshold_search_returns_a_usable_threshold():
    threshold = best_f1_threshold([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert 0.0 <= threshold <= 1.0


def test_ece_is_zero_for_a_perfectly_calibrated_predictor():
    import numpy as np
    y = np.array([0] * 50 + [1] * 50)
    probs = np.array([0.0] * 50 + [1.0] * 50)
    assert expected_calibration_error(y, probs) == pytest.approx(0.0, abs=1e-9)
