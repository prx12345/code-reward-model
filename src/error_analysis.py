"""Does it catch obvious breakage but miss logic bugs? Two-proportion z-test."""

from __future__ import annotations

import collections
import math
from typing import Any, Sequence

import numpy as np

VISIBLE_BREAKAGE = ("syntax_error", "exception", "timeout", "memory_exceeded")
SUBTLE_FAILURE = ("assertion_failure",)


def _two_proportion_z(k1: int, n1: int, k2: int, n2: int) -> tuple[float, float]:
    """Return (z, two-sided p) for H0: p1 == p2. Normal approximation."""
    if n1 == 0 or n2 == 0:
        return float("nan"), float("nan")
    p1, p2 = k1 / n1, k2 / n2
    pooled = (k1 + k2) / (n1 + n2)
    denominator = math.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    if denominator == 0:
        return float("nan"), float("nan")
    z = (p1 - p2) / denominator
    p = 2 * (1 - _normal_cdf(abs(z)))
    return z, p


def _normal_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def detection_by_outcome(rows: Sequence[dict[str, Any]], scores: Sequence[float],
                         threshold: float) -> dict[str, Any]:
    """For each ground-truth outcome, how often does the model call it correctly?"""
    scores = np.asarray(scores, dtype=float)
    buckets: dict[str, list[tuple[int, float]]] = collections.defaultdict(list)
    for row, score in zip(rows, scores):
        buckets[row["outcome"]].append((int(row["label"]), float(score)))

    report: dict[str, Any] = {}
    for outcome, items in sorted(buckets.items()):
        labels = np.array([label for label, _ in items])
        values = np.array([score for _, score in items])
        predicted_pass = values >= threshold
        correct = predicted_pass == labels.astype(bool)
        report[outcome] = {
            "n": len(items),
            "mean_score": round(float(values.mean()), 4),
            "median_score": round(float(np.median(values)), 4),
            "correctly_classified": int(correct.sum()),
            "detection_rate": round(float(correct.mean()), 4),
        }
    return report


def test_hypothesis(rows: Sequence[dict[str, Any]], scores: Sequence[float],
                    threshold: float) -> dict[str, Any]:
    """Visible-breakage vs wrong-answer detection rate among true failures."""
    scores = np.asarray(scores, dtype=float)
    visible = [(r, s) for r, s in zip(rows, scores)
               if r["label"] == 0 and r["outcome"] in VISIBLE_BREAKAGE]
    subtle = [(r, s) for r, s in zip(rows, scores)
              if r["label"] == 0 and r["outcome"] in SUBTLE_FAILURE]

    caught_visible = sum(1 for _, s in visible if s < threshold)
    caught_subtle = sum(1 for _, s in subtle if s < threshold)
    z, p = _two_proportion_z(caught_visible, len(visible), caught_subtle, len(subtle))

    rate_visible = caught_visible / len(visible) if visible else float("nan")
    rate_subtle = caught_subtle / len(subtle) if subtle else float("nan")
    gap = rate_visible - rate_subtle

    if not math.isfinite(gap):
        verdict = "inconclusive: one of the buckets is empty"
    elif p < 0.05 and gap > 0:
        verdict = ("SUPPORTED: visible breakage is caught significantly more "
                   "often than wrong answers")
    elif p < 0.05 and gap < 0:
        verdict = ("CONTRADICTED: wrong answers are caught MORE often than "
                   "visible breakage")
    else:
        verdict = ("NOT SUPPORTED: no significant difference between the two "
                   "failure types at p<0.05")

    return {
        "hypothesis": ("classifier catches obvious breakage, misses subtle "
                       "logic bugs"),
        "visible_breakage_outcomes": list(VISIBLE_BREAKAGE),
        "subtle_failure_outcomes": list(SUBTLE_FAILURE),
        "n_visible_breakage": len(visible),
        "n_subtle_failures": len(subtle),
        "recall_visible_breakage": round(rate_visible, 4) if math.isfinite(rate_visible) else None,
        "recall_subtle_failures": round(rate_subtle, 4) if math.isfinite(rate_subtle) else None,
        "gap": round(gap, 4) if math.isfinite(gap) else None,
        "z": round(z, 3) if math.isfinite(z) else None,
        "p_value": round(p, 6) if math.isfinite(p) else None,
        "verdict": verdict,
    }


def length_confound(rows: Sequence[dict[str, Any]],
                    scores: Sequence[float]) -> dict[str, Any]:
    """Is the model just reading code length?

    A strong length-score relationship would mean this is a length heuristic
    with extra steps.
    """
    from scipy.stats import spearmanr

    lengths = np.array([len(r["code"]) for r in rows], dtype=float)
    scores = np.asarray(scores, dtype=float)
    labels = np.array([r["label"] for r in rows], dtype=int)
    rho_score, p_score = spearmanr(lengths, scores)
    rho_label, p_label = spearmanr(lengths, labels)
    return {
        "spearman_length_vs_score": round(float(rho_score), 4),
        "p_length_vs_score": round(float(p_score), 6),
        "spearman_length_vs_label": round(float(rho_label), 4),
        "p_length_vs_label": round(float(p_label), 6),
        "note": ("If length-vs-score is much stronger than length-vs-label, the "
                 "model is leaning on a heuristic the data does not support."),
    }


def confident_mistakes(rows: Sequence[dict[str, Any]], scores: Sequence[float],
                       threshold: float, k: int = 5) -> dict[str, list[dict[str, Any]]]:
    """The k most confident false positives and false negatives, with code."""
    scores = np.asarray(scores, dtype=float)
    pairs = list(zip(rows, scores))

    false_positives = sorted(
        [(r, s) for r, s in pairs if r["label"] == 0 and s >= threshold],
        key=lambda item: -item[1])[:k]
    false_negatives = sorted(
        [(r, s) for r, s in pairs if r["label"] == 1 and s < threshold],
        key=lambda item: item[1])[:k]
    true_positives = sorted(
        [(r, s) for r, s in pairs if r["label"] == 1 and s >= threshold],
        key=lambda item: -item[1])[:k]
    true_negatives = sorted(
        [(r, s) for r, s in pairs if r["label"] == 0 and s < threshold],
        key=lambda item: item[1])[:k]

    def pack(items: list[tuple[dict[str, Any], float]]) -> list[dict[str, Any]]:
        return [{
            "candidate_id": r["candidate_id"],
            "problem_id": r["problem_id"],
            "problem_text": r["problem_text"][:300],
            "score": round(float(s), 4),
            "label": r["label"],
            "outcome": r["outcome"],
            "error_type": r.get("error_type"),
            "code": r["code"][:1500],
        } for r, s in items]

    return {
        "false_positives": pack(false_positives),
        "false_negatives": pack(false_negatives),
        "true_positives": pack(true_positives),
        "true_negatives": pack(true_negatives),
    }


def render_examples_markdown(examples: dict[str, list[dict[str, Any]]]) -> str:
    titles = {
        "false_positives": "False positives — model said PASS, code actually failed",
        "false_negatives": "False negatives — model said FAIL, code actually passed",
        "true_positives": "True positives — confidently and correctly called PASS",
        "true_negatives": "True negatives — confidently and correctly called FAIL",
    }
    lines = ["# Error analysis examples", ""]
    for key, title in titles.items():
        lines += [f"## {title}", ""]
        items = examples.get(key) or []
        if not items:
            lines += ["_none_", ""]
            continue
        for item in items:
            lines += [
                f"### `{item['candidate_id']}` — score {item['score']}, "
                f"outcome `{item['outcome']}`",
                "",
                f"**Problem:** {item['problem_text']}",
                "",
                "```python",
                item["code"],
                "```",
                "",
            ]
    return "\n".join(lines)


def full_report(rows: Sequence[dict[str, Any]], scores: Sequence[float],
                threshold: float) -> dict[str, Any]:
    return {
        "threshold": round(float(threshold), 4),
        "detection_by_outcome": detection_by_outcome(rows, scores, threshold),
        "hypothesis_test": test_hypothesis(rows, scores, threshold),
        "length_confound": length_confound(rows, scores),
    }
