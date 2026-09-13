"""Metrics, threshold selection, the too-good-to-be-true guard, and plots.

Every model in this repo -- majority, TF-IDF, static features, CodeBERT -- is
scored through this one module, on the same splits, with the same threshold
rule. That is the only way the comparison table means anything.

Threshold rule: the decision threshold is chosen on **validation** by maximising
F1 and then applied unchanged to test and to the held-out set. Picking the
threshold on the set you report is a small, extremely common, and entirely
real form of leakage.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

import config


class SuspiciousResultError(RuntimeError):
    """Raised when a score is high enough that a leak is the likelier story."""


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def expected_calibration_error(y_true: np.ndarray, probs: np.ndarray,
                               n_bins: int | None = None) -> float:
    """Equal-width-bin ECE: mean |confidence - accuracy| weighted by bin size."""
    n_bins = n_bins or config.CALIBRATION_BINS
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        mask = (probs > lower) & (probs <= upper) if lower > 0 else (probs >= lower) & (probs <= upper)
        if not mask.any():
            continue
        total += mask.mean() * abs(probs[mask].mean() - y_true[mask].mean())
    return float(total)


def compute_metrics(y_true: Sequence[int], scores: Sequence[float],
                    threshold: float = 0.5) -> dict[str, Any]:
    """Full metric set for one model on one split."""
    from sklearn.metrics import (accuracy_score, average_precision_score,
                                 brier_score_loss, confusion_matrix, f1_score,
                                 precision_score, recall_score, roc_auc_score)

    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    y_pred = (scores >= threshold).astype(int)

    single_class = len(np.unique(y_true)) < 2
    auc = float("nan") if single_class else float(roc_auc_score(y_true, scores))
    ap = float("nan") if single_class else float(average_precision_score(y_true, scores))

    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()

    return {
        "n": int(len(y_true)),
        "positive_rate": round(float(y_true.mean()), 4),
        "auc": round(auc, 4),
        "average_precision": round(ap, 4),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "brier": round(float(brier_score_loss(y_true, np.clip(scores, 0, 1))), 4),
        "ece": round(expected_calibration_error(y_true, np.clip(scores, 0, 1)), 4),
        "threshold": round(float(threshold), 4),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def best_f1_threshold(y_true: Sequence[int], scores: Sequence[float]) -> float:
    """Threshold maximising F1. Must only ever be called on validation data."""
    from sklearn.metrics import precision_recall_curve

    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if len(np.unique(y_true)) < 2:
        return 0.5
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = np.nan_to_num(2 * precision * recall / (precision + recall))
    # precision_recall_curve returns one more p/r point than thresholds.
    index = int(np.argmax(f1[:-1])) if len(thresholds) else 0
    return float(thresholds[index]) if len(thresholds) else 0.5


def check_suspicious(metrics: dict[str, Any], model_name: str, split: str,
                     allow: bool = False) -> None:
    """Hard-stop on implausibly good results.

    A held-out AUC above ~0.95 on this task would beat published code-reward
    models trained on far more data. The overwhelmingly likely explanation is a
    leak, so the pipeline refuses to report the number until someone has looked.
    """
    auc = metrics.get("auc")
    if auc is None or not np.isfinite(auc) or auc < config.SUSPICIOUS_AUC:
        return
    message = (
        f"\n{'!' * 70}\n"
        f"SUSPICIOUS RESULT: {model_name} scored AUC={auc:.4f} on {split}.\n"
        f"Anything above {config.SUSPICIOUS_AUC} on this task is more likely a\n"
        f"leak than a result. Check, in this order:\n"
        f"  1. Did the split go by problem_id? (src/dataset.assert_no_leakage)\n"
        f"  2. Is the problem statement leaking the tests into the input?\n"
        f"     (MBPP generation prompts contain the asserts; the CLASSIFIER\n"
        f"      prompt must be the bare task text -- see src/data.py)\n"
        f"  3. Was the threshold tuned on the split being reported?\n"
        f"  4. Are duplicate candidates spanning the split boundary?\n"
        f"  5. Is the label derivable from a formatting artefact of one\n"
        f"     generator? (run the generalization check)\n"
        f"{'!' * 70}\n"
    )
    if allow:
        print(message)
        return
    raise SuspiciousResultError(message)


# --------------------------------------------------------------------------
# Comparison table
# --------------------------------------------------------------------------

_TABLE_COLUMNS = ["auc", "average_precision", "precision", "recall", "f1",
                  "accuracy", "brier", "ece", "n"]


def comparison_table(results: dict[str, dict[str, Any]], split: str) -> str:
    """Markdown table of every model's metrics on one split."""
    header = "| Model | " + " | ".join(c.upper() if c in {"auc", "ece"} else
                                       c.replace("_", " ").title()
                                       for c in _TABLE_COLUMNS) + " |"
    divider = "|" + "---|" * (len(_TABLE_COLUMNS) + 1)
    lines = [f"**{split}**", "", header, divider]
    for name, per_split in results.items():
        metrics = per_split.get(split)
        if not metrics:
            continue
        cells = []
        for column in _TABLE_COLUMNS:
            value = metrics.get(column)
            if isinstance(value, float):
                cells.append("n/a" if not np.isfinite(value) else f"{value:.4f}")
            else:
                cells.append(str(value))
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------

def _figure_path(name: str) -> Path:
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return config.RESULTS_DIR / name


def plot_roc(curves: dict[str, tuple[Sequence[int], Sequence[float]]],
             title: str, filename: str) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_auc_score, roc_curve

    fig, ax = plt.subplots(figsize=(6, 5))
    for name, (y_true, scores) in curves.items():
        y_true = np.asarray(y_true, dtype=int)
        scores = np.asarray(scores, dtype=float)
        if len(np.unique(y_true)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_true, scores)
        ax.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_true, scores):.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    path = _figure_path(filename)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_confusion(metrics: dict[str, Any], title: str, filename: str) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    confusion = metrics["confusion"]
    matrix = np.array([[confusion["tn"], confusion["fp"]],
                       [confusion["fn"], confusion["tp"]]])
    fig, ax = plt.subplots(figsize=(4.2, 4))
    ax.imshow(matrix, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center",
                    color="black", fontsize=13)
    ax.set_xticks([0, 1], ["pred fail", "pred pass"])
    ax.set_yticks([0, 1], ["true fail", "true pass"])
    ax.set_title(f"{title}\n(threshold={metrics['threshold']:.3f})", fontsize=10)
    fig.tight_layout()
    path = _figure_path(filename)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_calibration(curves: dict[str, tuple[Sequence[int], Sequence[float]]],
                     title: str, filename: str) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.calibration import calibration_curve

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="perfectly calibrated")
    for name, (y_true, scores) in curves.items():
        y_true = np.asarray(y_true, dtype=int)
        scores = np.clip(np.asarray(scores, dtype=float), 0, 1)
        if len(np.unique(y_true)) < 2 or len(np.unique(scores)) < 3:
            continue
        n_bins = min(config.CALIBRATION_BINS, max(2, len(np.unique(scores)) - 1))
        true_frequency, predicted = calibration_curve(y_true, scores, n_bins=n_bins,
                                                      strategy="quantile")
        ax.plot(predicted, true_frequency, "o-", label=name, markersize=4)
    ax.set_xlabel("Predicted probability of passing")
    ax.set_ylabel("Observed pass rate")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    path = _figure_path(filename)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_score_histogram(y_true: Sequence[int], scores: Sequence[float],
                         title: str, filename: str) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    fig, ax = plt.subplots(figsize=(6, 4))
    bins = np.linspace(min(scores.min(), 0), max(scores.max(), 1), 30)
    ax.hist(scores[y_true == 0], bins=bins, alpha=0.6, label="truly fails")
    ax.hist(scores[y_true == 1], bins=bins, alpha=0.6, label="truly passes")
    ax.set_xlabel("Model score")
    ax.set_ylabel("Candidates")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = _figure_path(filename)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------
# Score persistence
# --------------------------------------------------------------------------

SCORES_DIR = config.RESULTS_DIR / "scores"


def save_scores(model_name: str, scores_by_split: dict[str, Sequence[float]],
                rows_by_split: dict[str, list[dict[str, Any]]]) -> Path:
    """Persist per-candidate scores so evaluation is decoupled from training.

    Candidate ids are stored alongside the scores: aligning by position across
    separately-run scripts is exactly the kind of silent bug that produces a
    beautiful, wrong table.
    """
    import json

    SCORES_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        split: {
            "candidate_ids": [r["candidate_id"] for r in rows_by_split[split]],
            "scores": [float(s) for s in scores_by_split[split]],
        }
        for split in scores_by_split
        if split in rows_by_split
    }
    path = SCORES_DIR / f"{model_name}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def load_scores(model_name: str) -> dict[str, dict[str, list[Any]]]:
    import json

    path = SCORES_DIR / f"{model_name}.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def align_scores(rows: list[dict[str, Any]], stored: dict[str, list[Any]]
                 ) -> np.ndarray | None:
    """Reorder stored scores to match *rows*, or return None on any mismatch."""
    lookup = dict(zip(stored["candidate_ids"], stored["scores"]))
    try:
        return np.array([lookup[r["candidate_id"]] for r in rows], dtype=float)
    except KeyError:
        return None
