"""Transfer to a second generator, plus a probe for whether style is readable."""

from __future__ import annotations

from typing import Any

import numpy as np

import config
from src.baselines import CODE_TOKEN_PATTERN


def generator_fingerprint_probe(rows_a: list[dict[str, Any]],
                                rows_b: list[dict[str, Any]],
                                seed: int | None = None) -> dict[str, Any]:
    """How separable are the two generators' outputs, from code alone?"""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.pipeline import Pipeline

    seed = config.SEED if seed is None else seed
    codes = [r["code"] for r in rows_a] + [r["code"] for r in rows_b]
    y = np.array([0] * len(rows_a) + [1] * len(rows_b))
    groups = ([r["problem_id"] for r in rows_a] + [r["problem_id"] for r in rows_b])

    if len(set(y)) < 2 or len(codes) < 50:
        return {"skipped": "not enough data for the fingerprint probe"}

    # Group split by problem, for the same reason the main split is grouped:
    # otherwise the probe can match a train snippet to its test sibling.
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
    train_index, test_index = next(splitter.split(codes, y, groups))

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(analyzer="word", token_pattern=CODE_TOKEN_PATTERN,
                                  ngram_range=(1, 2), min_df=2, max_features=50_000,
                                  sublinear_tf=True, lowercase=False)),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced",
                                   random_state=seed)),
    ])
    pipeline.fit([codes[i] for i in train_index], y[train_index])
    scores = pipeline.predict_proba([codes[i] for i in test_index])[:, 1]
    auc = float(roc_auc_score(y[test_index], scores))

    if auc > 0.90:
        reading = ("generator identity is trivially readable from the code; "
                   "style is an available shortcut")
    elif auc > 0.70:
        reading = "generator identity is partly readable from the code"
    else:
        reading = ("generators are hard to tell apart from code alone; style "
                   "is unlikely to be a shortcut")

    return {
        "generator_id_auc": round(auc, 4),
        "n_train": int(len(train_index)),
        "n_test": int(len(test_index)),
        "interpretation": reading,
    }


def compare_transfer(metrics_in_distribution: dict[str, Any],
                     metrics_transfer: dict[str, Any],
                     name_in: str, name_transfer: str) -> dict[str, Any]:
    """Summarise the AUC gap between the two generators."""
    auc_in = metrics_in_distribution.get("auc")
    auc_out = metrics_transfer.get("auc")
    if auc_in is None or auc_out is None or not (np.isfinite(auc_in) and np.isfinite(auc_out)):
        return {"skipped": "missing AUC on one side of the comparison"}

    drop = auc_in - auc_out
    # Thresholds are judgement calls, stated explicitly so a reader can
    # disagree with them rather than having to guess what "large" meant.
    if drop >= 0.10:
        reading = ("large drop: the classifier was substantially fitting the "
                   "training generator's distribution")
    elif drop >= 0.05:
        reading = "moderate drop: some generator-specific fitting"
    elif drop >= -0.02:
        reading = "holds up: transfers to an unseen generator with little loss"
    else:
        reading = ("scores HIGHER on the unseen generator; check whether that "
                   "generator's candidates are simply easier to separate")

    return {
        "in_distribution_generator": name_in,
        "transfer_generator": name_transfer,
        "auc_in_distribution": round(float(auc_in), 4),
        "auc_transfer": round(float(auc_out), 4),
        "auc_drop": round(float(drop), 4),
        "pass_rate_in_distribution": metrics_in_distribution.get("positive_rate"),
        "pass_rate_transfer": metrics_transfer.get("positive_rate"),
        "interpretation": reading,
    }
