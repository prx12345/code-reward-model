"""Majority class, TF-IDF + LR, static features + LR. Fit before the transformer."""

from __future__ import annotations

from typing import Any

import numpy as np

import config
from src.features import FEATURE_NAMES, feature_matrix

# Identifiers, numbers, and single punctuation characters. The default sklearn
# token pattern drops every operator and bracket, which throws away most of the
# signal in source code -- `if x = 1` vs `if x == 1` would tokenize identically.
CODE_TOKEN_PATTERN = r"[A-Za-z_][A-Za-z_0-9]*|\d+|[^\sA-Za-z_0-9]"


def _xy(rows: list[dict[str, Any]]) -> tuple[list[str], np.ndarray]:
    return [r["code"] for r in rows], np.array([r["label"] for r in rows], dtype=int)


def fit_majority(train: list[dict[str, Any]],
                 eval_sets: dict[str, list[dict[str, Any]]]) -> dict[str, np.ndarray]:
    """Constant predictor at the training pass rate."""
    _, y_train = _xy(train)
    rate = float(y_train.mean())
    return {name: np.full(len(rows), rate, dtype=float)
            for name, rows in eval_sets.items()}


def fit_tfidf_logreg(train: list[dict[str, Any]],
                     eval_sets: dict[str, list[dict[str, Any]]]
                     ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    x_train, y_train = _xy(train)
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="word",
            token_pattern=CODE_TOKEN_PATTERN,
            ngram_range=(1, 2),
            min_df=2,
            max_features=50_000,
            sublinear_tf=True,
            lowercase=False,   # `Sort` and `sort` are different names in code
        )),
        ("clf", LogisticRegression(
            max_iter=2000,
            C=1.0,
            # Mirrors the pos_weight used by the neural model, so the two are
            # comparing architectures rather than imbalance handling.
            class_weight="balanced",
            random_state=config.SEED,
        )),
    ])
    pipeline.fit(x_train, y_train)

    scores = {name: pipeline.predict_proba([r["code"] for r in rows])[:, 1]
              for name, rows in eval_sets.items()}
    info = {
        "vocabulary_size": len(pipeline.named_steps["tfidf"].vocabulary_),
        "top_positive_tokens": _top_tokens(pipeline, positive=True),
        "top_negative_tokens": _top_tokens(pipeline, positive=False),
    }
    return scores, info


def _top_tokens(pipeline: Any, positive: bool, k: int = 15) -> list[str]:
    """Most influential tokens -- cheap interpretability for the write-up."""
    names = np.array(pipeline.named_steps["tfidf"].get_feature_names_out())
    weights = pipeline.named_steps["clf"].coef_[0]
    order = np.argsort(weights)[::-1] if positive else np.argsort(weights)
    return [str(names[i]) for i in order[:k]]


def fit_static_logreg(train: list[dict[str, Any]],
                      eval_sets: dict[str, list[dict[str, Any]]]
                      ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    x_train, y_train = _xy(train)
    pipeline = Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced",
                                   random_state=config.SEED)),
    ])
    pipeline.fit(np.array(feature_matrix(x_train), dtype=float), y_train)

    scores = {
        name: pipeline.predict_proba(
            np.array(feature_matrix([r["code"] for r in rows]), dtype=float)
        )[:, 1]
        for name, rows in eval_sets.items()
    }
    weights = pipeline.named_steps["clf"].coef_[0]
    info = {
        "coefficients": {name: round(float(w), 4)
                         for name, w in zip(FEATURE_NAMES, weights)},
    }
    return scores, info


def fit_all_baselines(train: list[dict[str, Any]],
                      eval_sets: dict[str, list[dict[str, Any]]]
                      ) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    """Fit every baseline; return ``{model: {split: scores}}`` plus diagnostics."""
    scores: dict[str, dict[str, np.ndarray]] = {}
    info: dict[str, Any] = {}

    scores["majority"] = fit_majority(train, eval_sets)
    scores["tfidf_logreg"], info["tfidf_logreg"] = fit_tfidf_logreg(train, eval_sets)
    scores["static_logreg"], info["static_logreg"] = fit_static_logreg(train, eval_sets)
    return scores, info
