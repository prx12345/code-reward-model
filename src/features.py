"""Cheap static features over code. Baseline #2 — no execution involved."""

from __future__ import annotations

import ast
import re
import warnings
from typing import Any

FEATURE_NAMES: list[str] = [
    "n_chars",
    "n_lines",
    "n_nonblank_lines",
    "parses",
    "n_branches",
    "n_loops",
    "n_functions",
    "n_returns",
    "max_indent",
    "n_comments",
    "has_docstring",
    "n_try",
    "n_imports",
    "n_calls",
    "n_names",
    "is_recursive",
    "has_pass_only_body",
    "n_todo_markers",
]

_COMMENT = re.compile(r"^\s*#", re.MULTILINE)
_TODO = re.compile(r"#\s*(TODO|FIXME|XXX|\.\.\.)", re.IGNORECASE)


def static_features(code: str) -> dict[str, float]:
    """Return a fixed-length feature dict. Never raises on malformed code."""
    lines = code.split("\n")
    features: dict[str, float] = {name: 0.0 for name in FEATURE_NAMES}
    features["n_chars"] = float(len(code))
    features["n_lines"] = float(len(lines))
    features["n_nonblank_lines"] = float(sum(1 for line in lines if line.strip()))
    features["n_comments"] = float(len(_COMMENT.findall(code)))
    features["n_todo_markers"] = float(len(_TODO.findall(code)))
    features["max_indent"] = float(max(
        (len(line) - len(line.lstrip())) for line in lines if line.strip()
    ) if any(line.strip() for line in lines) else 0)

    try:
        # Candidate code is full of regex strings with stray backslashes;
        # ast.parse emits SyntaxWarning for each one. Those are properties of
        # the data, not of our code, and they would drown the run log.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tree = ast.parse(code)
    except (SyntaxError, ValueError, MemoryError, RecursionError):
        # `parses` stays 0. This one feature alone is a real signal: code that
        # does not compile cannot pass, so the baseline gets it for free.
        return features

    features["parses"] = 1.0
    function_names: set[str] = set()
    called_names: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, (ast.If, ast.IfExp)):
            features["n_branches"] += 1
        elif isinstance(node, (ast.For, ast.While, ast.AsyncFor,
                               ast.ListComp, ast.GeneratorExp, ast.DictComp, ast.SetComp)):
            features["n_loops"] += 1
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            features["n_functions"] += 1
            function_names.add(node.name)
            if ast.get_docstring(node):
                features["has_docstring"] = 1.0
            body = node.body
            if len(body) == 1 and isinstance(body[0], (ast.Pass, ast.Expr)):
                if isinstance(body[0], ast.Pass) or isinstance(
                    getattr(body[0], "value", None), ast.Constant
                ):
                    features["has_pass_only_body"] = 1.0
        elif isinstance(node, ast.Return):
            features["n_returns"] += 1
        elif isinstance(node, (ast.Try, ast.ExceptHandler)):
            features["n_try"] += 1
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            features["n_imports"] += 1
        elif isinstance(node, ast.Call):
            features["n_calls"] += 1
            if isinstance(node.func, ast.Name):
                called_names.append(node.func.id)
        elif isinstance(node, ast.Name):
            features["n_names"] += 1

    features["is_recursive"] = float(bool(function_names & set(called_names)))
    return features


def feature_vector(code: str) -> list[float]:
    feats = static_features(code)
    return [feats[name] for name in FEATURE_NAMES]


def feature_matrix(codes: list[str]) -> list[list[float]]:
    return [feature_vector(code) for code in codes]


def describe(code: str) -> dict[str, Any]:
    """Human-readable feature dump, used by the error-analysis report."""
    return {k: (int(v) if float(v).is_integer() else v)
            for k, v in static_features(code).items()}
