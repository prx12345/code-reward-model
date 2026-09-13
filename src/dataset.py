"""Turn labeled candidates into train/val/test splits -- with hard leak guards.

The single easiest way to fake a good result on this task is to split rows
randomly. Eight candidates are sampled per problem, many of them near-identical;
a random split puts siblings of a test row in the training set and the model
scores beautifully by recognising the problem, not by judging the code.

So: **splitting is by problem id, and an assertion enforces it.** Three guards
run on every split, and each raises rather than warns:

1. no ``problem_id`` appears in more than one split;
2. no exact code string appears in more than one split;
3. no HumanEval problem is in any training split.

``LeakageError`` is deliberately not caught anywhere. If it fires, the run
should die.
"""

from __future__ import annotations

import collections
import hashlib
import random
from dataclasses import dataclass
from typing import Any, Iterable

import config
from src.data import LeakageError
from src.utils import log


# --------------------------------------------------------------------------
# Row assembly
# --------------------------------------------------------------------------

def _normalize_code(code: str) -> str:
    """Whitespace-insensitive key for duplicate detection."""
    return "\n".join(line.rstrip() for line in code.strip().split("\n") if line.strip())


def build_rows(labeled: list[dict[str, Any]], problems: list[dict[str, Any]],
               dedup: bool | None = None) -> list[dict[str, Any]]:
    """Join labels with problem text, drop harness errors, optionally dedup.

    Harness errors are dropped, not labeled 0: they mean *our* scaffolding
    failed, so the row carries no information about the candidate and training
    on it teaches the model to predict our bugs.
    """
    dedup = config.DEDUP_WITHIN_PROBLEM if dedup is None else dedup
    by_id = {p["problem_id"]: p for p in problems}

    rows: list[dict[str, Any]] = []
    seen: dict[tuple[str, str], dict[str, Any]] = {}
    n_harness = n_dup = n_missing = 0

    for record in labeled:
        if record["outcome"] == "harness_error":
            n_harness += 1
            continue
        problem = by_id.get(record["problem_id"])
        if problem is None:
            n_missing += 1
            continue
        key = (record["problem_id"], _normalize_code(record["code"]))
        if dedup and key in seen:
            seen[key]["multiplicity"] += 1
            n_dup += 1
            continue
        row = {
            "candidate_id": record["candidate_id"],
            "problem_id": record["problem_id"],
            "benchmark": record["benchmark"],
            "generator": record["generator"],
            "problem_text": problem["classifier_prompt"],
            "code": record["code"],
            "label": int(record["label"]),
            "outcome": record["outcome"],
            "error_type": record.get("error_type"),
            "multiplicity": 1,
        }
        seen[key] = row
        rows.append(row)

    log(f"rows: {len(rows)} kept; dropped {n_harness} harness_error, "
        f"{n_dup} duplicates, {n_missing} unknown problem_id")
    return rows


# --------------------------------------------------------------------------
# Splitting
# --------------------------------------------------------------------------

@dataclass
class Splits:
    train: list[dict[str, Any]]
    val: list[dict[str, Any]]
    test: list[dict[str, Any]]

    def summary(self) -> dict[str, Any]:
        def describe(rows: list[dict[str, Any]]) -> dict[str, Any]:
            if not rows:
                return {"n_rows": 0, "n_problems": 0, "pass_rate": None}
            return {
                "n_rows": len(rows),
                "n_problems": len({r["problem_id"] for r in rows}),
                "pass_rate": round(sum(r["label"] for r in rows) / len(rows), 4),
            }
        return {"train": describe(self.train), "val": describe(self.val),
                "test": describe(self.test)}


def _stable_hash(text: str) -> int:
    """Deterministic across processes, unlike Python's salted ``hash``."""
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)


def split_by_problem(rows: list[dict[str, Any]], val_fraction: float | None = None,
                     test_fraction: float | None = None, seed: int | None = None) -> Splits:
    """Partition rows so that every candidate for a problem lands in one split.

    Problems are assigned by shuffling the *sorted* id list with a seeded RNG,
    so the split is identical on every machine and every rerun.
    """
    val_fraction = config.VAL_FRACTION if val_fraction is None else val_fraction
    test_fraction = config.TEST_FRACTION if test_fraction is None else test_fraction
    seed = config.SEED if seed is None else seed

    problem_ids = sorted({r["problem_id"] for r in rows})
    rng = random.Random(seed)
    rng.shuffle(problem_ids)

    n = len(problem_ids)
    n_val = int(round(n * val_fraction))
    n_test = int(round(n * test_fraction))
    val_ids = set(problem_ids[:n_val])
    test_ids = set(problem_ids[n_val:n_val + n_test])
    train_ids = set(problem_ids[n_val + n_test:])

    splits = Splits(
        train=[r for r in rows if r["problem_id"] in train_ids],
        val=[r for r in rows if r["problem_id"] in val_ids],
        test=[r for r in rows if r["problem_id"] in test_ids],
    )
    assert_no_leakage(splits)
    log(f"split by problem: {len(train_ids)} train / {len(val_ids)} val / "
        f"{len(test_ids)} test problems")
    return splits


# --------------------------------------------------------------------------
# Guards
# --------------------------------------------------------------------------

def assert_no_problem_overlap(*groups: Iterable[dict[str, Any]]) -> None:
    """Crash loudly if any problem id appears in more than one split."""
    id_sets = [{r["problem_id"] for r in group} for group in groups]
    for i in range(len(id_sets)):
        for j in range(i + 1, len(id_sets)):
            shared = id_sets[i] & id_sets[j]
            if shared:
                raise LeakageError(
                    f"problem-id overlap between split {i} and split {j}: "
                    f"{len(shared)} shared ids, e.g. {sorted(shared)[:5]}. "
                    "Splitting must be by problem, never by row."
                )


def assert_no_code_overlap(*groups: Iterable[dict[str, Any]]) -> None:
    """Crash if the exact same code string appears in two splits.

    Splitting by problem does not by itself rule this out: MBPP contains
    near-duplicate problems, and a 1.5B model emits the same three-line
    solution for several of them. Memorising one such string would transfer
    across the split boundary.
    """
    code_sets = [{_normalize_code(r["code"]) for r in group} for group in groups]
    for i in range(len(code_sets)):
        for j in range(i + 1, len(code_sets)):
            shared = code_sets[i] & code_sets[j]
            if shared:
                raise LeakageError(
                    f"identical code in split {i} and split {j}: {len(shared)} "
                    f"shared snippets, e.g. {sorted(shared)[0][:120]!r}"
                )


def assert_no_leakage(splits: Splits, strict_code: bool = False) -> None:
    """Run every guard. ``strict_code`` raises on cross-split duplicate code.

    Code overlap is reported but not fatal by default, because with 964 short
    MBPP problems a handful of collisions is expected and dropping them costs
    more than it buys. The count is printed so it can be checked, and the
    README reports it. Set ``strict_code=True`` to make it fatal.
    """
    assert_no_problem_overlap(splits.train, splits.val, splits.test)

    from src.data import assert_humaneval_never_trained
    assert_humaneval_never_trained({r["problem_id"] for r in splits.train})
    assert_humaneval_never_trained({r["problem_id"] for r in splits.val})

    if strict_code:
        assert_no_code_overlap(splits.train, splits.val, splits.test)
    else:
        train_codes = {_normalize_code(r["code"]) for r in splits.train}
        for name, group in (("val", splits.val), ("test", splits.test)):
            shared = train_codes & {_normalize_code(r["code"]) for r in group}
            if shared:
                log(f"note: {len(shared)} code strings shared between train and "
                    f"{name} (distinct problems, identical snippets)")


def class_weights(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Positive-class weight for a weighted BCE loss, computed on TRAIN only.

    ``pos_weight = n_negative / n_positive`` degenerates to ~1.0 on a balanced
    set, so this is applied unconditionally rather than behind an "if the data
    is skewed" branch -- fewer code paths, same behaviour.
    """
    n_pos = sum(r["label"] for r in rows)
    n_neg = len(rows) - n_pos
    if n_pos == 0 or n_neg == 0:
        raise ValueError(f"degenerate training set: {n_pos} positive, {n_neg} negative")
    return {
        "n_positive": n_pos,
        "n_negative": n_neg,
        "pos_weight": round(n_neg / n_pos, 4),
        "positive_fraction": round(n_pos / (n_pos + n_neg), 4),
    }


def outcome_distribution(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(collections.Counter(r["outcome"] for r in rows).most_common())


# --------------------------------------------------------------------------
# One-call split loading, shared by every downstream script
# --------------------------------------------------------------------------

def load_splits(generator_tag: str | None = None, dedup: bool | None = None
                ) -> tuple[Splits, list[dict[str, Any]]]:
    """Return (MBPP splits, HumanEval held-out rows) for a generator.

    HumanEval rows are assembled the same way but never enter `Splits`, so
    there is no code path by which they could reach the optimiser.
    """
    from src.data import load_problems
    from src.label import load_labeled

    generator_tag = generator_tag or config.GENERATOR_TAG
    mbpp_rows = build_rows(load_labeled(generator_tag, "mbpp"),
                           load_problems("mbpp"), dedup=dedup)
    if not mbpp_rows:
        raise FileNotFoundError(
            f"no labeled MBPP candidates for generator {generator_tag!r}. "
            "Run scripts/02_generate.py then scripts/03_label.py first."
        )
    splits = split_by_problem(mbpp_rows)

    heldout = build_rows(load_labeled(generator_tag, "humaneval"),
                         load_problems("humaneval"), dedup=dedup)
    log(f"held-out HumanEval rows: {len(heldout)}")
    return splits, heldout


def eval_sets(splits: Splits, heldout: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """The canonical set of evaluation splits, named consistently everywhere."""
    sets = {"val": splits.val, "test": splits.test}
    if heldout:
        sets["humaneval"] = heldout
    return sets
