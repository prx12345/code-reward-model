"""Stage 3 -- execute every candidate against the real tests to produce labels.

This is the part that makes the dataset ours. Everything upstream is public;
the ``(problem, code) -> did it actually pass`` mapping is produced here by
running the code.

Properties that matter:

* **Resumable.** Results are appended to JSONL one candidate at a time and
  keyed by ``candidate_id``. Restarting skips whatever is already on disk.
* **Parallel but still isolated.** Workers are threads, but each thread's work
  is a *separate sandboxed subprocess*, so parallelism does not weaken
  isolation -- it just overlaps the interpreter start-up cost.
* **Typed outcomes.** ``pass`` / ``assertion_failure`` / ``exception`` /
  ``timeout`` / ``syntax_error`` / ``memory_exceeded`` / ``harness_error``.
  The binary label is ``outcome == "pass"``; the outcome itself is what the
  error analysis needs.
"""

from __future__ import annotations

import collections
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable

import config
from src.sandbox import SandboxLimits, ScriptSandbox
from src.utils import JsonlAppender, done_ids, log, progress, read_jsonl

_DEF_AT_COL0 = re.compile(r"^(?:def|class|@)\s", re.MULTILINE)


def _make_sandbox() -> ScriptSandbox:
    return ScriptSandbox(SandboxLimits(
        wall_time_s=config.EXEC_TIMEOUT_S,
        cpu_time_s=config.EXEC_CPU_TIME_S,
        memory_mb=config.EXEC_MEMORY_MB,
    ))


def humaneval_preamble(stub: str) -> str:
    """Imports and constants that precede the function in a HumanEval stub.

    HumanEval prompts open with things like ``from typing import List``. A
    model asked for "the complete function" usually, but not always, repeats
    them. Prepending the preamble makes a candidate that omitted an import a
    *correct* candidate instead of a spurious ``NameError``, which is the
    difference between labeling code quality and labeling prompt compliance.
    Duplicate imports are harmless.
    """
    match = _DEF_AT_COL0.search(stub)
    return stub[: match.start()] if match else ""


def assemble_solution(problem: dict[str, Any], candidate_code: str) -> str:
    """Build the exact source that will be executed for this candidate."""
    if problem["benchmark"] == "humaneval":
        return humaneval_preamble(problem["classifier_prompt"] + "\n") + "\n" + candidate_code
    return candidate_code


def label_one(sandbox: ScriptSandbox, problem: dict[str, Any],
              candidate: dict[str, Any]) -> dict[str, Any]:
    """Execute one candidate and return its labeled record."""
    code = candidate.get("code") or ""
    result = sandbox.run(
        solution=assemble_solution(problem, code),
        tests=problem["tests"],
        test_defs=problem["test_defs"],
    )
    return {
        "candidate_id": candidate["candidate_id"],
        "problem_id": problem["problem_id"],
        "benchmark": problem["benchmark"],
        "generator": candidate.get("generator", "unknown"),
        "candidate_index": candidate.get("candidate_index", -1),
        "code": code,
        "outcome": result.outcome,
        "label": int(result.outcome == "pass"),
        "error_type": result.error_type,
        "error_message": (result.error_message or "")[:500],
        "failed_test_index": result.failed_test_index,
        "n_tests_passed": result.n_tests_passed,
        "n_tests_total": result.n_tests_total,
        "wall_time_ms": round(result.wall_time_ms, 1),
    }


def label_candidates(problems: list[dict[str, Any]], candidates: Iterable[dict[str, Any]],
                     out_path: Path, workers: int | None = None) -> Path:
    """Label every candidate, skipping anything already recorded in *out_path*."""
    by_id = {p["problem_id"]: p for p in problems}
    already = done_ids(out_path, "candidate_id")
    todo = [c for c in candidates
            if c["candidate_id"] not in already and c["problem_id"] in by_id]
    log(f"labeling: {len(already)} already done, {len(todo)} to run "
        f"-> {out_path.name}")
    if not todo:
        return out_path

    sandbox = _make_sandbox()
    n_workers = workers or config.EXEC_WORKERS
    with JsonlAppender(out_path) as writer:
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = {
                pool.submit(label_one, sandbox, by_id[c["problem_id"]], c): c
                for c in todo
            }
            for future in progress(as_completed(futures), total=len(futures),
                                   every=100, label="exec"):
                candidate = futures[future]
                try:
                    writer.write(future.result())
                except Exception as exc:  # noqa: BLE001 - never lose the run
                    log(f"harness error on {candidate['candidate_id']}: "
                        f"{type(exc).__name__}: {exc}")
                    writer.write({
                        "candidate_id": candidate["candidate_id"],
                        "problem_id": candidate["problem_id"],
                        "benchmark": candidate.get("benchmark", "unknown"),
                        "generator": candidate.get("generator", "unknown"),
                        "candidate_index": candidate.get("candidate_index", -1),
                        "code": candidate.get("code", ""),
                        "outcome": "harness_error", "label": 0,
                        "error_type": type(exc).__name__,
                        "error_message": str(exc)[:500],
                        "failed_test_index": None,
                        "n_tests_passed": 0, "n_tests_total": 0, "wall_time_ms": 0.0,
                    })
    return out_path


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

def outcome_report(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Class balance and outcome breakdown for a labeled file."""
    if not records:
        return {"n": 0}
    counts = collections.Counter(r["outcome"] for r in records)
    n = len(records)
    n_pass = counts.get("pass", 0)
    usable = [r for r in records if r["outcome"] != "harness_error"]
    by_problem = collections.defaultdict(list)
    for record in usable:
        by_problem[record["problem_id"]].append(record["label"])
    all_pass = sum(1 for v in by_problem.values() if v and all(v))
    none_pass = sum(1 for v in by_problem.values() if v and not any(v))
    return {
        "n_candidates": n,
        "n_problems": len(by_problem),
        "pass_rate": round(n_pass / n, 4),
        "positive_class": "pass",
        "minority_fraction": round(min(n_pass, n - n_pass) / n, 4),
        "outcome_counts": dict(counts.most_common()),
        "outcome_fractions": {k: round(v / n, 4) for k, v in counts.most_common()},
        "problems_all_candidates_pass": all_pass,
        "problems_no_candidate_passes": none_pass,
        "problems_mixed": len(by_problem) - all_pass - none_pass,
        "harness_errors": counts.get("harness_error", 0),
    }


def run_reference_validation(problems: list[dict[str, Any]],
                             workers: int | None = None) -> dict[str, Any]:
    """Execute each benchmark's own reference solution against its own tests.

    A reference solution that fails means the harness is wrong, not the model.
    We expect close to 100% -- MBPP is known to contain a handful of broken
    reference solutions, so a few failures are the dataset's fault, but a low
    rate here invalidates every label downstream.
    """
    sandbox = _make_sandbox()
    results: list[tuple[str, str]] = []

    def check(problem: dict[str, Any]) -> tuple[str, str]:
        solution = problem["reference_solution"]
        if problem["benchmark"] == "humaneval":
            solution = problem["reference_solution"]  # already stub + body
        outcome = sandbox.run(solution, problem["tests"], problem["test_defs"]).outcome
        return problem["problem_id"], outcome

    with ThreadPoolExecutor(max_workers=workers or config.EXEC_WORKERS) as pool:
        futures = [pool.submit(check, p) for p in problems]
        for future in progress(as_completed(futures), total=len(futures),
                               every=100, label="refcheck"):
            results.append(future.result())

    counts = collections.Counter(outcome for _, outcome in results)
    failures = [(pid, outcome) for pid, outcome in results if outcome != "pass"]
    return {
        "n_problems": len(results),
        "reference_pass_rate": round(counts.get("pass", 0) / max(1, len(results)), 4),
        "outcome_counts": dict(counts.most_common()),
        "n_failures": len(failures),
        "failures": sorted(failures)[:50],
    }


def load_labeled(generator_tag: str, benchmark: str) -> list[dict[str, Any]]:
    return read_jsonl(config.labels_path(generator_tag, benchmark))
