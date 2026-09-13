"""Load MBPP and HumanEval, normalise both into one problem schema.

HumanEval is the held-out set — assert_humaneval_never_trained() guards it.
"""

from __future__ import annotations

import ast
import builtins
import gzip
import json
import urllib.request
from pathlib import Path
from typing import Any

import config
from src.utils import log, read_jsonl, write_jsonl

# NOTE: `dir(__builtins__)` is wrong here -- inside an imported module
# `__builtins__` is the builtins *dict*, so `dir()` returns dict methods and
# every real builtin (len, set, sorted) would be treated as a candidate
# function name. Import the module explicitly instead.
_PY_BUILTINS = frozenset(dir(builtins))


# --------------------------------------------------------------------------
# Raw download
# --------------------------------------------------------------------------

def _download(url: str, dest: Path) -> Path:
    """Fetch *url* to *dest* once; reuse the cached copy afterwards."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        log(f"cached {dest.name} ({dest.stat().st_size} bytes)")
        return dest
    log(f"downloading {url}")
    with urllib.request.urlopen(url, timeout=60) as response:  # noqa: S310 - fixed URL
        payload = response.read()
    dest.write_bytes(payload)
    log(f"saved {dest.name} ({len(payload)} bytes)")
    return dest


def _load_mbpp_raw() -> list[dict[str, Any]]:
    """MBPP rows, preferring the HF hub and falling back to the upstream file.

    The HF route is tried first because that is what a reader will expect; the
    GitHub fallback exists because the hub is not always reachable (corporate
    proxies, offline runners) and because `google-research/mbpp.jsonl` is the
    file every MBPP paper actually used.
    """
    try:
        from datasets import load_dataset  # type: ignore

        rows: list[dict[str, Any]] = []
        dataset = load_dataset("mbpp", "full")
        for split in dataset:
            rows.extend(dict(row) for row in dataset[split])
        log(f"MBPP via huggingface datasets: {len(rows)} rows")
        return rows
    except Exception as exc:  # noqa: BLE001 - any hub failure falls back
        log(f"MBPP: hub unavailable ({type(exc).__name__}), using upstream file")

    path = _download(config.MBPP_URL, config.RAW_DIR / "mbpp.jsonl")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _load_humaneval_raw() -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset  # type: ignore

        dataset = load_dataset("openai_humaneval")
        rows = [dict(row) for row in dataset["test"]]
        log(f"HumanEval via huggingface datasets: {len(rows)} rows")
        return rows
    except Exception as exc:  # noqa: BLE001
        log(f"HumanEval: hub unavailable ({type(exc).__name__}), using upstream file")

    path = _download(config.HUMANEVAL_URL, config.RAW_DIR / "HumanEval.jsonl.gz")
    text = gzip.decompress(path.read_bytes()).decode("utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


# --------------------------------------------------------------------------
# Normalization
# --------------------------------------------------------------------------

def infer_entry_point(test_statements: list[str]) -> str | None:
    """Recover the function name the MBPP assertions call.

    MBPP does not record it. Parsing the assertion with ``ast`` and taking the
    first called ``Name`` that is not a builtin is robust to the wrappers that
    show up in the suite (``assert math.isclose(f(x), y)``,
    ``assert set(f(x)) == {...}``) in a way that a regex on ``\\w+\\(`` is not.
    """
    for statement in test_statements:
        try:
            tree = ast.parse(statement)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                name = node.func.id
                if name not in _PY_BUILTINS:
                    return name
    return None


def _mbpp_generation_prompt(text: str, tests: list[str]) -> str:
    joined = "\n".join(tests)
    return (
        f"{text.strip()}\n\n"
        f"Your code should satisfy these tests:\n{joined}\n\n"
        "Write the complete Python function. Respond with a single Python code "
        "block and no explanation."
    )


def normalize_mbpp(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    problems: list[dict[str, Any]] = []
    skipped_prompt_ids = 0
    skipped_no_entry = 0
    for row in rows:
        task_id = int(row["task_id"])
        if task_id in config.MBPP_PROMPT_TASK_IDS:
            skipped_prompt_ids += 1
            continue
        tests = list(row.get("test_list") or [])
        if not tests:
            continue
        entry_point = infer_entry_point(tests)
        if entry_point is None:
            skipped_no_entry += 1
            continue
        text = str(row.get("text", "")).strip()
        problems.append({
            "problem_id": f"mbpp/{task_id}",
            "benchmark": "mbpp",
            "prompt_kind": "nl",
            "classifier_prompt": text,
            "generation_prompt": _mbpp_generation_prompt(text, tests),
            "entry_point": entry_point,
            # MBPP stores \r\n; normalize so nothing downstream has to care.
            "test_defs": str(row.get("test_setup_code", "") or "").replace("\r\n", "\n"),
            "tests": [t.replace("\r\n", "\n") for t in tests],
            "reference_solution": str(row.get("code", "")).replace("\r\n", "\n"),
            "n_tests": len(tests),
        })
    log(f"MBPP normalized: {len(problems)} problems "
        f"(dropped {skipped_prompt_ids} few-shot ids, {skipped_no_entry} unparseable)")
    return problems


def _humaneval_generation_prompt(stub: str) -> str:
    return (
        "Complete this Python function.\n\n"
        f"```python\n{stub.rstrip()}\n```\n\n"
        "Respond with the complete function (signature included) in a single "
        "Python code block and no explanation."
    )


def normalize_humaneval(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    problems: list[dict[str, Any]] = []
    for row in rows:
        task_id = str(row["task_id"])                 # "HumanEval/2"
        slug = task_id.replace("/", "_")              # "HumanEval_2"
        entry_point = str(row["entry_point"])
        stub = str(row["prompt"])
        problems.append({
            "problem_id": f"humaneval/{slug}",
            "benchmark": "humaneval",
            "prompt_kind": "signature",
            "classifier_prompt": stub.strip(),
            "generation_prompt": _humaneval_generation_prompt(stub),
            "entry_point": entry_point,
            # The whole `check()` definition is scaffolding; the single
            # statement that can fail is the call to it.
            "test_defs": str(row["test"]),
            "tests": [f"check({entry_point})"],
            "reference_solution": stub + str(row["canonical_solution"]),
            "n_tests": 1,
        })
    log(f"HumanEval normalized: {len(problems)} problems")
    return problems


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------

def build_problem_sets(force: bool = False) -> dict[str, list[dict[str, Any]]]:
    """Download, normalize and cache both benchmarks. Idempotent."""
    config.ensure_dirs()
    if not force and config.PROBLEMS_MBPP.exists() and config.PROBLEMS_HUMANEVAL.exists():
        log("problem sets already built; loading from disk")
        return {
            "mbpp": read_jsonl(config.PROBLEMS_MBPP),
            "humaneval": read_jsonl(config.PROBLEMS_HUMANEVAL),
        }
    mbpp = normalize_mbpp(_load_mbpp_raw())
    humaneval = normalize_humaneval(_load_humaneval_raw())
    write_jsonl(config.PROBLEMS_MBPP, mbpp)
    write_jsonl(config.PROBLEMS_HUMANEVAL, humaneval)
    return {"mbpp": mbpp, "humaneval": humaneval}


def load_problems(benchmark: str) -> list[dict[str, Any]]:
    path = config.PROBLEMS_MBPP if benchmark == "mbpp" else config.PROBLEMS_HUMANEVAL
    if not path.exists():
        build_problem_sets()
    return read_jsonl(path)


def assert_humaneval_never_trained(problem_ids: list[str] | set[str]) -> None:
    """Hard guard: crash if any HumanEval problem reached a training split.

    HumanEval is the canonical benchmark. Training on it makes every number
    reported against it meaningless, and the mistake is silent -- nothing about
    a leaked test set looks wrong until someone else fails to reproduce you.
    """
    leaked = sorted(pid for pid in problem_ids if str(pid).startswith("humaneval/"))
    if leaked:
        raise LeakageError(
            f"{len(leaked)} HumanEval problems are in a training split "
            f"(e.g. {leaked[:5]}). HumanEval is held out; refusing to continue."
        )


class LeakageError(RuntimeError):
    """Raised when a split guard detects train/test contamination."""


# --------------------------------------------------------------------------
# Stats
# --------------------------------------------------------------------------

def dataset_stats(problems: list[dict[str, Any]]) -> dict[str, Any]:
    if not problems:
        return {"n_problems": 0}
    prompt_chars = [len(p["classifier_prompt"]) for p in problems]
    ref_chars = [len(p["reference_solution"]) for p in problems]
    tests = [p["n_tests"] for p in problems]
    return {
        "n_problems": len(problems),
        "benchmark": problems[0]["benchmark"],
        "prompt_kind": problems[0]["prompt_kind"],
        "prompt_chars_mean": round(sum(prompt_chars) / len(prompt_chars), 1),
        "prompt_chars_median": sorted(prompt_chars)[len(prompt_chars) // 2],
        "prompt_chars_max": max(prompt_chars),
        "reference_chars_mean": round(sum(ref_chars) / len(ref_chars), 1),
        "reference_chars_max": max(ref_chars),
        "tests_per_problem_mean": round(sum(tests) / len(tests), 2),
        "problems_with_test_defs": sum(1 for p in problems if p["test_defs"].strip()),
    }
