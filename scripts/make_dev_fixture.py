"""Make a CPU-only fixture so the pipeline can be exercised without a GPU.

**This does not produce results and must never be reported as one.** It writes
a generations file under the tag ``devfixture-mutants`` whose "candidates" are
deterministic mutations of each benchmark's own reference solution, not model
samples. Its only job is to let you run stages 3 through 7 end to end -- so
that a bug in the splitting, baselines or evaluation code surfaces in thirty
seconds on a laptop instead of after three hours of Colab generation.

Every mutation is a category of bug the real generator also makes, which makes
the fixture a reasonable shakedown of the label taxonomy too:

    identity        -> should pass
    comparison flip -> off-by-one (assertion_failure)
    range shrink    -> off-by-one (assertion_failure)
    literal bump    -> wrong constant (assertion_failure)
    drop return     -> returns None (assertion_failure)
    undefined name  -> NameError (exception)
    broken syntax   -> syntax_error
    delete a line   -> anything

Usage:
    python scripts/make_dev_fixture.py --limit 150
    python scripts/03_label.py --benchmark mbpp --generator-tag devfixture-mutants
"""

from __future__ import annotations

import argparse
import random
import re
from typing import Callable

import _bootstrap  # noqa: F401
import config
from src.data import load_problems
from src.generate import _meta_record
from src.utils import JsonlAppender, banner, log, set_seed

FIXTURE_TAG = "devfixture-mutants"


def _identity(code: str, rng: random.Random) -> str:
    return code


def _flip_comparison(code: str, rng: random.Random) -> str:
    for old, new in ((" < ", " <= "), (" > ", " >= "), (" <= ", " < "), (" >= ", " > ")):
        if old in code:
            return code.replace(old, new, 1)
    return code + "\n"


def _shrink_range(code: str, rng: random.Random) -> str:
    return re.sub(r"range\(([^),]+)\)", r"range(\1 - 1)", code, count=1)


def _bump_literal(code: str, rng: random.Random) -> str:
    def replace(match: re.Match[str]) -> str:
        return str(int(match.group(0)) + 1)
    return re.sub(r"(?<![\w.])\d+(?![\w.])", replace, code, count=1)


def _drop_return(code: str, rng: random.Random) -> str:
    lines = code.split("\n")
    for index in range(len(lines) - 1, -1, -1):
        if lines[index].strip().startswith("return "):
            indent = lines[index][: len(lines[index]) - len(lines[index].lstrip())]
            lines[index] = f"{indent}pass"
            return "\n".join(lines)
    return code


def _undefined_name(code: str, rng: random.Random) -> str:
    lines = code.split("\n")
    for index, line in enumerate(lines):
        if line.strip().startswith("return "):
            indent = line[: len(line) - len(line.lstrip())]
            lines.insert(index, f"{indent}_ = undefined_helper_fn(1)")
            return "\n".join(lines)
    return code + "\nundefined_helper_fn(1)\n"


def _break_syntax(code: str, rng: random.Random) -> str:
    return code.replace(":", "", 1) if ":" in code else code + "\n  if True\n"


def _delete_line(code: str, rng: random.Random) -> str:
    lines = [line for line in code.split("\n") if line.strip()]
    if len(lines) < 3:
        return code
    index = rng.randrange(1, len(lines))
    del lines[index]
    return "\n".join(lines)


MUTATIONS: list[tuple[str, Callable[[str, random.Random], str]]] = [
    ("identity", _identity),
    ("flip_comparison", _flip_comparison),
    ("shrink_range", _shrink_range),
    ("bump_literal", _bump_literal),
    ("drop_return", _drop_return),
    ("undefined_name", _undefined_name),
    ("break_syntax", _break_syntax),
    ("delete_line", _delete_line),
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", choices=["mbpp", "humaneval"], default="mbpp")
    parser.add_argument("--limit", type=int, default=200)
    args = parser.parse_args()

    set_seed(config.SEED)
    config.ensure_dirs()

    problems = load_problems(args.benchmark)[: args.limit or None]
    out_path = config.generations_path(FIXTURE_TAG, args.benchmark)
    if out_path.exists():
        out_path.unlink()  # fixtures are disposable; real generations never are

    banner(f"DEV FIXTURE (NOT A RESULT) — {len(problems)} problems x "
           f"{len(MUTATIONS)} mutants")
    rng = random.Random(config.SEED)
    with JsonlAppender(out_path) as writer:
        meta = _meta_record("SYNTHETIC-MUTANTS-NOT-A-MODEL", FIXTURE_TAG,
                            args.benchmark, len(problems))
        meta["warning"] = ("synthetic mutants of reference solutions; for "
                           "pipeline smoke-testing only, never a result")
        writer.write(meta)
        for problem in problems:
            for index, (name, mutate) in enumerate(MUTATIONS):
                writer.write({
                    "candidate_id": f"{problem['problem_id']}#{FIXTURE_TAG}#{index}",
                    "problem_id": problem["problem_id"],
                    "benchmark": problem["benchmark"],
                    "generator": FIXTURE_TAG,
                    "candidate_index": index,
                    "code": mutate(problem["reference_solution"], rng),
                    "mutation": name,
                })
    log(f"wrote {out_path}")
    log("next: python scripts/03_label.py --benchmark "
        f"{args.benchmark} --generator-tag {FIXTURE_TAG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
