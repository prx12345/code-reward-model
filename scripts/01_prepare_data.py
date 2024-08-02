"""Stage 1 -- build the normalized problem sets and sanity-check the harness.

Run:  python scripts/01_prepare_data.py [--validate-harness] [--limit N]

``--validate-harness`` executes each benchmark's *own reference solution*
against its *own tests*. This is the cheapest possible check that the executor
and the test-assembly code are correct: if a canonical solution does not pass,
the bug is ours, not the model's, and every label downstream would be wrong.
It is also the number to quote when someone asks how you know your labels are
trustworthy.
"""

from __future__ import annotations

import argparse
import collections

import _bootstrap  # noqa: F401
import config
from src.data import build_problem_sets, dataset_stats
from src.label import run_reference_validation
from src.utils import banner, log, set_seed, write_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="re-download and re-normalize")
    parser.add_argument("--validate-harness", action="store_true",
                        help="execute reference solutions against their own tests")
    parser.add_argument("--limit", type=int, default=0,
                        help="validate only the first N problems per benchmark (0 = all)")
    args = parser.parse_args()

    set_seed(config.SEED)
    config.ensure_dirs()

    problems = build_problem_sets(force=args.force)

    stats = {}
    for name, rows in problems.items():
        banner(f"{name.upper()} — dataset stats")
        stats[name] = dataset_stats(rows)
        for key, value in stats[name].items():
            print(f"  {key:28s} {value}")
        lengths = collections.Counter(len(r["tests"]) for r in rows)
        print(f"  {'tests_per_problem_hist':28s} {dict(sorted(lengths.items()))}")

    if args.validate_harness:
        for name, rows in problems.items():
            subset = rows[: args.limit] if args.limit else rows
            banner(f"{name.upper()} — harness validation on reference solutions "
                   f"(n={len(subset)})")
            report = run_reference_validation(subset)
            for key, value in report.items():
                if key != "failures":
                    print(f"  {key:28s} {value}")
            if report["failures"]:
                print(f"  first failures: {report['failures'][:5]}")
            stats[f"{name}_reference_validation"] = report

    write_json(config.RESULTS_DIR / "dataset_stats.json", stats)
    log(f"wrote {config.RESULTS_DIR / 'dataset_stats.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
