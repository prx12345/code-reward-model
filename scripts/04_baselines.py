"""Stage 4 -- fit the baselines and persist their scores.

    python scripts/04_baselines.py

Deliberately runs before the transformer. If a bag of code tokens already gets
most of the achievable AUC, that is the headline finding, not a footnote.
"""

from __future__ import annotations

import argparse
import json

import _bootstrap  # noqa: F401
import config
from src.baselines import fit_all_baselines
from src.dataset import class_weights, eval_sets, load_splits, outcome_distribution
from src.evaluate import save_scores
from src.utils import banner, log, set_seed, write_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generator-tag", default=config.GENERATOR_TAG)
    args = parser.parse_args()

    set_seed(config.SEED)
    config.ensure_dirs()

    splits, heldout = load_splits(args.generator_tag)
    sets = eval_sets(splits, heldout)

    banner("SPLITS")
    summary = splits.summary()
    summary["humaneval_heldout"] = {
        "n_rows": len(heldout),
        "n_problems": len({r["problem_id"] for r in heldout}),
        "pass_rate": (round(sum(r["label"] for r in heldout) / len(heldout), 4)
                      if heldout else None),
    }
    print(json.dumps(summary, indent=2))

    banner("CLASS BALANCE (train)")
    weights = class_weights(splits.train)
    print(json.dumps(weights, indent=2))
    print("outcome distribution (train):",
          json.dumps(outcome_distribution(splits.train), indent=2))

    banner("FITTING BASELINES")
    scores, info = fit_all_baselines(splits.train, sets)
    for name, by_split in scores.items():
        path = save_scores(name, by_split, sets)
        log(f"{name}: scores -> {path.name}")

    write_json(config.RESULTS_DIR / "splits_summary.json", summary)
    write_json(config.RESULTS_DIR / "class_weights.json", weights)
    write_json(config.RESULTS_DIR / "baseline_info.json", info)
    log("baselines done; run scripts/06_evaluate.py for the comparison table")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
