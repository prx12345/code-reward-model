"""Stage 5 -- fine-tune CodeBERT and persist its scores.

    python scripts/05_train_classifier.py

Resumes automatically from models/codebert_checkpoint.pt if one exists.
Pass --fresh to ignore it.
"""

from __future__ import annotations

import argparse
import json

import _bootstrap  # noqa: F401
import config
from src.data import assert_humaneval_never_trained
from src.dataset import assert_no_leakage, class_weights, eval_sets, load_splits
from src.classifier import score_rows, train_classifier
from src.evaluate import save_scores
from src.utils import banner, log, set_seed, write_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generator-tag", default=config.GENERATOR_TAG)
    parser.add_argument("--fresh", action="store_true", help="ignore any checkpoint")
    args = parser.parse_args()

    set_seed(config.SEED)
    config.ensure_dirs()

    splits, heldout = load_splits(args.generator_tag)

    # Belt and braces. `split_by_problem` already ran these; running them again
    # here means the guard sits directly in front of the optimiser, where a
    # future refactor is most likely to break it.
    banner("LEAKAGE GUARDS")
    assert_no_leakage(splits)
    assert_humaneval_never_trained({r["problem_id"] for r in splits.train})
    assert_humaneval_never_trained({r["problem_id"] for r in splits.val})
    train_ids = {r["problem_id"] for r in splits.train}
    test_ids = {r["problem_id"] for r in splits.test}
    assert not (train_ids & test_ids), "train/test problem overlap"
    print(f"  OK: {len(train_ids)} train problems, {len(test_ids)} test problems, "
          f"0 shared")
    print(f"  OK: 0 HumanEval problems in train or val "
          f"({len({r['problem_id'] for r in heldout})} held out)")

    weights = class_weights(splits.train)
    banner(f"TRAINING {config.CLASSIFIER_MODEL}")
    print(json.dumps(weights, indent=2))

    result = train_classifier(splits.train, splits.val,
                              pos_weight=weights["pos_weight"],
                              resume=not args.fresh)
    model = result.pop("_model")
    tokenizer = result.pop("_tokenizer")

    sets = eval_sets(splits, heldout)
    scores = {}
    truncation = {}
    for name, rows in sets.items():
        scores[name], truncation[name] = score_rows(model, tokenizer, rows,
                                                    result["device"])
    save_scores("codebert", scores, sets)

    result["eval_truncation"] = truncation
    write_json(config.RESULTS_DIR / "training_report.json", result)
    banner("TRAINING REPORT")
    print(json.dumps(result, indent=2, default=str))
    log("run scripts/06_evaluate.py for the comparison table")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
