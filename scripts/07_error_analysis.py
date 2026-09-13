"""Stage 7 -- test the stated hypothesis about which bugs the model misses.

    python scripts/07_error_analysis.py [--model codebert] [--split test]
"""

from __future__ import annotations

import argparse
import json

import _bootstrap  # noqa: F401
import config
from src.dataset import eval_sets, load_splits
from src.error_analysis import (confident_mistakes, full_report,
                                render_examples_markdown)
from src.evaluate import align_scores, best_f1_threshold, load_scores
from src.utils import banner, log, set_seed, write_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="codebert")
    parser.add_argument("--split", default="test", choices=["test", "humaneval", "val"])
    parser.add_argument("--generator-tag", default=config.GENERATOR_TAG)
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    set_seed(config.SEED)
    splits, heldout = load_splits(args.generator_tag)
    sets = eval_sets(splits, heldout)

    stored = load_scores(args.model)
    if not stored:
        raise FileNotFoundError(f"no scores for model {args.model!r}; run its stage first")
    if args.split not in stored or "val" not in stored:
        raise KeyError(f"{args.model} is missing scores for 'val' or {args.split!r}")

    val_scores = align_scores(sets["val"], stored["val"])
    scores = align_scores(sets[args.split], stored[args.split])
    if val_scores is None or scores is None:
        raise RuntimeError("stored scores do not match current rows; re-run the "
                           "producing stage")

    threshold = best_f1_threshold([r["label"] for r in sets["val"]], val_scores)
    rows = sets[args.split]

    banner(f"ERROR ANALYSIS — {args.model} on {args.split}")
    report = full_report(rows, scores, threshold)
    print(json.dumps(report, indent=2))

    examples = confident_mistakes(rows, scores, threshold, k=args.k)
    out_md = config.RESULTS_DIR / f"error_examples__{args.model}__{args.split}.md"
    out_md.write_text(render_examples_markdown(examples), encoding="utf-8")
    write_json(config.RESULTS_DIR / f"error_analysis__{args.model}__{args.split}.json",
               report)
    log(f"wrote {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
