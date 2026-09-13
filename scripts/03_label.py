"""Stage 3 -- execute every candidate and write ground-truth labels.

    python scripts/03_label.py --benchmark mbpp
    python scripts/03_label.py --benchmark humaneval

Resumable: already-labeled candidate_ids are skipped.
"""

from __future__ import annotations

import argparse
import json

import _bootstrap  # noqa: F401
import config
from src.data import load_problems
from src.generate import generation_meta, load_generations
from src.label import label_candidates, outcome_report
from src.utils import banner, log, read_jsonl, set_seed, write_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", choices=["mbpp", "humaneval"], required=True)
    parser.add_argument("--generator", choices=["primary", "secondary"], default="primary")
    parser.add_argument("--generator-tag", default="",
                        help="explicit tag; overrides --generator (used by the dev fixture)")
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()

    set_seed(config.SEED)
    config.ensure_dirs()
    tag = args.generator_tag or (
        config.GENERATOR_TAG if args.generator == "primary" else config.GENERATOR2_TAG)

    gen_path = config.generations_path(tag, args.benchmark)
    candidates = load_generations(gen_path)
    if not candidates:
        raise FileNotFoundError(f"no generations at {gen_path}; run 02_generate.py first")

    banner(f"EXECUTION + LABELING — {tag} on {args.benchmark} "
           f"({len(candidates)} candidates)")
    meta = generation_meta(gen_path)
    if meta:
        log(f"generation config: {json.dumps({k: v for k, v in meta.items() if k != 'record_type'})}")

    out_path = config.labels_path(tag, args.benchmark)
    label_candidates(load_problems(args.benchmark), candidates, out_path,
                     workers=args.workers or None)

    report = outcome_report(read_jsonl(out_path))
    banner("CLASS BALANCE")
    print(json.dumps(report, indent=2))
    write_json(config.RESULTS_DIR / f"label_report__{tag}__{args.benchmark}.json", report)

    minority = report.get("minority_fraction", 0.5)
    if minority < 0.20:
        print(f"\nNOTE: minority class is {minority:.1%} of rows. The neural model "
              f"uses pos_weight = n_neg/n_pos and the baselines use "
              f"class_weight='balanced', so this is handled; the number is "
              f"recorded here so it goes in the README rather than being "
              f"discovered later.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
