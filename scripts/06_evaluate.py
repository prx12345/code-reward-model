"""Stage 6 -- one comparison table over every model that has scores on disk.

    python scripts/06_evaluate.py

Thresholds are chosen on VAL and applied unchanged to test and HumanEval.
Any model whose held-out AUC exceeds config.SUSPICIOUS_AUC aborts the run
(override with --allow-high-auc only after you have found the leak).
"""

from __future__ import annotations

import argparse
import json

import _bootstrap  # noqa: F401
import config
from src.dataset import eval_sets, load_splits
from src.evaluate import (SCORES_DIR, align_scores, best_f1_threshold,
                          check_suspicious, comparison_table, compute_metrics,
                          load_scores, plot_calibration, plot_confusion,
                          plot_roc, plot_score_histogram)
from src.utils import banner, log, set_seed, write_json

MODEL_ORDER = ["majority", "static_logreg", "tfidf_logreg", "codebert"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generator-tag", default=config.GENERATOR_TAG)
    parser.add_argument("--allow-high-auc", action="store_true",
                        help="report a >0.95 AUC instead of aborting")
    args = parser.parse_args()

    set_seed(config.SEED)
    config.ensure_dirs()

    splits, heldout = load_splits(args.generator_tag)
    sets = eval_sets(splits, heldout)
    labels = {name: [r["label"] for r in rows] for name, rows in sets.items()}

    available = sorted(p.stem for p in SCORES_DIR.glob("*.json")) if SCORES_DIR.exists() else []
    models = [m for m in MODEL_ORDER if m in available] + \
             [m for m in available if m not in MODEL_ORDER]
    if not models:
        raise FileNotFoundError(
            f"no score files in {SCORES_DIR}. Run 04_baselines.py "
            "(and 05_train_classifier.py) first."
        )
    log(f"models with scores on disk: {models}")

    results: dict[str, dict] = {}
    thresholds: dict[str, float] = {}
    all_scores: dict[str, dict] = {}

    for model in models:
        stored = load_scores(model)
        per_split_scores = {}
        for split, rows in sets.items():
            if split not in stored:
                continue
            aligned = align_scores(rows, stored[split])
            if aligned is None:
                log(f"WARNING: {model} scores for split {split} do not cover the "
                    f"current rows; skipping (re-run the stage that produced them)")
                continue
            per_split_scores[split] = aligned
        if "val" not in per_split_scores:
            log(f"WARNING: {model} has no val scores; cannot pick a threshold, skipping")
            continue

        threshold = best_f1_threshold(labels["val"], per_split_scores["val"])
        thresholds[model] = threshold
        all_scores[model] = per_split_scores
        results[model] = {
            split: compute_metrics(labels[split], scores, threshold)
            for split, scores in per_split_scores.items()
        }

    banner("RESULTS")
    table_sections = []
    for split in ("val", "test", "humaneval"):
        if not any(split in per_split for per_split in results.values()):
            continue
        table = comparison_table(results, split)
        print("\n" + table)
        table_sections.append(table)

    # Leak guard, after printing so the number is visible before the abort.
    for model, per_split in results.items():
        for split in ("test", "humaneval"):
            if split in per_split:
                check_suspicious(per_split[split], model, split,
                                 allow=args.allow_high_auc)

    banner("PLOTS")
    for split in ("test", "humaneval"):
        curves = {m: (labels[split], s[split]) for m, s in all_scores.items()
                  if split in s}
        if not curves:
            continue
        log(str(plot_roc(curves, f"ROC — {split}", f"roc_{split}.png")))
        log(str(plot_calibration(curves, f"Calibration — {split}",
                                 f"calibration_{split}.png")))
    for model, per_split in results.items():
        for split in ("test", "humaneval"):
            if split in per_split:
                log(str(plot_confusion(per_split[split],
                                       f"{model} — {split}",
                                       f"confusion_{model}_{split}.png")))
    # The score histogram is only interesting for the strongest model; for
    # `majority` it is a single bar and tells you nothing.
    scored_on_test = {m: results[m]["test"]["auc"] for m in all_scores
                      if "test" in results.get(m, {})}
    best = ("codebert" if "codebert" in all_scores
            else (max(scored_on_test, key=scored_on_test.get) if scored_on_test else None))
    if best and "test" in all_scores.get(best, {}):
        log(str(plot_score_histogram(labels["test"], all_scores[best]["test"],
                                     f"{best} score distribution — test",
                                     f"score_hist_{best}_test.png")))

    payload = {"thresholds": thresholds, "metrics": results,
               "generator_tag": args.generator_tag}
    write_json(config.RESULTS_DIR / "metrics.json", payload)
    (config.RESULTS_DIR / "comparison_table.md").write_text(
        "\n\n".join(table_sections) + "\n", encoding="utf-8")
    log(f"wrote {config.RESULTS_DIR / 'metrics.json'} and comparison_table.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
