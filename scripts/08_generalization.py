"""Stage 8 -- does the classifier transfer to a different generator?

Prerequisites (both cheap relative to the main run):

    python scripts/02_generate.py --benchmark mbpp --generator secondary
    python scripts/03_label.py    --benchmark mbpp --generator secondary
    python scripts/08_generalization.py

Scores the SECOND generator's candidates with the model trained on the FIRST,
restricted to the held-out test problems so the comparison is like-for-like
(same problems, different generator).
"""

from __future__ import annotations

import argparse
import json

import _bootstrap  # noqa: F401
import config
from src.classifier import BEST_NAME, score_rows
from src.dataset import build_rows, load_splits
from src.data import load_problems
from src.evaluate import align_scores, best_f1_threshold, compute_metrics, load_scores
from src.generalization import compare_transfer, generator_fingerprint_probe
from src.label import load_labeled
from src.utils import banner, log, set_seed, write_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary-tag", default=config.GENERATOR_TAG)
    parser.add_argument("--secondary-tag", default=config.GENERATOR2_TAG)
    args = parser.parse_args()

    set_seed(config.SEED)
    config.ensure_dirs()

    splits, _ = load_splits(args.primary_tag)
    test_problem_ids = {r["problem_id"] for r in splits.test}

    secondary = build_rows(load_labeled(args.secondary_tag, "mbpp"),
                           load_problems("mbpp"))
    if not secondary:
        raise FileNotFoundError(
            f"no labeled candidates for generator {args.secondary_tag!r}; "
            "run 02_generate.py --generator secondary and 03_label.py first")

    # Same problems, different generator: isolates the generator effect from
    # any difference in problem difficulty.
    transfer_rows = [r for r in secondary if r["problem_id"] in test_problem_ids]
    log(f"{len(transfer_rows)} transfer rows on {len(test_problem_ids)} test problems")
    if not transfer_rows:
        raise RuntimeError("second generator produced nothing on the test problems")

    banner("GENERATOR FINGERPRINT PROBE")
    probe = generator_fingerprint_probe(splits.test, transfer_rows)
    print(json.dumps(probe, indent=2))

    banner("TRANSFER")
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    best_path = config.MODELS_DIR / BEST_NAME
    if not best_path.exists():
        raise FileNotFoundError(f"trained model not found at {best_path}; "
                                "run 05_train_classifier.py first")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(config.CLASSIFIER_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.CLASSIFIER_MODEL, num_labels=1)
    model.load_state_dict(torch.load(best_path, map_location=device,
                                     weights_only=False)["model"])
    model.to(device)

    transfer_scores, truncation = score_rows(model, tokenizer, transfer_rows, device)

    stored = load_scores("codebert")
    val_scores = align_scores(splits.val, stored["val"])
    threshold = best_f1_threshold([r["label"] for r in splits.val], val_scores)

    in_dist = compute_metrics([r["label"] for r in splits.test],
                              align_scores(splits.test, stored["test"]), threshold)
    transfer = compute_metrics([r["label"] for r in transfer_rows],
                               transfer_scores, threshold)

    comparison = compare_transfer(in_dist, transfer, args.primary_tag,
                                  args.secondary_tag)
    report = {
        "fingerprint_probe": probe,
        "in_distribution": in_dist,
        "transfer": transfer,
        "comparison": comparison,
        "transfer_truncation": truncation,
        "threshold_source": "val split of the primary generator",
    }
    print(json.dumps(report, indent=2))
    write_json(config.RESULTS_DIR / "generalization.json", report)
    log(f"wrote {config.RESULTS_DIR / 'generalization.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
