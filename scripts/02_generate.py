"""Stage 2 -- sample candidate solutions. Resumable; safe to re-run after a crash.

    python scripts/02_generate.py --benchmark mbpp
    python scripts/02_generate.py --benchmark humaneval
    python scripts/02_generate.py --benchmark mbpp --generator secondary
"""

from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
import config
from src.data import load_problems
from src.generate import generate_candidates
from src.utils import banner, log, set_seed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", choices=["mbpp", "humaneval"], required=True)
    parser.add_argument("--generator", choices=["primary", "secondary"], default="primary")
    parser.add_argument("--limit", type=int, default=0, help="first N problems only")
    parser.add_argument("--n-candidates", type=int, default=0)
    args = parser.parse_args()

    set_seed(config.SEED)
    config.ensure_dirs()

    if args.generator == "primary":
        model_name, tag = config.GENERATOR_MODEL, config.GENERATOR_TAG
    else:
        model_name, tag = config.GENERATOR2_MODEL, config.GENERATOR2_TAG

    banner(f"GENERATION — {model_name} on {args.benchmark}")
    problems = load_problems(args.benchmark)
    path = generate_candidates(
        problems, model_name=model_name, generator_tag=tag,
        benchmark=args.benchmark,
        n_candidates=args.n_candidates or config.N_CANDIDATES,
        limit=args.limit,
    )
    log(f"generations at {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
