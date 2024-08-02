"""Single source of truth for every tunable in the pipeline.

Every stage imports from here. Nothing else in the repo hard-codes a path, a
seed, a model name or a hyperparameter. If you want to change the experiment,
change this file and re-run -- that is the only knob.
"""

from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"            # downloaded benchmark files
INTERIM_DIR = DATA_DIR / "interim"    # normalized problems, generations
PROCESSED_DIR = DATA_DIR / "processed"  # labeled candidates, splits
RESULTS_DIR = ROOT / "results"
MODELS_DIR = ROOT / "models"

PROBLEMS_MBPP = INTERIM_DIR / "problems_mbpp.jsonl"
PROBLEMS_HUMANEVAL = INTERIM_DIR / "problems_humaneval.jsonl"

# Generations and labels are keyed by (generator_tag, benchmark) so the
# generalization check can live alongside the main run without collisions.
def generations_path(generator_tag: str, benchmark: str) -> Path:
    return INTERIM_DIR / f"generations__{generator_tag}__{benchmark}.jsonl"


def labels_path(generator_tag: str, benchmark: str) -> Path:
    return PROCESSED_DIR / f"labeled__{generator_tag}__{benchmark}.jsonl"


# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------

SEED = 1337

# --------------------------------------------------------------------------
# Benchmarks
# --------------------------------------------------------------------------

# We pull the raw files from their canonical GitHub homes rather than through
# `datasets.load_dataset`. Two reasons: (a) it works in environments where the
# HF hub is unreachable, (b) the HF mirrors of MBPP have been re-cut more than
# once, so pinning the upstream file keeps the dataset stable across reruns.
# `src/data.py` still tries the HF `datasets` route first when it is available.
MBPP_URL = "https://raw.githubusercontent.com/google-research/google-research/master/mbpp/mbpp.jsonl"
HUMANEVAL_URL = "https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz"

# MBPP task_ids 1-10 are the canonical few-shot prompting examples used by
# every paper that reports MBPP numbers. We drop them from the training pool so
# nothing that is conventionally shown to a model in-context ends up as data.
MBPP_PROMPT_TASK_IDS = set(range(1, 11))

# HumanEval is the held-out test set. Nothing here ever trains on it.
HELD_OUT_BENCHMARK = "humaneval"
TRAIN_BENCHMARK = "mbpp"

# --------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------

GENERATOR_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
GENERATOR_TAG = "qwen2.5-coder-1.5b"

# A second, different-family generator used only for the generalization check.
# Different family (not just a different size of Qwen) is the whole point: if
# the classifier has merely memorised one model's formatting habits, a sibling
# checkpoint would not expose it.
GENERATOR2_MODEL = "deepseek-ai/deepseek-coder-1.3b-instruct"
GENERATOR2_TAG = "deepseek-coder-1.3b"

N_CANDIDATES = 8
GEN_TEMPERATURE = 0.8
GEN_TOP_P = 0.95
GEN_MAX_NEW_TOKENS = 512
GEN_BATCH_SIZE = 8        # candidates generated in one forward batch
GEN_DTYPE = "float16"     # T4 has no bf16
GEN_FLUSH_EVERY = 1       # problems between disk flushes; 1 = crash-proof

# --------------------------------------------------------------------------
# Execution / labeling
# --------------------------------------------------------------------------

EXEC_TIMEOUT_S = 6.0      # wall-clock, per candidate (all its asserts together)
EXEC_MEMORY_MB = 512
EXEC_CPU_TIME_S = 8       # belt-and-braces if the parent itself is starved
EXEC_WORKERS = 4          # parallel subprocesses; each is independently sandboxed

# Ground truth is binary, but we record the *reason* for a failure so error
# analysis can ask "which kinds of bugs does the classifier miss?".
OUTCOMES = (
    "pass",
    "assertion_failure",
    "exception",
    "timeout",
    "syntax_error",
    "memory_exceeded",
    "harness_error",
)

# --------------------------------------------------------------------------
# Dataset assembly
# --------------------------------------------------------------------------

# Temperature-0.8 sampling produces exact duplicates within a problem. Keeping
# them makes the metrics reflect how often the sampler repeats itself rather
# than how well the classifier discriminates, so we collapse them and keep the
# multiplicity as a column. See README "Deduplication".
DEDUP_WITHIN_PROBLEM = True

VAL_FRACTION = 0.15       # fraction of MBPP *problems* (not rows) held for val
TEST_FRACTION = 0.15      # in-distribution MBPP test; HumanEval is separate

# --------------------------------------------------------------------------
# Classifier
# --------------------------------------------------------------------------

CLASSIFIER_MODEL = "microsoft/codebert-base"
MAX_LENGTH = 512
PROBLEM_TOKEN_BUDGET = 128   # tokens reserved for the problem statement
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
EPOCHS = 3
TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 32
WARMUP_RATIO = 0.1
MAX_GRAD_NORM = 1.0
FP16 = True
EARLY_STOP_METRIC = "val_auc"
CHECKPOINT_EVERY_STEPS = 200   # so a dead Colab session loses minutes, not hours

# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------

# A code-correctness reward model that scores ~0.95+ AUC on held-out problems
# would be extraordinary. Far more likely: a leak. Evaluation hard-stops above
# this and tells you where to look.
SUSPICIOUS_AUC = 0.95

CALIBRATION_BINS = 10

# --------------------------------------------------------------------------
# Environment
# --------------------------------------------------------------------------

def ensure_dirs() -> None:
    """Create every output directory. Safe to call repeatedly."""
    for path in (RAW_DIR, INTERIM_DIR, PROCESSED_DIR, RESULTS_DIR, MODELS_DIR):
        path.mkdir(parents=True, exist_ok=True)


# Set by scripts/ before importing torch so cuBLAS is deterministic.
CUBLAS_WORKSPACE_CONFIG = ":4096:8"
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", CUBLAS_WORKSPACE_CONFIG)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
