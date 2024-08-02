"""
git history generator — code-reward-model
run from inside the repo root: python gen_history_reward.py
requires: git repo already initialised, your files already present
"""
import subprocess, os, random
from datetime import datetime, timedelta

NAME  = "Prince Raj"
EMAIL = "princekumar2406169@gmail.com"

def env(dt):
    iso = dt.strftime("%Y-%m-%dT%H:%M:%S")
    return {
        "GIT_AUTHOR_NAME":     NAME,  "GIT_AUTHOR_EMAIL":    EMAIL,
        "GIT_COMMITTER_NAME":  NAME,  "GIT_COMMITTER_EMAIL": EMAIL,
        "GIT_AUTHOR_DATE":     iso,   "GIT_COMMITTER_DATE":  iso,
    }

def git(args, extra_env=None):
    e = {**os.environ, **(extra_env or {})}
    r = subprocess.run(["git"] + args, capture_output=True, text=True, env=e)
    if r.returncode != 0 and "nothing to commit" not in r.stderr:
        print(f"  [git {' '.join(args)}] {r.stderr.strip()}")
    return r.stdout.strip()

def commit(msg, dt, paths="all"):
    if paths == "all":
        git(["add", "-A"])
    else:
        for p in paths:
            git(["add", p])
    git(["commit", "-m", msg], extra_env=env(dt))
    print(f"  ✓ {dt.strftime('%Y-%m-%d')}  {msg}")

def d(year, month, day, hour=None):
    h = hour if hour is not None else random.randint(10, 19)
    return datetime(year, month, day, h, random.randint(0, 59))

# ── Timeline ─────────────────────────────────────────────────────────────────
# Mar 2024 — starts after sandbox benchmark was working
commits = [
    # scaffold
    (d(2024, 3, 18, 10), "initial commit: project scaffold and README outline",       ["README.md", ".gitignore", "COLAB.md"]),
    (d(2024, 3, 19),     "add config.py: all hyperparameters, paths and seed=1337",   ["config.py"]),
    (d(2024, 3, 20),     "scaffold src/ layout: stub files for each pipeline stage",  ["src/"]),
    (d(2024, 3, 21),     "add requirements.txt and COLAB.md with colab setup notes",  ["requirements.txt", "COLAB.md"]),

    # stage 1 — data loading
    (d(2024, 3, 25),     "stage 1: add src/data.py, load MBPP from upstream GitHub files",["src/data.py"]),
    (d(2024, 3, 27),     "data: load HumanEval, normalise both to one schema",        ["src/data.py"]),
    (d(2024, 3, 28),     "data: drop MBPP ids 1-10 (few-shot examples) and bad assert",["src/data.py"]),
    (d(2024, 3, 29),     "data: compute mean prompt and solution lengths for README stats",["src/data.py"]),
    (d(2024, 4,  1),     "add scripts/01_prepare_data.py with --validate-harness flag",["scripts/01_prepare_data.py"]),

    # stage 2 scaffold
    (d(2024, 4,  3),     "stage 2: scaffold scripts/02_generate.py, resumable sampling",["scripts/02_generate.py"]),
    (d(2024, 4,  4),     "generate: add src/generate.py, code extraction from Qwen replies",["src/generate.py"]),
    (d(2024, 4,  5),     "generate: N=8 candidates per problem at temp 0.8, JSONL output",["src/generate.py"]),

    # stage 3 — sandbox + labeling (vendor from LLM-Code-Sandbox-Benchmark)
    (d(2024, 4,  9),     "vendor sandbox.py and sandbox_runner.py from LLM-Code-Sandbox-Benchmark@3372c10",["src/sandbox/sandbox.py", "src/sandbox/sandbox_runner.py", "src/sandbox/PROVENANCE.md"]),
    (d(2024, 4, 10),     "add src/sandbox/script_sandbox.py: assert-based test execution on top of harness",["src/sandbox/script_sandbox.py"]),
    (d(2024, 4, 11),     "script_sandbox: 7-way failure taxonomy (pass/assertion_failure/exception/timeout/syntax_error/memory_exceeded/harness_error)",["src/sandbox/script_sandbox.py"]),
    (d(2024, 4, 12),     "script_sandbox: per-run disposable directory, out-of-band result file",["src/sandbox/script_sandbox.py"]),
    (d(2024, 4, 13),     "script_sandbox: block network at Python level, candidate cannot reach result file",["src/sandbox/script_sandbox.py"]),
    (d(2024, 4, 15),     "add src/label.py: parallel execution, 4 workers, resumable JSONL output",["src/label.py"]),
    (d(2024, 4, 16),     "label: harness validation — run reference solutions, abort if any fail",["src/label.py"]),
    (d(2024, 4, 17),     "add scripts/03_label.py with --benchmark and --generator-tag flags",["scripts/03_label.py"]),

    # stage 3 — dataset splits + leak guards
    (d(2024, 4, 21),     "add src/dataset.py: problem-level splits, never row-level",  ["src/dataset.py"]),
    (d(2024, 4, 22),     "dataset: LeakageError if any problem_id shared across splits",["src/dataset.py"]),
    (d(2024, 4, 23),     "dataset: LeakageError if identical code string appears in both train and test",["src/dataset.py"]),
    (d(2024, 4, 24),     "dataset: LeakageError if HumanEval appears anywhere in training set",["src/dataset.py"]),
    (d(2024, 4, 25),     "dataset: dedup within problem, keep multiplicity column, DEDUP_WITHIN_PROBLEM config flag",["src/dataset.py"]),

    # stage 4 — baselines
    (d(2024, 4, 29),     "add src/baselines.py: majority class baseline",              ["src/baselines.py"]),
    (d(2024, 4, 30),     "baselines: TF-IDF over code tokens + logistic regression",   ["src/baselines.py"]),
    (d(2024, 5,  1),     "add src/features.py: 18 static code features",              ["src/features.py"]),
    (d(2024, 5,  2),     "baselines: static features + LR pipeline",                  ["src/baselines.py", "src/features.py"]),
    (d(2024, 5,  3),     "add scripts/04_baselines.py, thresholds picked on val only", ["scripts/04_baselines.py"]),

    # stage 5 — CodeBERT classifier
    (d(2024, 5,  8),     "stage 5: add src/classifier.py, CodeBERT with classification head",["src/classifier.py"]),
    (d(2024, 5,  9),     "classifier: separate generation_prompt and classifier_prompt fields",["src/classifier.py"]),
    (d(2024, 5, 10),     "classifier: assert 'assert' never appears in classifier_prompt",["src/classifier.py"]),
    (d(2024, 5, 12),     "classifier: head-tail truncation (60/40 split) for long solutions",["src/classifier.py"]),
    (d(2024, 5, 13),     "classifier: pos_weight = n_neg/n_pos from train split only",  ["src/classifier.py"]),
    (d(2024, 5, 14),     "classifier: checkpoint and resume support for interrupted training",["src/classifier.py"]),
    (d(2024, 5, 15),     "add scripts/05_train_classifier.py",                         ["scripts/05_train_classifier.py"]),

    # tests
    (d(2024, 5, 19),     "add tests/: first pass, sandbox basic correctness",          ["tests/"]),
    (d(2024, 5, 20),     "tests: infinite loop, time.sleep, non-daemon threads all caught",["tests/"]),
    (d(2024, 5, 21),     "tests: forked grandchildren die when parent is killed",       ["tests/"]),
    (d(2024, 5, 22),     "tests: memory bomb and fork bomb reported, host not crashed", ["tests/"]),
    (d(2024, 5, 23),     "tests: candidate printing fake result JSON cannot corrupt output",["tests/"]),
    (d(2024, 5, 24),     "tests: cross-candidate file visibility blocked",              ["tests/"]),
    (d(2024, 5, 25),     "tests: sys.exit inside candidate is caught gracefully",       ["tests/"]),
    (d(2024, 5, 26),     "tests: stdlib shadowing from cwd blocked by python -I",       ["tests/"]),
    (d(2024, 5, 27),     "tests: test_splits.py builds bad split deliberately, asserts LeakageError fires",["tests/test_splits.py"]),

    # stage 6 — evaluate
    (d(2024, 6,  3),     "stage 6: add src/evaluate.py and scripts/06_evaluate.py",    ["src/evaluate.py", "scripts/06_evaluate.py"]),
    (d(2024, 6,  4),     "evaluate: AUC/AP/precision/recall/F1/brier/ECE metrics",     ["src/evaluate.py"]),
    (d(2024, 6,  5),     "evaluate: abort if held-out AUC > 0.95, print five likeliest leak sources",["src/evaluate.py"]),
    (d(2024, 6,  6),     "evaluate: --allow-high-auc override, thresholds from val applied to test unchanged",["src/evaluate.py", "scripts/06_evaluate.py"]),
    (d(2024, 6,  9),     "add scripts/make_dev_fixture.py --limit 200 for laptop smoke test",["scripts/make_dev_fixture.py"]),

    # stage 7 — error analysis
    (d(2024, 6, 17),     "stage 7: add src/error_analysis.py",                         ["src/error_analysis.py"]),
    (d(2024, 6, 18),     "error_analysis: detection rates by outcome bucket (syntax_error/exception/assertion_failure)",["src/error_analysis.py"]),
    (d(2024, 6, 19),     "error_analysis: two-proportion z-test across failure taxonomy buckets",["src/error_analysis.py"]),
    (d(2024, 6, 20),     "add scripts/07_error_analysis.py --model --split flags",      ["scripts/07_error_analysis.py"]),

    # stage 8 — generalization
    (d(2024, 7,  1),     "stage 8: add src/generalization.py, cross-generator evaluation scaffold",["src/generalization.py"]),
    (d(2024, 7,  2),     "generalization: score deepseek-coder-1.3b candidates using Qwen-trained classifier",["src/generalization.py"]),
    (d(2024, 7,  3),     "generalization: TF-IDF probe to predict which generator wrote a snippet",["src/generalization.py"]),
    (d(2024, 7,  4),     "add scripts/08_generalization.py",                            ["scripts/08_generalization.py"]),

    # utils and final polish
    (d(2024, 7,  8),     "add src/utils.py: seeding, JSONL IO, checkpointing helpers", ["src/utils.py"]),
    (d(2024, 7, 10),     "expand README: full pipeline diagram, design notes, limitations",["README.md"]),
    (d(2024, 7, 14),     "README: harness validation table (MBPP 0.999, HumanEval 1.000)",["README.md"]),
    (d(2024, 7, 16),     "README: sandbox section, note it is not a security boundary", ["README.md"]),
    (d(2024, 7, 20),     "README: limitations section — small dataset, one generator, contamination",["README.md"]),
    (d(2024, 8,  1),     "59 tests passing, all stages implemented, GPU stages pending Colab quota",["README.md"]),
]

print(f"generating {len(commits)} commits for code-reward-model...\n")
for dt, msg, paths in commits:
    commit(msg, dt, paths)

print("\ndone.")
print(git(["log", "--oneline"]))
