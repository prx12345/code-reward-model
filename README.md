# Code Correctness Reward Model

Can a small model tell whether generated code will pass its tests, without
running it? That's the question here. The labels aren't downloaded — I sample
solutions from a code model, execute every one against the real unit tests in a
sandbox, and use pass/fail as ground truth.

## Status

Not finished. The pipeline is built and tested; the classifier isn't trained yet.

| Stage | |
|---|---|
| 1. Data loading + normalisation | done |
| 2. Candidate generation (Qwen2.5-Coder-1.5B) | **not run** |
| 3. Sandboxed execution + labeling | done, on reference solutions and a synthetic fixture |
| 4. Baselines | not run on real data |
| 5. CodeBERT classifier | **not run** |
| 6. Evaluation | not run on real data |
| 7. Error analysis | not run on real data |
| 8. Generalization across generators | **not run** |
| Tests | 59 passing |

Stages 2 and 5 need a GPU and the HuggingFace hub, and I've been out of Colab
quota. MBPP and HumanEval load from their upstream GitHub files, so everything
that doesn't need model weights does work.

**No classifier numbers appear below, because I haven't measured any.** The
results table is empty on purpose. Anything in `results/_pipeline_smoke_test/`
came from synthetic mutants of reference solutions, not model output — see the
warning file in that directory.

## Pipeline

```
MBPP (963 problems)        HumanEval (164, HELD OUT)
        │                             │
        ├────────── normalise to one schema ──────────┤
        ▼                                             ▼
  Qwen2.5-Coder-1.5B-Instruct, N=8 @ temp 0.8  ───────┤
        ▼                                             ▼
  sandboxed execution against the real tests  ────────┤
  → pass / assertion_failure / exception / timeout /
    syntax_error / memory_exceeded / harness_error
        ▼                                             │
  split BY PROBLEM ID (never by row)                  │
        ├── majority baseline                         │
        ├── TF-IDF over code tokens + LR              │
        ├── static features + LR                      │
        └── CodeBERT: [problem] </s> [code] → P(pass) │
                     └──── evaluate ◄─────────────────┘
```

The executor records *why* a candidate failed, not just whether it did. That's
what makes the error analysis possible.

## What I've measured

### Dataset

| | MBPP | HumanEval |
|---|---|---|
| Problems | 963 | 164 |
| Role | train / val / test | held out |
| Problem statement | natural language | signature + docstring |
| Tests per problem | 3.0 | 1 (`check()` wrapping many asserts) |
| Mean prompt length | 78.5 chars | 448.5 chars |
| Mean reference solution | 175.6 chars | 631.5 chars |

MBPP ships 974. Ids 1–10 are the standard few-shot examples so they're dropped,
and one more is dropped because its asserts wouldn't parse.

### Harness validation

Reference solutions run against their own tests. If a canonical solution fails,
the harness is broken and every label downstream is garbage.

| Benchmark | Problems | Pass rate | Failures |
|---|---|---|---|
| MBPP | 963 | 0.999 | 1 — `mbpp/123`, a real timeout (O(n²) amicable-number search) |
| HumanEval | 164 | 1.000 | 0 |

~24 candidates/sec on one CPU core, 4 workers.

### Tests

```
$ python -m pytest tests/ -q
59 passed
```

17 of those hammer the executor: infinite loops, `time.sleep`, forked
grandchildren, non-daemon threads, memory bombs, network calls, candidates
printing fake result JSON, cross-candidate file visibility, `sys.exit`, and
trying to shadow a stdlib module from the cwd.

### Classifier results

Not measured. Filled in from `results/metrics.json` once stage 6 runs.

| Model | AUC | AP | Precision | Recall | F1 | Accuracy | Brier | ECE |
|---|---|---|---|---|---|---|---|---|
| Majority class | — | — | — | — | — | — | — | — |
| Static features + LR | — | — | — | — | — | — | — | — |
| TF-IDF + LR | — | — | — | — | — | — | — | — |
| CodeBERT | — | — | — | — | — | — | — | — |

### Error analysis

Not measured. The hypothesis, written before running anything:

> It catches obvious breakage — syntax errors, wrong API usage, code that
> raises — and misses subtle logic bugs: off-by-one, wrong edge case, right
> shape and wrong answer.

The outcome taxonomy gives the split for free. `syntax_error` and `exception`
broke visibly; `assertion_failure` means the code ran and returned the wrong
thing. `scripts/07_error_analysis.py` compares detection rates across those
buckets with a two-proportion z-test. I don't know the answer yet and I'm not
guessing at it here.

### Generalization

Not measured. `scripts/08_generalization.py` scores candidates from
`deepseek-coder-1.3b-instruct` using the Qwen-trained classifier, on the same
held-out problems so only the generator changes. It also trains a TF-IDF probe
to predict *which* generator wrote a snippet — if that's easy, any transfer drop
has an obvious cause.

## Design notes

**Splitting by problem, with a guard that crashes.** Eight candidates per
problem, many near-identical. A random row-level split puts siblings of a test
row into training and the model scores brilliantly by recognising the problem
instead of judging the code. So the split is by `problem_id`, and three guards
raise `LeakageError` rather than warn: no shared problem id, no identical code
string across splits, no HumanEval anywhere in training. `tests/test_splits.py`
builds the bad split deliberately and asserts the guard fires.

**Tests go in the generation prompt, not the classifier prompt.** MBPP's task
text doesn't pin down the signature — "find the kth element in the given array"
has a reference taking `(arr, n, k)`. Without the asserts, nearly everything
fails on arity and the labels measure signature-guessing. But the classifier
must not see them: they're the label source, and a model reading
`assert kth_element(...) == 3` next to the code is running the tests by proxy.
`generation_prompt` and `classifier_prompt` are separate fields, and a test
asserts the word `assert` never appears in the second.

**Aborting on a suspicious AUC.** `scripts/06_evaluate.py` stops if any held-out
AUC exceeds 0.95 and prints the five likeliest leak sources. You override it
with `--allow-high-auc`, deliberately. A number above that here would beat
published reward models trained on far more data — a leak is the better bet.

Shorter ones: problem statements are capped at 128 tokens and code gets the
rest, with head-tail truncation (first ~60%, last ~40%) because a function's
return and its off-by-one live at the end. `pos_weight = n_neg/n_pos` from train
only. Exact duplicate candidates within a problem are collapsed, with the count
kept as a `multiplicity` column — arguable, since a real reward model scores
duplicates too; `config.DEDUP_WITHIN_PROBLEM = False` flips it. Thresholds are
picked on val and applied unchanged to test.

## The sandbox

Every candidate runs in its own subprocess with a wall-clock timeout enforced by
the parent (which kills the whole process group, so forked grandchildren die
too), `RLIMIT_CPU`, `RLIMIT_AS` at 512 MB, plus `RLIMIT_FSIZE`/`NOFILE`/`NPROC`.
It runs under `python -I`, in a fresh disposable working directory that's
destroyed afterwards, with results written out of band to a file the candidate
can't reach — a candidate that prints, or prints fake result JSON, can't corrupt
the channel. Network is blocked at the Python level.

**It is not a security boundary.** Same uid, readable filesystem, and the
network block is a monkeypatch `ctypes` would walk straight past. That's the
right trade for a 1.5B model's output on a throwaway VM, and the wrong one for
anything hostile.

`src/sandbox/sandbox.py` and `sandbox_runner.py` are vendored unchanged from my
earlier [LLM-Code-Sandbox-Benchmark](https://github.com/prx12345/LLM-Code-Sandbox-Benchmark)
at commit `3372c10` — details in `src/sandbox/PROVENANCE.md`. I extended rather
than rewrote because the hard parts were already right. What it couldn't do was
run assert-style tests, which is the entire ground-truth format of both
benchmarks; `script_sandbox.py` adds that plus the failure taxonomy, the
per-run directory, out-of-band results and the network block.

## Limitations

**963 problems is small.** After sampling and dedup the training set is a few
thousand rows — enough to fine-tune an encoder, nowhere near enough for
something deployable.

**One generator.** All training data comes from a single 1.5B model, so the
classifier may have learned that model's habits rather than what correct code
looks like. Stage 8 measures how bad this is; it doesn't fix it.

**Contamination.** MBPP and HumanEval almost certainly sit in the pretraining
data of every model used here. That inflates pass rates and may make the
failures unrepresentative. Nothing here corrects for it and nothing can, short
of a private benchmark.

**Prompt asymmetry.** MBPP's classifier input is bare natural language;
HumanEval's is a signature with doctests inside it. So the held-out number is
measuring a slightly different task, and part of any gap is that rather than
generalization. Known flaw, not an oversight.

**Binary labels throw information away.** "2 of 3 tests passed" becomes `False`.
A regression head on pass fraction is the first thing I'd change.

**Three asserts is a weak test.** Code passing three assertions isn't correct
code, it's code that passes three assertions. Some positive labels are simply
wrong.

To be actually useful this would need far more problems from an uncontaminated
source, candidates from several generators, pass-fraction labels, a longer
context so solutions aren't truncated, and evaluation as a *reranker* — does
best-of-n using this score beat picking at random — since that's the real use.

## Running it

Colab instructions, timings and resume behaviour: [COLAB.md](COLAB.md).

```bash
git clone https://github.com/prx12345/code-reward-model.git
cd code-reward-model
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

No GPU needed for these:

```bash
python scripts/01_prepare_data.py --validate-harness
python -m pytest tests/ -q
```

Generation and training need a GPU and the HF hub; everything between them
doesn't:

```bash
python scripts/02_generate.py --benchmark mbpp        # GPU
python scripts/02_generate.py --benchmark humaneval   # GPU
python scripts/03_label.py --benchmark mbpp           # CPU
python scripts/03_label.py --benchmark humaneval      # CPU
python scripts/04_baselines.py                        # CPU
python scripts/05_train_classifier.py                 # GPU
python scripts/06_evaluate.py                         # CPU
python scripts/07_error_analysis.py --model codebert --split test
```

Optional, cross-generator:

```bash
python scripts/02_generate.py --benchmark mbpp --generator secondary
python scripts/03_label.py    --benchmark mbpp --generator secondary
python scripts/08_generalization.py
```

Every long stage resumes — re-run the same command after an interruption. Seed
is 1337, set in `config.py` along with every other hyperparameter.

To exercise stages 3–7 on a laptop in a few minutes:

```bash
python scripts/make_dev_fixture.py --limit 200
python scripts/03_label.py --benchmark mbpp --generator-tag devfixture-mutants
python scripts/04_baselines.py --generator-tag devfixture-mutants
python scripts/06_evaluate.py --generator-tag devfixture-mutants
```

That produces **no results** — see `results/_pipeline_smoke_test/README.md`.

## Layout

```
config.py          every hyperparameter, path and seed
src/
  data.py          MBPP/HumanEval loading, normalisation, HumanEval guard
  generate.py      resumable sampling, code extraction from replies
  sandbox/         vendored harness + the assert-based extension
  label.py         execution → labels, parallel and resumable
  dataset.py       dedup, problem-level splits, leak guards
  features.py      18 static code features
  baselines.py     majority / TF-IDF+LR / static+LR
  classifier.py    CodeBERT, head-tail truncation, checkpoint+resume
  evaluate.py      metrics, thresholds, suspicious-AUC guard, plots
  error_analysis.py
  generalization.py
  utils.py         seeding, JSONL IO, checkpointing
scripts/01..08     one entry point per stage, run in order
tests/             59 tests
```

## License

MIT.
