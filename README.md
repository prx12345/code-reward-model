# Code Correctness Reward Model

A classifier that predicts whether LLM-generated Python code will pass its unit
tests — **without executing it**. The training labels come from actually
executing every candidate against the real tests in a sandbox, so the dataset
is produced here rather than downloaded.

---

## ⚠️ Status: pipeline complete, model not yet trained

Read this before anything else in the repo.

| Stage | Status |
|---|---|
| 1. Data loading + normalization | ✅ **Ran.** Real numbers below. |
| 2. Candidate generation (Qwen2.5-Coder-1.5B) | ⬜ **Not yet run.** Code written and unit-tested. |
| 3. Sandboxed execution + labeling | ✅ **Ran** on reference solutions and on a synthetic fixture. Not yet run on model samples. |
| 4. Baselines (majority / TF-IDF / static features) | ⬜ **Not yet run on real data.** Code path verified on a fixture. |
| 5. CodeBERT classifier | ⬜ **Not yet run.** |
| 6. Evaluation (AUC, PR, calibration, plots) | ⬜ **Not yet run on real data.** Code path verified on a fixture. |
| 7. Error analysis | ⬜ **Not yet run on real data.** Code path verified on a fixture. |
| 8. Generalization across generators | ⬜ **Not yet run.** |
| Test suite | ✅ **59 tests passing.** |

**Why:** the environment this was built in has no GPU and no network route to
the Hugging Face hub, so neither `Qwen/Qwen2.5-Coder-1.5B-Instruct` nor
`microsoft/codebert-base` could be downloaded. MBPP and HumanEval *are*
reachable from their upstream GitHub homes, which is why the data and execution
stages are real and the model stages are not.

**There are no classifier metrics in this README because none have been
measured.** No AUC, no precision, no recall, no illustrative or placeholder
numbers anywhere. The results table below is empty and stays empty until
[`COLAB.md`](COLAB.md) has been run end to end.

Everything in `results/_pipeline_smoke_test/` was produced from *synthetic
mutants of reference solutions*, not model output. It exists to prove the code
runs. It is not a result. See the warning file in that directory.

---

## The problem

If you sample eight solutions from a code model and want the best one, the
honest way to pick is to run all eight against the tests. That is expensive,
requires the tests to exist, and requires executing untrusted code. A reward
model that scores code statically is the cheap substitute — it is what drives
best-of-n sampling, RLHF reward signals, and candidate ranking in coding
agents.

The question this project asks: **how well can a small encoder do that, and
which kinds of bugs does it miss?**

The second half of that question is the interesting one, and it is why the
executor records *why* a candidate failed rather than just whether it did.

## Pipeline

```
MBPP (963 problems)        HumanEval (164 problems, HELD OUT)
        │                             │
        ├──────────── normalize to one schema ────────────┤
        │                                                  │
        ▼                                                  ▼
  Qwen2.5-Coder-1.5B-Instruct, N=8 @ temp 0.8  ────────────┤
        │                                                  │
        ▼                                                  ▼
  sandboxed execution against the real tests  ─────────────┤
  → pass / assertion_failure / exception / timeout /
    syntax_error / memory_exceeded / harness_error
        │                                                  │
        ▼                                                  │
  split BY PROBLEM ID (never by row)                       │
        │                                                  │
        ├── majority baseline                              │
        ├── TF-IDF over code tokens + logistic regression   │
        ├── static features + logistic regression           │
        └── CodeBERT: [problem] </s> [code] → P(pass)       │
                     │                                      │
                     └──── evaluate ◄───────────────────────┘
```

## Measured results

Only stages that actually ran appear here.

### Dataset

| | MBPP | HumanEval |
|---|---|---|
| Problems | 963 | 164 |
| Role | train / val / test | **held out, never trained on** |
| Problem statement | natural language | function signature + docstring |
| Tests per problem | 3.0 | 1 (`check()` covering many asserts) |
| Mean prompt length | 78.5 chars | 448.5 chars |
| Mean reference solution | 175.6 chars | 631.5 chars |

MBPP ships 974 problems. Task ids 1–10 are the canonical few-shot prompting
examples used by every paper reporting MBPP numbers, so they are excluded; one
further problem is dropped because its assertions could not be parsed to
recover the target function name. 974 − 10 − 1 = 963.

### Harness validation

Every benchmark's own reference solution, executed against its own tests. This
is the number that says the labels can be trusted: if a canonical solution
fails, the bug is in the harness, and every label downstream would be wrong.

| Benchmark | Problems | Reference pass rate | Failures |
|---|---|---|---|
| MBPP | 963 | **0.999** | 1 — `mbpp/123`, a genuine timeout (the reference is an O(n²) amicable-number search) |
| HumanEval | 164 | **1.000** | 0 |

Throughput: ~24 candidates/sec on one CPU core with 4 workers.

### Test suite

**59 tests passing**, 17 of them adversarial tests of the executor:

```
$ python -m pytest tests/ -q
59 passed
```

The executor tests cover infinite loops, `time.sleep`, forked grandchildren
outliving their parent, non-daemon threads left running, memory bombs, outbound
network attempts, candidates printing fake result JSON to stdout, candidates
writing files, cross-candidate file visibility, `sys.exit`, and an attempt to
shadow a stdlib module from the working directory.

### Classifier results

**Not yet measured.** This table is filled in from `results/metrics.json` after
`scripts/06_evaluate.py` runs.

| Model | AUC | AP | Precision | Recall | F1 | Accuracy | Brier | ECE |
|---|---|---|---|---|---|---|---|---|
| Majority class | — | — | — | — | — | — | — | — |
| Static features + LR | — | — | — | — | — | — | — | — |
| TF-IDF + LR | — | — | — | — | — | — | — | — |
| CodeBERT | — | — | — | — | — | — | — | — |

### Error analysis

**Not yet measured.** The hypothesis under test, stated in advance:

> The classifier catches obvious breakage — syntax errors, wrong API usage,
> code that raises — and misses subtle logic bugs: off-by-one, wrong edge case,
> right shape and wrong answer.

The execution stage gives exactly the split needed to test it. A candidate that
fails with `syntax_error` or `exception` broke *visibly*; one that fails with
`assertion_failure` ran to completion and returned the wrong value — that is the
subtle-bug bucket, by construction. `scripts/07_error_analysis.py` compares
detection rates between the two buckets with a two-proportion z-test and writes
a verdict of SUPPORTED / NOT SUPPORTED / CONTRADICTED. **The verdict is not yet
known and is not guessed at here.**

### Generalization

**Not yet measured.** `scripts/08_generalization.py` scores candidates from
`deepseek-coder-1.3b-instruct` (a different model family, not a Qwen sibling)
using the classifier trained on Qwen output, restricted to the same held-out
problems so generator is the only thing that varies. It also runs a fingerprint
probe: a TF-IDF classifier trained to predict *which generator* wrote a snippet.
If generator identity is trivially readable, any transfer drop has an obvious
mechanism; if it is not, style was never an available shortcut.

---

## Design decisions worth defending

### Splitting by problem, with a guard that crashes

Eight candidates are sampled per problem and many are near-identical. A random
row-level split puts siblings of a test row into training, and the model scores
beautifully by recognising the problem rather than judging the code. This is
the easiest way to fake a good result on this task.

So splitting is by `problem_id`, and three guards run on every split — each
raises `LeakageError` rather than warning:

1. no `problem_id` in more than one split;
2. no identical code string across splits (reported by default, fatal under
   `strict_code=True`);
3. no HumanEval problem in any training split.

`tests/test_splits.py` includes a test that constructs the row-level split
deliberately and asserts the guard fires.

### The generation prompt contains the tests; the classifier prompt does not

MBPP's task text ("Write a function to find the kth element in the given
array") does not pin down the signature — the reference takes `(arr, n, k)`.
Without the asserts in the prompt, nearly every sample fails on arity and the
labels measure signature-guessing rather than correctness. Every published MBPP
number is produced with the tests or few-shot examples in the prompt for this
reason.

The classifier must **not** see the assertions: they are the label source. A
model reading `assert kth_element(...) == 3` next to the code is doing test
execution by proxy. `src/data.py` keeps `generation_prompt` and
`classifier_prompt` as separate fields, and `tests/test_data.py` asserts that
the word `assert` never appears in the latter.

### Head-tail truncation for code over 512 tokens

The problem statement is capped at 128 tokens and the code gets the remaining
budget. When code exceeds it, the first ~60% and last ~40% are kept and spliced.

The usual default — keep the first 512 tokens, drop the tail — is close to the
worst possible choice here. A function's `return`, its base case and its
off-by-one are usually at the *end*. The fraction of examples affected is
measured at training time and written to `results/training_report.json` rather
than assumed to be small.

### Class imbalance

`pos_weight = n_negative / n_positive`, computed on the training split only and
applied unconditionally to `BCEWithLogitsLoss`. On a balanced set this is ~1.0
and changes nothing; on a skewed one it prevents collapse to the majority class.
The baselines use `class_weight="balanced"` so the comparison is between
architectures, not between imbalance strategies. The observed balance is
reported by `scripts/03_label.py`.

### Deduplication

Temperature-0.8 sampling produces exact duplicates within a problem. They are
collapsed and the count kept as a `multiplicity` column, because keeping them
makes metrics reflect how often the sampler repeats itself rather than how well
the classifier discriminates.

**This is arguable** and I want to flag it rather than bury it: a reward model
in production scores every sample, duplicates included, so there is a real case
for leaving them in. `config.DEDUP_WITHIN_PROBLEM = False` runs it the other way.

### Threshold selection

Chosen on **validation** by maximising F1, then applied unchanged to test and
to the held-out set. Tuning the threshold on the set you report is a small,
extremely common and entirely real form of leakage.

### The too-good-to-be-true guard

`scripts/06_evaluate.py` **aborts** if any model's held-out AUC exceeds 0.95,
printing the five most likely leak sources in the order worth checking. It has
to be overridden explicitly with `--allow-high-auc`. A result above that bar on
this task would beat published code reward models trained on far more data; a
leak is the likelier story and the pipeline refuses to report the number until
someone has looked.

---

## Safety: the sandbox

We execute code sampled from a language model. Each candidate runs in its own
subprocess with:

- a **wall-clock timeout** enforced by the parent, which kills the entire
  process group (`start_new_session` + `killpg`) so forked grandchildren die too;
- **`RLIMIT_CPU`**, so a busy loop dies even if the parent is starved;
- **`RLIMIT_AS`** (512 MB), turning runaway allocation into a clean `MemoryError`
  instead of an OOM-killed session;
- **`RLIMIT_FSIZE` / `RLIMIT_NOFILE` / `RLIMIT_NPROC`**, against disk filling,
  descriptor hoarding and fork bombs;
- **`python -I`** — no `PYTHONPATH`, no user site-packages, no inherited
  environment, and the working directory is off `sys.path` so a candidate
  cannot shadow a stdlib module by writing `json.py` next to itself;
- a **fresh disposable working directory per candidate**, destroyed afterwards;
- **results delivered out of band** to a file the candidate's working directory
  cannot reach, so a candidate that prints — or prints fake result JSON — cannot
  corrupt the channel;
- a **Python-level network block**.

### What it is not

**This is not a security boundary.** It is a robust *resource and accident*
sandbox. The child runs as the same uid against a readable filesystem, and the
network block is a monkeypatch that `ctypes` would defeat. That is the right
trade for code sampled from a 1.5B model on a disposable Colab VM. It would be
the wrong trade for code from an adversary, which needs a container, a network
namespace, or gVisor.

### Relationship to my earlier project

`src/sandbox/sandbox.py` and `src/sandbox/sandbox_runner.py` are vendored
**unchanged** from [LLM-Code-Sandbox-Benchmark](https://github.com/prx12345/LLM-Code-Sandbox-Benchmark)
at commit `3372c10`. Full provenance and reasoning in
[`src/sandbox/PROVENANCE.md`](src/sandbox/PROVENANCE.md).

It was extended rather than rewritten because it already got the hard parts
right — separate interpreter, the full RLIMIT set, process-group kill, and
compiling candidate source directly instead of importing it (which avoids a
subtle stale-`.pyc` bug: bytecode caches are keyed on mtime-seconds plus file
size, so rapidly rewriting a same-length candidate file can execute stale code).

What it could not do was run **assert-style tests**, which is the entire
ground-truth format of MBPP and HumanEval — its contract is "call `f(*args)`
and JSON-compare the return value." `src/sandbox/script_sandbox.py` adds that,
plus the failure taxonomy, the per-run disposable directory (upstream shares
`/tmp` across candidates), out-of-band result delivery (upstream parses stdout),
and the network block.

---

## Limitations

Stated plainly, because the gap between this and something genuinely useful is
the most interesting part of the write-up.

**Dataset size.** 963 problems is small. After an 8-candidate sample and
deduplication the training set is on the order of a few thousand rows — enough
to fine-tune an encoder, nowhere near enough to train a reward model you would
deploy.

**A single generator.** The training data comes from one 1.5B model. A reward
model that has learned one policy's habits is worthless against a different
policy, which is exactly what a reward model is for. Stage 8 measures how bad
this is; it does not fix it. Fixing it needs several generators of different
sizes and families mixed into training.

**Benchmark contamination.** MBPP and HumanEval predate every model used here
and are near-certainly in their pretraining data. That inflates the generator's
pass rate and may make its failures unrepresentative of failures on genuinely
novel problems. Nothing in this project corrects for it, and it cannot be
corrected for without a private benchmark.

**MBPP/HumanEval prompt asymmetry.** MBPP's classifier input is bare natural
language. HumanEval's is a function signature *with doctests in it*. The
held-out number is therefore measuring a slightly different task than the
in-distribution number, and part of any gap between them is this asymmetry
rather than generalization. This is a known flaw in the design, not an oversight.

**Binary labels discard information.** "2 of 3 tests passed" is thrown away in
favour of `False`. A regression head on pass *fraction* would be a strictly
richer signal and is the first thing I would change.

**Weak tests mean noisy labels.** MBPP has three assertions per problem. Code
that passes three assertions is not correct code; it is code that passes three
assertions. Some fraction of the positive labels are wrong in a way no amount of
modelling can fix.

**What this would need to be genuinely useful:** an order of magnitude more
problems from a non-contaminated source; candidates from a mix of generators;
pass-fraction rather than binary labels; a decoder-based scorer with a longer
context so long solutions are not truncated; and evaluation as a *reranker*
(does best-of-n selection using this score beat random selection, and by how
much) rather than as a classifier, since reranking is the actual downstream use.

---

## Reproduction

Full Colab instructions, timings and resume behaviour: [`COLAB.md`](COLAB.md).

```bash
git clone https://github.com/prx12345/code-reward-model.git
cd code-reward-model
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Stages 1 and 3 need no GPU and no HF access — they run on a laptop.
python scripts/01_prepare_data.py --validate-harness
python -m pytest tests/ -q

# Stages 2 and 5 need a GPU and the HF hub.
python scripts/02_generate.py --benchmark mbpp
python scripts/02_generate.py --benchmark humaneval
python scripts/03_label.py --benchmark mbpp
python scripts/03_label.py --benchmark humaneval
python scripts/04_baselines.py
python scripts/05_train_classifier.py
python scripts/06_evaluate.py
python scripts/07_error_analysis.py --model codebert --split test

# Optional: generalization across generators.
python scripts/02_generate.py --benchmark mbpp --generator secondary
python scripts/03_label.py    --benchmark mbpp --generator secondary
python scripts/08_generalization.py
```

Every long stage is resumable — re-run the same command after an interruption.
Seed is fixed at 1337 in `config.py`, which is also the single place any
hyperparameter lives.

To exercise stages 3–7 on a laptop in a few minutes with no GPU:

```bash
python scripts/make_dev_fixture.py --limit 200
python scripts/03_label.py --benchmark mbpp --generator-tag devfixture-mutants
python scripts/04_baselines.py --generator-tag devfixture-mutants
python scripts/06_evaluate.py --generator-tag devfixture-mutants
```

This produces **no results** — see `results/_pipeline_smoke_test/README.md`.

## Layout

```
config.py                    every hyperparameter, path and seed
src/
  data.py                    MBPP/HumanEval loading, normalization, HF guard
  generate.py                resumable sampling, code extraction from replies
  sandbox/
    sandbox.py               vendored, unchanged (see PROVENANCE.md)
    sandbox_runner.py        vendored, unchanged
    script_sandbox.py        assert-based execution + failure taxonomy
    script_runner.py         child process
  label.py                   execution → labels, parallel and resumable
  dataset.py                 dedup, problem-level splits, leakage guards
  features.py                18 static code features
  baselines.py               majority / TF-IDF+LR / static+LR
  classifier.py              CodeBERT, head-tail truncation, checkpoint+resume
  evaluate.py                metrics, thresholds, suspicious-AUC guard, plots
  error_analysis.py          hypothesis test, length confound, examples
  generalization.py          cross-generator transfer + fingerprint probe
  utils.py                   seeding, JSONL IO, checkpointing
scripts/01..08               one entry point per stage, run in order
tests/                       59 tests
results/                     committed outputs
```

## Bugs found during development

Kept because they are more informative than a clean history.

1. **`python -I` implies `-P`**, so the script's own directory is not added to
   `sys.path` and the child could not import the vendored runner. Fixed by
   adding the one trusted directory explicitly, rather than dropping isolation.
2. **A long-lived append file descriptor is invalidated across `subprocess`
   calls** on FUSE-backed mounts — observed as `OSError: [Errno 9] Bad file
   descriptor` partway into a 1600-candidate run. `JsonlAppender` now opens,
   appends and closes per record; that costs about a millisecond against a
   ~50 ms subprocess and makes checkpointing immune to the same failure on
   Colab's Drive mounts. Found by the dev fixture.
3. **`dir(__builtins__)` inside a module returns dict methods, not builtins**,
   so `infer_entry_point` returned `"len"` or `"set"` for MBPP assertions that
   wrap the call. Found by a parametrized test in `tests/test_data.py`.

## Deviations from the original spec

- **Stages 2, 5 and 8 have not been run**, for the environment reasons in the
  status section. The code is written and unit-tested; no metrics are reported
  for them.
- **MBPP and HumanEval are loaded from their upstream GitHub files**, with the
  `datasets` hub tried first and fallen back from. The hub was unreachable, and
  the upstream files are what the published MBPP numbers were computed against.
- **A second sandbox entry point was added** rather than modifying the vendored
  one, so the original files stay byte-identical to the earlier project.
- **A synthetic dev fixture was added** (`scripts/make_dev_fixture.py`), not in
  the original plan. It exists because stages 4–7 could not otherwise be
  exercised without a GPU, and it immediately earned its place by finding bug 2
  above.

## For a CV

Lines using **only numbers measured in this repo**. The classifier has not been
trained, so nothing about AUC or model performance appears here.

- Built an execution-based labeling harness for LLM-generated Python, executing
  candidate solutions against MBPP and HumanEval unit tests in an isolated
  subprocess and classifying outcomes into seven failure types; validated at
  99.9% (963 problems) and 100% (164 problems) on the benchmarks' own reference
  solutions.
- Hardened an untrusted-code sandbox with wall-clock and CPU timeouts,
  address-space and process limits, process-group kill for forked children,
  per-run disposable working directories and out-of-band result delivery;
  verified by 17 adversarial tests covering infinite loops, fork escapes, memory
  bombs, network attempts and stdout corruption.
- Designed a fully resumable, checkpointed ML pipeline for unreliable free-tier
  compute, with problem-level train/test splitting enforced by assertions that
  abort on leakage and an evaluation stage that refuses to report an AUC above
  0.95 without explicit override; 59 tests passing.

Once `COLAB.md` has been run, `results/metrics.json` supplies the numbers for a
fourth line about the classifier itself. **Do not write that line before then.**

## License

MIT.
