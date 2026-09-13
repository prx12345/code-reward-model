# Running this on free-tier Colab

Free Colab sessions die without warning. Every long stage here checkpoints to
disk and resumes, so the recovery procedure is always the same: **re-run the
exact same command.** Nothing is recomputed.

## Setup (once per session)

Runtime → Change runtime type → **T4 GPU**.

```python
from google.colab import drive
drive.mount('/content/drive')

%cd /content/drive/MyDrive
!git clone https://github.com/prx12345/code-reward-model.git
%cd code-reward-model
!pip install -q -r requirements.txt
```

Cloning into Drive rather than `/content` is the whole trick: `/content` is
wiped when the session ends, Drive is not. The checkpoint files survive and the
next session picks up where the last one died.

## The stages, in order

```bash
# 1. Data — seconds, CPU only.
python scripts/01_prepare_data.py --validate-harness

# 2. Generation — the long one. Resumable.
python scripts/02_generate.py --benchmark mbpp
python scripts/02_generate.py --benchmark humaneval

# 3. Execution + labeling — CPU bound, resumable.
python scripts/03_label.py --benchmark mbpp
python scripts/03_label.py --benchmark humaneval

# 4. Baselines — seconds.
python scripts/04_baselines.py

# 5. Classifier — GPU. Resumable from models/codebert_checkpoint.pt.
python scripts/05_train_classifier.py

# 6. Evaluation — writes the comparison table and every plot.
python scripts/06_evaluate.py

# 7. Error analysis.
python scripts/07_error_analysis.py --model codebert --split test

# 8. Generalization (optional but it is the most interesting number).
python scripts/02_generate.py --benchmark mbpp --generator secondary
python scripts/03_label.py    --benchmark mbpp --generator secondary
python scripts/08_generalization.py
```

## Rough timings on a T4

These are estimates from the model sizes and the measured execution throughput
(~24 candidates/sec on one CPU core), not from a completed run. Treat them as
planning numbers, not measurements.

| Stage | Scale | Estimate |
|---|---|---|
| 01 data | 1127 problems | under a minute |
| 02 generation, MBPP | 963 × 8 = 7704 samples | 3–5 hours |
| 02 generation, HumanEval | 164 × 8 = 1312 samples | 40–60 min |
| 03 labeling | ~9000 subprocesses | 15–30 min |
| 04 baselines | — | under a minute |
| 05 classifier | ~6500 rows × 3 epochs | 20–40 min |
| 06–08 | — | a few minutes |

Generation dominates. Split it across sessions without worrying — it resumes
per problem, and a problem that got 3 of its 8 samples before the session died
is topped up to 8, not restarted.

## If a session dies

Re-run the same command. That is the entire procedure.

To confirm it resumed rather than restarted, check the log line each stage
prints on startup:

```
generation: 412/963 problems complete, 551 to do
labeling: 3296 already done, 4408 to run
resumed from codebert_checkpoint.pt: epoch 1, step 400, best_auc 0.7314
```

## Things that will bite you

- **Out of memory during generation.** Lower `GEN_BATCH_SIZE` in `config.py`
  from 8 to 4. Sampling still produces 8 candidates per problem, just in two
  passes.
- **`06_evaluate.py` aborts with `SuspiciousResultError`.** That is the guard
  working, not a crash. Read what it prints and check the five listed causes
  before reaching for `--allow-high-auc`.
- **Drive I/O is slow.** The JSONL writers open-append-close per record, which
  is deliberate (see `src/utils.JsonlAppender`) and costs roughly a millisecond
  against a ~50 ms subprocess. If it becomes a real bottleneck, run with the
  repo in `/content` and copy checkpoints to Drive between stages.
- **Do not run `scripts/make_dev_fixture.py` on Colab.** It is a laptop
  shakedown tool. Its output is not data.
