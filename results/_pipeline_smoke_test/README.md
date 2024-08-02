# THESE ARE NOT RESULTS

Everything in this directory was produced by `scripts/make_dev_fixture.py`,
whose "candidates" are **deterministic mutations of the benchmarks' own
reference solutions** — not samples from a language model.

Its only purpose is to exercise stages 3 through 7 end to end on a CPU in a few
minutes, so that a bug in splitting, baselines or evaluation surfaces before
you spend hours of Colab time on generation. It caught two real bugs while this
repo was being written (see the README's "Bugs found during development").

Any AUC, precision or recall in these files describes how separable
hand-written mutants are. It says nothing about the research question and must
never be quoted as a result. The real results go in `results/` one level up,
once `scripts/02_generate.py` and `scripts/05_train_classifier.py` have run.
