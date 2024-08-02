# Provenance of the sandbox

`sandbox.py` and `sandbox_runner.py` are copied **unchanged** from:

- Repository: https://github.com/prx12345/LLM-Code-Sandbox-Benchmark
- Commit: `3372c100b3e344d5a2dfd98c93fc6d0f466f7b0a` (2026-08-29)
- Files: `src/sandbox.py`, `src/sandbox_runner.py`
- License: see the LICENSE file in that repository.

They are vendored rather than pip-installed because that project ships as a
loose `src/` directory with no package metadata, and because pinning a copy
means this repo's results cannot silently change when that repo does.

## Why it was extended rather than rewritten

The upstream harness was already doing the hard parts correctly:

- the child runs in a separate `python -I` interpreter, not a thread;
- `RLIMIT_AS`, `RLIMIT_CPU`, `RLIMIT_FSIZE`, `RLIMIT_NOFILE` and `RLIMIT_NPROC`
  are all applied in the child;
- the parent enforces a wall-clock timeout and kills the whole **process
  group** (`start_new_session=True` + `killpg`), so a candidate that forks
  cannot leave orphans behind;
- `SIGXCPU` / `SIGKILL` / `SIGXFSZ` child deaths are decoded into meaningful
  statuses instead of a generic failure;
- candidate source is `compile()`d directly rather than imported, which avoids
  a real and subtle bug: `.pyc` caches are keyed on mtime-seconds plus file
  size, so rapidly rewriting a same-length candidate file can execute stale
  bytecode.

That is more care than most such harnesses get, so rewriting it would have
been a downgrade. What it could not do was run *assert-style* tests, which is
the entire ground-truth format of MBPP and HumanEval — its contract is
"call `f(*args)` and JSON-compare the return value". The extension in
`script_sandbox.py` adds that, plus a failure taxonomy, a disposable per-run
working directory, out-of-band result delivery, and a network block. Those are
documented inline in `script_sandbox.py`.

## What it is still not

It is not a security boundary. Documented in `script_sandbox.py`'s module
docstring and in the README's Safety section.
