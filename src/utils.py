"""Small shared helpers: seeding, JSONL IO, resumable checkpointing, logging.

The checkpointing model used everywhere in this repo is deliberately the
dumbest one that survives a Colab session dying mid-stage:

    append-only JSONL + a stable record id + "skip ids already on disk"

No sqlite, no lock files, no resume manifests. A partially written final line
is the only failure mode, and :func:`read_jsonl` drops it (see ``tolerant``).
"""

from __future__ import annotations

import json
import os
import random
import sys
import threading
import time
from pathlib import Path
from typing import Any, Iterable, Iterator


# --------------------------------------------------------------------------
# Seeding
# --------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Seed every RNG we might touch. Torch/numpy are seeded only if present."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # Determinism costs throughput but makes "did my change help?" answerable.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


# --------------------------------------------------------------------------
# JSONL
# --------------------------------------------------------------------------

def read_jsonl(path: str | Path, tolerant: bool = True) -> list[dict[str, Any]]:
    """Read a JSONL file into a list of dicts.

    Args:
        path: File to read. A missing file yields ``[]`` -- that is the normal
            "nothing done yet" state for a resumable stage, not an error.
        tolerant: When True, a malformed trailing line (the signature of a
            process killed mid-write) is dropped with a warning instead of
            raising. A malformed line anywhere else still raises, because that
            means something worse than an interruption happened.
    """
    path = Path(path)
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        lines = handle.readlines()
    for index, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except ValueError:
            is_last = index == len(lines) - 1
            if tolerant and is_last:
                log(f"warn: dropping truncated final line of {path.name} "
                    f"(interrupted write)")
                continue
            raise ValueError(f"{path}:{index + 1} is not valid JSON")
    return records


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> None:
    """Write records to a JSONL file atomically (temp file + rename)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    tmp.replace(path)


class JsonlAppender:
    """Thread-safe append-only JSONL writer, one open-append-close per record.

    Re-opening the file for every record looks wasteful and is not: the work
    between records is a model forward pass or a sandboxed subprocess, each
    three to four orders of magnitude more expensive than an `open()`. What it
    buys is worth far more than the microseconds:

    * a crash at any instant leaves a complete file, never a half-flushed one;
    * no file descriptor is held across a `fork`/`exec`, which matters on
      network and FUSE-backed filesystems (Google Drive mounts in Colab,
      overlay mounts in containers) where a long-lived descriptor can be
      invalidated out from under the process -- observed as
      `OSError: [Errno 9] Bad file descriptor` mid-run, hours in;
    * another process can safely read or tail the file while it grows.

    `os.O_APPEND` makes each write atomic for lines under PIPE_BUF, so the
    lock only serialises this process's threads and the file stays valid even
    if two processes append at once.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.path.touch(exist_ok=True)

    def write(self, record: dict[str, Any]) -> None:
        line = json.dumps(record, ensure_ascii=False) + "\n"
        with self._lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line)

    def close(self) -> None:
        """Nothing to close; kept so callers can use this as a context manager."""

    def __enter__(self) -> "JsonlAppender":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def done_ids(path: str | Path, id_field: str) -> set[str]:
    """Ids already present in a checkpoint file -- the resume primitive."""
    return {str(record[id_field]) for record in read_jsonl(path) if id_field in record}


def write_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


# --------------------------------------------------------------------------
# Logging / progress
# --------------------------------------------------------------------------

_START = time.time()


def log(message: str) -> None:
    """Timestamped line to stderr, so piping stdout to a file stays clean."""
    elapsed = time.time() - _START
    print(f"[{elapsed:7.1f}s] {message}", file=sys.stderr, flush=True)


def progress(iterable: Iterable[Any], total: int | None = None, every: int = 25,
             label: str = "") -> Iterator[Any]:
    """Minimal progress logger. tqdm would be nicer; one less dependency wins."""
    if total is None:
        try:
            total = len(iterable)  # type: ignore[arg-type]
        except TypeError:
            total = None
    started = time.time()
    for index, item in enumerate(iterable, start=1):
        yield item
        if index % every == 0 or index == total:
            elapsed = time.time() - started
            rate = index / elapsed if elapsed else 0.0
            eta = (total - index) / rate if (total and rate) else float("nan")
            log(f"{label} {index}/{total or '?'}  {rate:.2f}/s  eta {eta:.0f}s")


def banner(title: str) -> None:
    print("\n" + "=" * 78, flush=True)
    print(title, flush=True)
    print("=" * 78, flush=True)
