"""Run a candidate against assert-style tests in a sandboxed subprocess.

Extends the vendored harness (PROVENANCE.md), which only knew how to call a
function and JSON-compare its return value. Adds the failure taxonomy, a
disposable cwd per run, out-of-band results, and a network block.

NOT a security boundary — same uid, readable fs. Fine for a 1.5B model on a
throwaway VM, not for hostile code.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .sandbox import SandboxLimits  # reused verbatim from the upstream project

_RUNNER = Path(__file__).resolve().with_name("script_runner.py")
_STARTUP_GRACE_S = 2.0  # interpreter start-up + result write, not candidate time


@dataclass
class ExecResult:
    """Outcome of running one candidate against one problem's tests."""

    outcome: str                      # one of config.OUTCOMES
    passed: bool
    error_type: str | None = None
    error_message: str | None = None
    failed_test_index: int | None = None
    n_tests_passed: int = 0
    n_tests_total: int = 0
    wall_time_ms: float = 0.0
    exit_code: int | None = None
    stdout: str = ""
    stderr: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ScriptSandbox:
    """Runs untrusted candidate code against assert-style tests, out of process."""

    def __init__(self, limits: SandboxLimits | None = None,
                 python_executable: str | None = None) -> None:
        self.limits = limits or SandboxLimits()
        self._python = python_executable or sys.executable
        if not _RUNNER.is_file():  # pragma: no cover - packaging error
            raise RuntimeError(f"script runner missing at {_RUNNER}")

    def run(
        self,
        solution: str,
        tests: list[str],
        test_defs: str = "",
        *,
        wall_time_s: float | None = None,
    ) -> ExecResult:
        """Execute *solution*, then each statement in *tests*, in a child process.

        Returns an :class:`ExecResult`; candidate failures are results, not
        exceptions. Only harness misconfiguration raises.
        """
        wall = float(wall_time_s or self.limits.wall_time_s)
        job = json.dumps({
            "solution": solution,
            "test_defs": test_defs,
            "tests": tests,
            "memory_limit_mb": self.limits.memory_mb,
            "cpu_time_limit_s": self.limits.resolved_cpu_s(wall),
        })

        # Two separate temp dirs: the candidate gets `work_dir` as its cwd and
        # may do anything it likes in there; the result file lives in
        # `out_dir`, whose path the candidate is never told.
        work_dir = tempfile.mkdtemp(prefix="crm_work_")
        out_dir = tempfile.mkdtemp(prefix="crm_out_")
        result_path = Path(out_dir) / "result.json"

        popen_kwargs: dict[str, Any] = {
            "stdin": subprocess.PIPE,
            "stdout": subprocess.PIPE,
            "stderr": subprocess.PIPE,
            "text": True,
            "encoding": "utf-8",
            "errors": "replace",
            "cwd": work_dir,
            # `-I` implies -E -s -P: no PYTHONPATH, no user site-packages, and
            # the script's own directory (not cwd) heads sys.path. A candidate
            # cannot shadow a stdlib module by dropping a file in its cwd.
            "env": {"PATH": os.environ.get("PATH", ""), "LC_ALL": "C.UTF-8",
                    "PYTHONIOENCODING": "utf-8"},
        }
        if os.name == "posix":
            popen_kwargs["start_new_session"] = True  # own process group

        started = time.perf_counter()
        timed_out = False
        proc = subprocess.Popen(  # noqa: S603 - fixed argv, no shell
            [self._python, "-I", str(_RUNNER), str(result_path)], **popen_kwargs
        )
        try:
            out, err = proc.communicate(job, timeout=wall + _STARTUP_GRACE_S)
        except subprocess.TimeoutExpired:
            self._kill_group(proc)
            timed_out = True
            try:
                out, err = proc.communicate(timeout=5)
            except subprocess.TimeoutExpired:  # pragma: no cover
                out, err = "", ""
        wall_ms = (time.perf_counter() - started) * 1000.0

        try:
            return self._interpret(result_path, proc.returncode, timed_out,
                                   wall_ms, out or "", err or "")
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)
            shutil.rmtree(out_dir, ignore_errors=True)

    # ------------------------------------------------------------- internals

    @staticmethod
    def _kill_group(proc: subprocess.Popen[str]) -> None:
        """SIGKILL the child *and everything it forked*."""
        import contextlib
        import signal

        try:
            if os.name == "posix":
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            else:  # pragma: no cover - Windows
                proc.kill()
        except (ProcessLookupError, PermissionError, OSError):
            with contextlib.suppress(Exception):
                proc.kill()

    def _interpret(self, result_path: Path, returncode: int | None, timed_out: bool,
                   wall_ms: float, out: str, err: str) -> ExecResult:
        if timed_out:
            return ExecResult(
                outcome="timeout", passed=False, error_type="Timeout",
                error_message=f"exceeded {self.limits.wall_time_s:.1f}s wall clock",
                wall_time_ms=wall_ms, exit_code=returncode, stderr=err[-2000:],
            )

        if result_path.exists():
            try:
                payload = json.loads(result_path.read_text(encoding="utf-8"))
            except ValueError:
                payload = None
            if payload is not None:
                error = payload.get("error") or {}
                return ExecResult(
                    outcome=str(payload.get("outcome", "harness_error")),
                    passed=payload.get("outcome") == "pass",
                    error_type=error.get("type"),
                    error_message=(error.get("message") or None),
                    failed_test_index=payload.get("failed_test_index"),
                    n_tests_passed=int(payload.get("n_tests_passed", 0)),
                    n_tests_total=int(payload.get("n_tests_total", 0)),
                    wall_time_ms=wall_ms, exit_code=returncode,
                    stdout=payload.get("stdout", ""), stderr=payload.get("stderr", ""),
                )

        # No result file: the child died before it could write one. The most
        # common cause by far is the kernel OOM-killer or a hard RLIMIT kill,
        # both of which are legitimately the candidate's failure.
        return self._from_signal(returncode, wall_ms, out, err)

    @staticmethod
    def _from_signal(returncode: int | None, wall_ms: float, out: str,
                     err: str) -> ExecResult:
        import signal

        outcome, message = "harness_error", "child produced no result file"
        if returncode is not None and returncode < 0 and os.name == "posix":
            sig = -returncode
            if sig == signal.SIGXCPU:
                outcome, message = "timeout", "CPU-time limit exceeded (SIGXCPU)"
            elif sig == signal.SIGKILL:
                outcome, message = "memory_exceeded", "killed by SIGKILL (likely OOM)"
            elif sig == getattr(signal, "SIGSEGV", -1):
                outcome, message = "exception", "segmentation fault in child"
            else:
                outcome, message = "exception", f"killed by signal {sig}"
        elif returncode:
            outcome, message = "exception", f"child exited with code {returncode}"
        return ExecResult(
            outcome=outcome, passed=False, error_type="ChildProcessFailure",
            error_message=message, wall_time_ms=wall_ms, exit_code=returncode,
            stdout=out[-2000:], stderr=err[-2000:],
        )
