"""Child process that executes one candidate solution against its unit tests.

Launched by :class:`script_sandbox.ScriptSandbox` as::

    python -I script_runner.py <result_path>

with a single JSON job on stdin. It never writes its answer to stdout (the
candidate owns stdout and may close, spam or binary-corrupt it) -- the result
goes to ``result_path`` and the parent reads the file.

Why a second runner instead of reusing ``sandbox_runner.py``: the upstream
runner's contract is "call function(*args) and JSON-compare the return value".
MBPP and HumanEval ship their ground truth as *Python assert statements*, not
as JSON argument/expected pairs, so there is no return value to compare. This
runner executes the tests as code and classifies *how* they failed, which is
what the labeling stage needs.

Resource limits are imported from the upstream runner rather than duplicated.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import traceback
from typing import Any

# `python -I` implies `-P`, which deliberately does *not* prepend the script's
# directory to sys.path. That is the behaviour we want for candidate code (it
# cannot shadow stdlib modules), so instead of dropping isolation we add the
# one directory we trust -- this file's own -- by absolute path.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sandbox_runner import _apply_resource_limits, _truncate  # noqa: E402

_STREAM_CAP = 8_000
_MSG_CAP = 2_000


def _disable_network() -> None:
    """Best-effort, Python-level network block.

    This stops the realistic failure mode -- generated code that calls
    ``requests.get`` or ``urllib`` because the model hallucinated an API-backed
    solution -- and nothing more. A candidate that really wanted out could
    reach ``ctypes`` and call ``connect(2)`` directly. See the README threat
    model: for genuinely hostile code you need a network namespace or a
    container, not a monkeypatch.
    """
    for var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY",
                "ALL_PROXY", "all_proxy"):
        os.environ.pop(var, None)

    try:
        import socket
    except ImportError:  # pragma: no cover
        return

    def _blocked(*_args: Any, **_kwargs: Any) -> Any:
        raise OSError("network access is disabled inside the sandbox")

    socket.socket = _blocked  # type: ignore[assignment]
    socket.create_connection = _blocked  # type: ignore[assignment]
    socket.create_server = _blocked  # type: ignore[assignment]
    with contextlib.suppress(AttributeError):
        socket.socketpair = _blocked  # type: ignore[assignment]
    socket.getaddrinfo = _blocked  # type: ignore[assignment]


def _classify(exc: BaseException) -> str:
    """Map a raised exception to one of the dataset's outcome labels."""
    if isinstance(exc, AssertionError):
        return "assertion_failure"
    if isinstance(exc, MemoryError):
        return "memory_exceeded"
    if isinstance(exc, (SyntaxError, IndentationError)):
        # Reached only if the candidate compiles something at runtime (eval/exec).
        return "syntax_error"
    if isinstance(exc, RecursionError):
        return "exception"
    return "exception"


def _error_dict(exc: BaseException) -> dict[str, str]:
    return {
        "type": type(exc).__name__,
        "message": str(exc)[:_MSG_CAP],
        "traceback": traceback.format_exc(limit=12)[-4_000:],
    }


def main() -> int:
    result: dict[str, Any] = {
        "outcome": "harness_error",
        "error": None,
        "failed_test_index": None,
        "n_tests_passed": 0,
        "n_tests_total": 0,
        "stdout": "",
        "stderr": "",
    }
    result_path = sys.argv[1] if len(sys.argv) > 1 else None

    def emit(code: int) -> int:
        if result_path:
            with contextlib.suppress(OSError):
                with open(result_path, "w", encoding="utf-8") as handle:
                    json.dump(result, handle)
                    handle.flush()
                    os.fsync(handle.fileno())
        return code

    try:
        job: dict[str, Any] = json.loads(sys.stdin.read())
    except (ValueError, OSError) as exc:
        result["error"] = {"type": type(exc).__name__, "message": str(exc)[:_MSG_CAP]}
        return emit(3)

    _apply_resource_limits(job.get("memory_limit_mb"), job.get("cpu_time_limit_s"))
    _disable_network()

    solution: str = job.get("solution", "")
    test_defs: str = job.get("test_defs", "") or ""
    tests: list[str] = list(job.get("tests") or [])
    result["n_tests_total"] = len(tests)

    out_capture, err_capture = io.StringIO(), io.StringIO()
    namespace: dict[str, Any] = {"__name__": "__candidate__"}

    try:
        with contextlib.redirect_stdout(out_capture), contextlib.redirect_stderr(err_capture):
            # 1. Compile the candidate on its own, so a syntax error is
            #    attributed to the candidate and not to the test harness.
            try:
                compiled = compile(solution, "<candidate>", "exec")
            except (SyntaxError, ValueError) as exc:
                result.update(outcome="syntax_error", error=_error_dict(exc))
                raise _Done

            # 2. Define the candidate. Import-time / top-level failures are the
            #    candidate's fault (bad import, undefined name at module level).
            try:
                exec(compiled, namespace)  # noqa: S102 - this is the point
            except BaseException as exc:  # noqa: BLE001
                result.update(outcome=_classify(exc), error=_error_dict(exc))
                raise _Done

            # 3. Test scaffolding (MBPP's test_setup_code, HumanEval's check()).
            #    Failures here are *ours*, not the candidate's, so they get
            #    their own label and are dropped from the dataset upstream.
            if test_defs.strip():
                try:
                    exec(compile(test_defs, "<test_defs>", "exec"), namespace)  # noqa: S102
                except BaseException as exc:  # noqa: BLE001
                    result.update(outcome="harness_error", error=_error_dict(exc))
                    raise _Done

            # 4. Run the assertions one at a time so we know which one broke.
            for index, statement in enumerate(tests):
                try:
                    exec(compile(statement, f"<test_{index}>", "exec"), namespace)  # noqa: S102
                except BaseException as exc:  # noqa: BLE001
                    result.update(
                        outcome=_classify(exc),
                        error=_error_dict(exc),
                        failed_test_index=index,
                    )
                    raise _Done
                result["n_tests_passed"] = index + 1

            result["outcome"] = "pass"
    except _Done:
        pass
    except BaseException as exc:  # noqa: BLE001 - never let the child die silently
        result.update(outcome=_classify(exc), error=_error_dict(exc))

    result["stdout"] = _truncate(out_capture.getvalue(), _STREAM_CAP)
    result["stderr"] = _truncate(err_capture.getvalue(), _STREAM_CAP)
    emit(0)
    # Hard-exit: candidate code may have started non-daemon threads or
    # registered atexit hooks that would otherwise keep this process alive
    # past the parent's timeout and turn a clean result into a "timeout".
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


class _Done(BaseException):
    """Internal control-flow signal to break out of the nested try blocks."""


if __name__ == "__main__":
    raise SystemExit(main())
