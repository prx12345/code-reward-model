"""Sandboxed execution of untrusted, model-generated code.

`sandbox.py` / `sandbox_runner.py` are vendored unchanged from the author's
earlier project (see PROVENANCE.md). `script_sandbox.py` / `script_runner.py`
extend them for assert-based benchmark tests.
"""

from .sandbox import Sandbox, SandboxLimits, SandboxResult  # noqa: F401
from .script_sandbox import ExecResult, ScriptSandbox  # noqa: F401

__all__ = ["Sandbox", "SandboxLimits", "SandboxResult", "ExecResult", "ScriptSandbox"]
