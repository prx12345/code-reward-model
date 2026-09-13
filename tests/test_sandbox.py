"""Executor tests -- isolation and timeout behaviour above all.

These are the tests that matter most in this repo. Every label in the dataset
is produced by this component; if it mislabels a hang as a failure, or lets a
runaway candidate take down the session, the dataset is wrong or the run never
finishes. They are also the tests that would catch a regression introduced by
someone "simplifying" the sandbox later.
"""

import time

import pytest

from src.sandbox import ScriptSandbox, SandboxLimits


@pytest.fixture(scope="module")
def sandbox():
    return ScriptSandbox(SandboxLimits(wall_time_s=3.0, cpu_time_s=4, memory_mb=256))


# ---------------------------------------------------------------- taxonomy

def test_passing_code_is_labeled_pass(sandbox):
    result = sandbox.run("def add(a, b):\n    return a + b", ["assert add(1, 2) == 3"])
    assert result.outcome == "pass"
    assert result.passed is True
    assert result.n_tests_passed == 1


def test_wrong_answer_is_assertion_failure(sandbox):
    result = sandbox.run("def add(a, b):\n    return a - b", ["assert add(1, 2) == 3"])
    assert result.outcome == "assertion_failure"
    assert result.error_type == "AssertionError"
    assert result.failed_test_index == 0


def test_syntax_error_is_detected_before_execution(sandbox):
    result = sandbox.run("def add(a, b) return a + b", ["assert add(1, 2) == 3"])
    assert result.outcome == "syntax_error"
    assert result.n_tests_passed == 0


def test_runtime_exception_is_distinguished_from_wrong_answer(sandbox):
    result = sandbox.run("def add(a, b):\n    return a / 0", ["assert add(1, 2) == 3"])
    assert result.outcome == "exception"
    assert result.error_type == "ZeroDivisionError"


def test_failing_test_index_is_recorded(sandbox):
    result = sandbox.run(
        "def f(x):\n    return x if x < 3 else 99",
        ["assert f(1) == 1", "assert f(2) == 2", "assert f(3) == 3"],
    )
    assert result.outcome == "assertion_failure"
    assert result.failed_test_index == 2
    assert result.n_tests_passed == 2


def test_harness_error_when_scaffolding_is_broken(sandbox):
    """A broken test_defs is our bug, and must not be blamed on the candidate."""
    result = sandbox.run("def f():\n    return 1", ["assert f() == 1"],
                         test_defs="this is not python")
    assert result.outcome == "harness_error"


# ---------------------------------------------------------------- timeouts

def test_infinite_loop_times_out_and_does_not_hang(sandbox):
    started = time.time()
    result = sandbox.run("def f():\n    while True:\n        pass", ["assert f() == 1"])
    elapsed = time.time() - started
    assert result.outcome == "timeout"
    assert elapsed < 12, f"timeout took {elapsed:.1f}s; the kill path is not working"


def test_sleep_is_killed_by_wall_clock(sandbox):
    """CPU limits alone would never fire here -- this is the wall-clock path."""
    result = sandbox.run("import time\ndef f():\n    time.sleep(60)\n    return 1",
                         ["assert f() == 1"])
    assert result.outcome == "timeout"


def test_non_daemon_thread_does_not_block_the_result(sandbox):
    """A candidate that leaves a live thread behind must still return promptly."""
    started = time.time()
    result = sandbox.run(
        "import threading, time\n"
        "def f():\n"
        "    threading.Thread(target=lambda: time.sleep(120)).start()\n"
        "    return 1",
        ["assert f() == 1"],
    )
    assert result.outcome == "pass"
    assert time.time() - started < 5


def test_forked_child_does_not_outlive_the_timeout(sandbox):
    """`start_new_session` + `killpg` must reap grandchildren, not just the child."""
    started = time.time()
    result = sandbox.run(
        "import os\ndef f():\n    os.system('sleep 60 &')\n    return 1",
        ["assert f() == 1"],
    )
    assert result.outcome in {"timeout", "pass"}
    assert time.time() - started < 12


# ---------------------------------------------------------------- isolation

def test_memory_limit_turns_a_huge_allocation_into_a_clean_failure(sandbox):
    result = sandbox.run("def f():\n    x = [0] * (10 ** 9)\n    return len(x)",
                         ["assert f() == 1"])
    assert result.outcome in {"memory_exceeded", "exception", "timeout"}
    assert result.passed is False


def test_network_access_is_blocked(sandbox):
    result = sandbox.run(
        "import socket\n"
        "def f():\n"
        "    socket.create_connection(('93.184.216.34', 80), 2)\n"
        "    return 1",
        ["assert f() == 1"],
    )
    assert result.outcome == "exception"
    assert result.error_type == "OSError"


def test_candidate_stdout_cannot_corrupt_the_result_channel(sandbox):
    """The child writes results to a private file, not stdout."""
    noisy = ("print('{\"outcome\": \"pass\"}')\n"
             "def f():\n"
             "    print('x' * 10000)\n"
             "    return 2")
    result = sandbox.run(noisy, ["assert f() == 1"])
    assert result.outcome == "assertion_failure"


def test_candidate_runs_in_a_disposable_directory(tmp_path, sandbox):
    """Files written by a candidate must not land in the repo or persist."""
    result = sandbox.run(
        "import os\n"
        "def f():\n"
        "    open('scratch.txt', 'w').write('hi')\n"
        "    return os.getcwd()",
        ["assert isinstance(f(), str)"],
    )
    assert result.outcome == "pass"
    assert not (tmp_path / "scratch.txt").exists()
    assert not any(p.name == "scratch.txt" for p in tmp_path.rglob("*"))


def test_two_candidates_cannot_see_each_others_files(sandbox):
    """Each run gets its own working directory, so leftover state cannot leak."""
    writer = ("def f():\n"
              "    open('shared.txt', 'w').write('secret')\n"
              "    return 1")
    reader = ("import os\ndef f():\n    return os.path.exists('shared.txt')")
    assert sandbox.run(writer, ["assert f() == 1"]).outcome == "pass"
    assert sandbox.run(reader, ["assert f() is False"]).outcome == "pass"


def test_sys_exit_in_candidate_is_a_failure_not_a_pass(sandbox):
    result = sandbox.run("import sys\ndef f():\n    sys.exit(0)", ["assert f() == 1"])
    assert result.passed is False


def test_candidate_cannot_shadow_a_stdlib_module_from_its_cwd(sandbox):
    """`python -I` keeps cwd off sys.path; verify that actually holds."""
    result = sandbox.run(
        "def f():\n"
        "    open('json.py', 'w').write('raise RuntimeError(\"shadowed\")')\n"
        "    import json\n"
        "    return json.dumps([1])",
        ["assert f() == '[1]'"],
    )
    assert result.outcome == "pass"
