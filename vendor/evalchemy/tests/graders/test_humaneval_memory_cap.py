# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Regression coverage for the memory cap in the HumanEval grader.

Issue #147: ``eval.graders.humaneval`` has the same latent gap as the vendored LiveCodeBench
stacks -- ``reliability_guard`` accepts ``maximum_memory_bytes`` but ``check_correctness``
never passes it, so the cap is never armed. Before the fix the cap default was ``None`` and
a runaway candidate ran unbounded; these tests assert the cap is wired by default and that a
candidate exhausting it is resolved deterministically (no hang).

The *enforcement* of the cap (failing the candidate) is asserted on Linux, where ``RLIMIT_AS``
is honored. macOS refuses to lower it / overcommits, so there the cap is best-effort (the
grader must not crash) and only the "wired" and "returns without hanging" guarantees hold.
"""

import inspect
import pathlib
import platform
import sys
import time

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from eval.graders import humaneval  # noqa: E402

LINUX = platform.system() == "Linux"


def test_check_correctness_wires_the_memory_cap_by_default():
    """The pre-#147 ``check_correctness`` had no ``maximum_memory_bytes`` parameter at all;
    a runaway ran unbounded. This is the direct regression: the parameter must exist and
    have a non-None default, so a caller who never passes one still gets a cap."""
    defaults = inspect.signature(humaneval.check_correctness).parameters["maximum_memory_bytes"].default
    assert defaults is not None, "check_correctness must wire a memory cap by default"


def test_a_candidate_that_exhausts_memory_returns_without_hanging():
    """A 1 GiB allocation is resolved quickly (MemoryError, kill, or pass) -- never a hang.
    On macOS the cap is not enforced (the guard degrades there), so only the "returns
    without hanging" guarantee is asserted; the Linux-enforced failure is below."""
    start = time.perf_counter()
    humaneval.check_correctness("x = bytearray(1 * 1024 * 1024 * 1024)\n", timeout=10.0)
    assert time.perf_counter() - start < 15.0, "deterministic resolution, not a hang"


@pytest.mark.skipif(not LINUX, reason="macOS refuses to lower RLIMIT_AS / overcommits, so the cap is best-effort there")
def test_cap_is_enforced_on_linux():
    """On Linux the RLIMIT_AS actually trips: a 1 GiB alloc under the default 1 GiB cap is a failure."""
    outcome = humaneval.check_correctness("x = bytearray(1 * 1024 * 1024 * 1024)\n", timeout=10.0)
    assert outcome != humaneval.PASSED

