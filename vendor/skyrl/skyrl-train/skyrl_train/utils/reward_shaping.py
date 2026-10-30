"""
Reward shaping utilities for parsing test outputs and computing shaped rewards.

This module provides a flexible framework for:
1. Parsing test output from various frameworks (pytest, unittest, etc.)
2. Computing shaped rewards based on test pass/fail ratios

Usage:
    from skyrl_train.utils.reward_shaping import (
        get_output_parser,
        get_reward_shaper,
        shape_reward_from_output,
    )

    # Parse and shape in one call
    shaped_reward = shape_reward_from_output(
        stdout=verifier_stdout,
        original_reward=0.0,
        parser_name="pytest",
        shaper_name="pass_ratio",
    )

    # Or use components separately
    parser = get_output_parser("pytest")
    shaper = get_reward_shaper("pass_ratio")

    parsed = parser.parse(stdout)
    if parsed is not None:
        shaped_reward = shaper.shape(parsed, original_reward)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from enum import StrEnum
from hashlib import sha256
import json
import math
import re
from typing import Any, Dict, List, Optional, Tuple, Type

from loguru import logger
from skyrl_train.trajectory_runners.types import TrajectoryID, VerifierTestCollection, VerifierTestRecord


class RewardOutputParseError(ValueError):
    """Verifier output cannot supply a required shaping score."""


# =============================================================================
# Data Classes
# =============================================================================


class VerifierTestOutcome(StrEnum):
    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"
    XFAILED = "xfailed"
    XPASSED = "xpassed"
    SKIPPED = "skipped"


@dataclass(frozen=True)
class ParsedVerifierTest:
    test_id: str
    outcome: VerifierTestOutcome
    output: str


@dataclass
class ParsedTestResult:
    """
    Structured representation of test results from any test framework.

    Attributes:
        passed: Number of tests that passed
        failed: Number of tests that failed (assertions failed)
        errors: Number of tests that errored (couldn't run, setup issues)
        xfailed: Expected failures (test failed as expected, counts as success)
        xpassed: Unexpected passes (test passed when expected to fail)
        skipped: Tests that were skipped
        warnings: Number of warnings (informational)
        total: Total number of tests (computed if not set)
        duration_sec: Test duration in seconds (if available)
        raw_output: Original output string for debugging
        metadata: Additional framework-specific data
        tests: Individually identified test outcomes and their literal output lines
        tests_complete: Whether tests account for every aggregate outcome with unique identities
    """

    passed: int = 0
    failed: int = 0
    errors: int = 0
    xfailed: int = 0
    xpassed: int = 0
    skipped: int = 0
    warnings: int = 0
    total: int = 0
    duration_sec: Optional[float] = None
    raw_output: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    tests: tuple[ParsedVerifierTest, ...] = ()
    tests_complete: bool = False

    def __post_init__(self):
        # Compute total if not explicitly set
        if self.total == 0:
            self.total = self.passed + self.failed + self.errors + self.xfailed + self.xpassed + self.skipped

    @property
    def effective_passed(self) -> int:
        """Tests that behaved as expected (passed + xfailed)."""
        return self.passed + self.xfailed

    @property
    def effective_failed(self) -> int:
        """Tests that did not behave as expected (failed + errors + xpassed)."""
        return self.failed + self.errors + self.xpassed

    @property
    def runnable_total(self) -> int:
        """Total tests excluding skipped (tests that actually ran)."""
        return self.total - self.skipped

    @property
    def pass_ratio(self) -> float:
        """Simple pass ratio: passed / total (0.0 if no tests)."""
        if self.total == 0:
            return 0.0
        return self.passed / self.total

    @property
    def effective_pass_ratio(self) -> float:
        """Effective pass ratio: effective_passed / runnable_total."""
        if self.runnable_total == 0:
            return 0.0
        return self.effective_passed / self.runnable_total


_OUTCOME_BY_LABEL = {
    "PASS": VerifierTestOutcome.PASSED,
    "PASSED": VerifierTestOutcome.PASSED,
    "FAIL": VerifierTestOutcome.FAILED,
    "FAILED": VerifierTestOutcome.FAILED,
    "ERROR": VerifierTestOutcome.ERROR,
    "XFAIL": VerifierTestOutcome.XFAILED,
    "XFAILED": VerifierTestOutcome.XFAILED,
    "XPASS": VerifierTestOutcome.XPASSED,
    "XPASSED": VerifierTestOutcome.XPASSED,
    "SKIP": VerifierTestOutcome.SKIPPED,
    "SKIPPED": VerifierTestOutcome.SKIPPED,
}

_PYTEST_TEST_LINE = re.compile(r"^(?P<outcome>PASSED|FAILED|ERROR|XFAIL|XPASS|SKIPPED)\s+(?P<test_id>\S+)(?:\s+.*)?$")
_PYTEST_STANDARD_TEST_LINE = re.compile(
    r"^(?P<test_id>\S+)\s+(?P<outcome>PASSED|FAILED|ERROR|XFAIL|XPASS|SKIPPED)(?:\s+.*)?$"
)
_BRACKET_TEST_LINE = re.compile(
    r"^\[(?P<test_id>[^\]]+)\]\s+(?P<outcome>PASS|PASSED|FAIL|FAILED|ERROR|SKIP|SKIPPED|XFAIL|XPASS)(?::.*)?$"
)
_SUFFIX_TEST_LINE = re.compile(
    r"^(?P<test_id>.+?):\s*(?P<outcome>PASS|PASSED|FAIL|FAILED|ERROR|SKIP|SKIPPED|XFAIL|XPASS)(?::.*)?$"
)
_PREFIX_TEST_LINE = re.compile(
    r"^(?P<outcome>PASS|PASSED|FAIL|FAILED|ERROR|SKIP|SKIPPED|XFAIL|XPASS)\s+(?P<test_id>\S+)(?:\s+.*)?$"
)


def _identified_tests(output: str, parser_name: str) -> tuple[ParsedVerifierTest, ...]:
    patterns = (
        (_PYTEST_TEST_LINE, _PYTEST_STANDARD_TEST_LINE)
        if parser_name == PytestOutputParser.name()
        else (_BRACKET_TEST_LINE, _SUFFIX_TEST_LINE, _PREFIX_TEST_LINE)
    )
    tests = []
    for raw_line in output.splitlines():
        line = raw_line.strip()
        for pattern in patterns:
            match = pattern.fullmatch(line)
            if match is None:
                continue
            tests.append(
                ParsedVerifierTest(
                    test_id=match.group("test_id").strip(),
                    outcome=_OUTCOME_BY_LABEL[match.group("outcome")],
                    output=raw_line,
                )
            )
            break
    return tuple(tests)


def _tests_are_complete(parsed: ParsedTestResult) -> bool:
    if len(parsed.tests) != parsed.total:
        return False
    if len({test.test_id for test in parsed.tests}) != len(parsed.tests):
        return False
    expected = {
        VerifierTestOutcome.PASSED: parsed.passed,
        VerifierTestOutcome.FAILED: parsed.failed,
        VerifierTestOutcome.ERROR: parsed.errors,
        VerifierTestOutcome.XFAILED: parsed.xfailed,
        VerifierTestOutcome.XPASSED: parsed.xpassed,
        VerifierTestOutcome.SKIPPED: parsed.skipped,
    }
    return all(sum(test.outcome is outcome for test in parsed.tests) == count for outcome, count in expected.items())


def _with_identified_tests(parsed: ParsedTestResult, tests: tuple[ParsedVerifierTest, ...]) -> ParsedTestResult:
    parsed_with_tests = replace(parsed, tests=tests)
    return replace(parsed_with_tests, tests_complete=_tests_are_complete(parsed_with_tests))


def verifier_test_collection(
    parsed: ParsedTestResult | None,
    *,
    parser_name: str | None,
    instance_id: str,
    repetition_id: int,
) -> VerifierTestCollection:
    """Materialize parsed tests with deterministic trial-scoped record IDs."""
    records: list[VerifierTestRecord] = []
    for occurrence, test in enumerate(() if parsed is None else parsed.tests):
        identity = json.dumps(
            [instance_id, repetition_id, test.test_id, occurrence], ensure_ascii=False, separators=(",", ":")
        )
        records.append(
            {
                "record_id": sha256(identity.encode()).hexdigest(),
                "trial_id": TrajectoryID(instance_id=instance_id, repetition_id=repetition_id),
                "test_id": test.test_id,
                "outcome": test.outcome.value,
                "output": test.output,
            }
        )
    return {
        "parser": parser_name,
        "complete": parsed is not None and parsed.tests_complete,
        "tests": records,
    }


# =============================================================================
# Output Parsers
# =============================================================================


class OutputParser(ABC):
    """
    Abstract base class for parsing test output strings.

    Subclasses implement parsing logic for specific test frameworks.
    """

    @classmethod
    @abstractmethod
    def name(cls) -> str:
        """Return the parser name for registry lookup."""
        pass

    @abstractmethod
    def parse(self, output: str) -> Optional[ParsedTestResult]:
        """
        Parse test output and extract structured results.

        Args:
            output: Raw test output string (stdout/stderr)

        Returns:
            ParsedTestResult if parsing succeeded, None if output format
            not recognized or parsing failed.
        """
        pass

    def can_parse(self, output: str) -> bool:
        """
        Quick check if this parser can handle the output.

        Override for more efficient detection before full parsing.
        """
        return self.parse(output) is not None


class PytestOutputParser(OutputParser):
    """
    Parser for pytest output.

    Recognizes pytest summary lines like:
        ============== 1 failed, 62 passed, 2 xfailed, 66 errors in 2.39s ==============
        ============================= 5 passed in 0.12s ==============================

    Also counts individual test result lines:
        PASSED test_file.py::test_name
        FAILED test_file.py::test_name - AssertionError
        ERROR test_file.py::test_name
        XFAIL test_file.py::test_name - reason
        XPASS test_file.py::test_name
        SKIPPED test_file.py::test_name - reason

    Collection errors (where pytest can't load tests) are detected and
    return None to signal unparseable output, falling back to original reward.
    """

    # Regex for the summary line at the end of pytest output
    # Matches: "=== 1 failed, 62 passed, 2 xfailed in 2.39s ==="
    SUMMARY_PATTERN = re.compile(
        r"=+\s*" r"(?P<results>(?:\d+\s+\w+(?:,\s*)?)+)" r"\s+in\s+" r"(?P<duration>[\d.]+)s?" r"\s*=+",
        re.IGNORECASE,
    )

    # Pattern to extract individual counts from summary
    COUNT_PATTERN = re.compile(r"(\d+)\s+(\w+)", re.IGNORECASE)

    # Patterns that indicate pytest couldn't collect/load tests
    # These are infrastructure failures, not agent failures
    COLLECTION_ERROR_PATTERNS = [
        re.compile(r"error during collection", re.IGNORECASE),
        re.compile(r"Interrupted:.*error", re.IGNORECASE),
        re.compile(r"no tests ran", re.IGNORECASE),
        re.compile(r"collection error", re.IGNORECASE),
        re.compile(r"import error", re.IGNORECASE),
    ]

    @classmethod
    def name(cls) -> str:
        return "pytest"

    def _is_collection_error(self, output: str) -> bool:
        """Check if output indicates a pytest collection/import error."""
        for pattern in self.COLLECTION_ERROR_PATTERNS:
            if pattern.search(output):
                return True
        return False

    def parse(self, output: str) -> Optional[ParsedTestResult]:
        """Parse pytest output to extract test counts.

        Returns None for collection errors (where pytest couldn't load tests),
        which causes fallback to original reward rather than treating as 0/N failures.
        """
        if not output:
            return None

        # Check for collection errors FIRST - these indicate pytest couldn't
        # even load the tests, which is typically an infrastructure issue
        # (bad test file, missing dependencies) not an agent failure.
        # Return None to fall back to original verifier reward.
        if self._is_collection_error(output):
            logger.debug(
                "Detected pytest collection error - skipping reward shaping. Falling back to original verifier reward."
            )
            return None

        # Try to find the summary line first (most reliable)
        summary_match = self.SUMMARY_PATTERN.search(output)

        if summary_match:
            return self._parse_from_summary(output, summary_match)

        # Fall back to counting individual result lines
        return self._parse_from_lines(output)

    def _parse_from_summary(self, output: str, summary_match: re.Match) -> Optional[ParsedTestResult]:
        """Parse from the pytest summary line."""
        results_str = summary_match.group("results")
        duration_str = summary_match.group("duration")

        counts = {
            "passed": 0,
            "failed": 0,
            "error": 0,
            "errors": 0,
            "xfailed": 0,
            "xpassed": 0,
            "skipped": 0,
            "warnings": 0,
            "warning": 0,
            "deselected": 0,
        }

        for count_match in self.COUNT_PATTERN.finditer(results_str):
            count = int(count_match.group(1))
            status = count_match.group(2).lower()
            if status in counts:
                counts[status] = count

        # Combine error/errors (pytest uses both)
        errors = counts["error"] + counts["errors"]
        warnings = counts["warning"] + counts["warnings"]

        try:
            duration = float(duration_str)
        except (ValueError, TypeError):
            duration = None

        return ParsedTestResult(
            passed=counts["passed"],
            failed=counts["failed"],
            errors=errors,
            xfailed=counts["xfailed"],
            xpassed=counts["xpassed"],
            skipped=counts["skipped"],
            warnings=warnings,
            duration_sec=duration,
            raw_output=output,
            metadata={"parse_method": "summary", "deselected": counts["deselected"]},
        )

    def _parse_from_lines(self, output: str) -> Optional[ParsedTestResult]:
        """Parse by counting individual test result lines."""
        tests = _identified_tests(output, self.name())
        if not tests:
            return None

        return ParsedTestResult(
            passed=sum(test.outcome is VerifierTestOutcome.PASSED for test in tests),
            failed=sum(test.outcome is VerifierTestOutcome.FAILED for test in tests),
            errors=sum(test.outcome is VerifierTestOutcome.ERROR for test in tests),
            xfailed=sum(test.outcome is VerifierTestOutcome.XFAILED for test in tests),
            xpassed=sum(test.outcome is VerifierTestOutcome.XPASSED for test in tests),
            skipped=sum(test.outcome is VerifierTestOutcome.SKIPPED for test in tests),
            raw_output=output,
            metadata={"parse_method": "line_count"},
        )

    def can_parse(self, output: str) -> bool:
        """Quick check for pytest indicators."""
        if not output:
            return False
        # Look for pytest-specific markers
        return (
            self.SUMMARY_PATTERN.search(output) is not None
            or "PASSED " in output
            or "FAILED " in output
            or "pytest" in output.lower()
        )


class UnittestOutputParser(OutputParser):
    """
    Parser for Python unittest output.

    Recognizes unittest summary lines like:
        Ran 5 tests in 0.003s
        OK
        FAILED (failures=2, errors=1)
        OK (skipped=3)
    """

    # "Ran X tests in Y.YYs"
    RAN_PATTERN = re.compile(r"Ran\s+(\d+)\s+tests?\s+in\s+([\d.]+)s", re.IGNORECASE)

    # "FAILED (failures=2, errors=1)"
    FAILED_PATTERN = re.compile(
        r"FAILED\s*\(([^)]+)\)",
        re.IGNORECASE,
    )

    # "OK" or "OK (skipped=3)"
    OK_PATTERN = re.compile(r"^OK(?:\s*\(([^)]+)\))?", re.MULTILINE | re.IGNORECASE)

    # Extract key=value pairs
    KV_PATTERN = re.compile(r"(\w+)=(\d+)")

    @classmethod
    def name(cls) -> str:
        return "unittest"

    def parse(self, output: str) -> Optional[ParsedTestResult]:
        """Parse unittest output."""
        if not output:
            return None

        # Find "Ran X tests"
        ran_match = self.RAN_PATTERN.search(output)
        if not ran_match:
            return None

        total = int(ran_match.group(1))
        try:
            duration = float(ran_match.group(2))
        except (ValueError, TypeError):
            duration = None

        counts = {
            "failures": 0,
            "errors": 0,
            "skipped": 0,
            "expected failures": 0,
            "unexpected successes": 0,
        }

        # Check for FAILED line
        failed_match = self.FAILED_PATTERN.search(output)
        if failed_match:
            details = failed_match.group(1)
            for kv_match in self.KV_PATTERN.finditer(details):
                key = kv_match.group(1).lower()
                value = int(kv_match.group(2))
                if key in counts:
                    counts[key] = value

        # Check for OK line (may have skipped, etc.)
        ok_match = self.OK_PATTERN.search(output)
        if ok_match and ok_match.group(1):
            details = ok_match.group(1)
            for kv_match in self.KV_PATTERN.finditer(details):
                key = kv_match.group(1).lower()
                value = int(kv_match.group(2))
                if key in counts:
                    counts[key] = value

        failed = counts["failures"]
        errors = counts["errors"]
        skipped = counts["skipped"]
        xfailed = counts["expected failures"]
        xpassed = counts["unexpected successes"]
        passed = total - failed - errors - skipped - xfailed - xpassed

        return ParsedTestResult(
            passed=max(0, passed),
            failed=failed,
            errors=errors,
            xfailed=xfailed,
            xpassed=xpassed,
            skipped=skipped,
            total=total,
            duration_sec=duration,
            raw_output=output,
            metadata={"parse_method": "unittest"},
        )

    def can_parse(self, output: str) -> bool:
        """Quick check for unittest indicators."""
        if not output:
            return False
        return self.RAN_PATTERN.search(output) is not None


class JestOutputParser(OutputParser):
    """
    Parser for Jest (JavaScript/TypeScript test runner) output.

    Recognizes the Jest "Tests:" summary line, e.g.:
        Tests:       1 failed, 12 passed, 13 total
        Tests:       3 passed, 3 total
        Tests:       2 failed, 1 skipped, 5 passed, 8 total

    Jest reports a per-test "Tests:" tally (we read that, not the per-suite
    "Test Suites:" tally). Returns None when no Jest "Tests:" line is present
    (so the caller falls back to the next framework / no-signal).
    """

    # The "Tests:" summary line (status counts up to a "<N> total").
    TESTS_LINE_PATTERN = re.compile(r"^\s*Tests:\s+(?P<results>.+?)$", re.MULTILINE | re.IGNORECASE)
    # Individual "<count> <status>" tokens within that line.
    COUNT_PATTERN = re.compile(r"(\d+)\s+(passed|failed|skipped|todo|total)", re.IGNORECASE)

    @classmethod
    def name(cls) -> str:
        return "jest"

    def parse(self, output: str) -> Optional[ParsedTestResult]:
        if not output:
            return None
        m = self.TESTS_LINE_PATTERN.search(output)
        if not m:
            return None
        counts = {"passed": 0, "failed": 0, "skipped": 0, "todo": 0, "total": 0}
        found = False
        for cm in self.COUNT_PATTERN.finditer(m.group("results")):
            status = cm.group(2).lower()
            if status in counts:
                counts[status] = int(cm.group(1))
                found = True
        if not found:
            return None
        passed = counts["passed"]
        failed = counts["failed"]
        skipped = counts["skipped"] + counts["todo"]
        total = counts["total"] if counts["total"] else (passed + failed + skipped)
        return ParsedTestResult(
            passed=passed,
            failed=failed,
            skipped=skipped,
            total=total,
            raw_output=output,
            metadata={"parse_method": "jest"},
        )

    def can_parse(self, output: str) -> bool:
        if not output:
            return False
        return self.TESTS_LINE_PATTERN.search(output) is not None


class GenericOutputParser(OutputParser):
    """
    Generic fallback parser that counts PASS/FAIL/ERROR keywords.

    Less accurate but works as a fallback for unknown formats.
    """

    PASS_PATTERNS = [
        re.compile(r"\bPASS(?:ED)?\b", re.IGNORECASE),
        re.compile(r"\bOK\b"),
        re.compile(r"\bSUCCESS\b", re.IGNORECASE),
        re.compile(r"\[PASS\]", re.IGNORECASE),
        re.compile(r"✓"),
    ]

    FAIL_PATTERNS = [
        re.compile(r"\bFAIL(?:ED|URE)?\b", re.IGNORECASE),
        re.compile(r"\[FAIL\]", re.IGNORECASE),
        re.compile(r"✗"),
        re.compile(r"✘"),
    ]

    ERROR_PATTERNS = [
        re.compile(r"\bERROR\b", re.IGNORECASE),
        re.compile(r"\[ERROR\]", re.IGNORECASE),
    ]

    @classmethod
    def name(cls) -> str:
        return "generic"

    def parse(self, output: str) -> Optional[ParsedTestResult]:
        """Count pass/fail/error keywords in output."""
        if not output:
            return None

        passed = sum(len(p.findall(output)) for p in self.PASS_PATTERNS)
        failed = sum(len(p.findall(output)) for p in self.FAIL_PATTERNS)
        errors = sum(len(p.findall(output)) for p in self.ERROR_PATTERNS)

        # Only return result if we found something
        if passed == 0 and failed == 0 and errors == 0:
            return None

        return ParsedTestResult(
            passed=passed,
            failed=failed,
            errors=errors,
            raw_output=output,
            metadata={"parse_method": "generic_keywords"},
        )


class PassRatioSummaryOutputParser(OutputParser):
    """Parse pass-ratio summaries and explicit no-solution failures."""

    SUMMARY_PATTERN = re.compile(
        r"^Results:\s*(\d+)\s*/\s*(\d+)\s+passed(?:\s*\(ratio\s*=\s*[\d.]+\))?\s*$",
        re.IGNORECASE | re.MULTILINE,
    )
    NO_SOLUTION_PATTERN = re.compile(r"^No solution found at .+$", re.IGNORECASE | re.MULTILINE)

    @classmethod
    def name(cls) -> str:
        return "pass_ratio_summary"

    def parse(self, output: str) -> Optional[ParsedTestResult]:
        match = self.SUMMARY_PATTERN.search(output)
        if match is None:
            if self.NO_SOLUTION_PATTERN.search(output) is None:
                return None
            return ParsedTestResult(
                passed=0,
                failed=1,
                raw_output=output,
                metadata={"parse_method": "no_solution"},
            )
        passed = int(match.group(1))
        total = int(match.group(2))
        if passed > total:
            return None
        return ParsedTestResult(
            passed=passed,
            failed=total - passed,
            raw_output=output,
            metadata={"parse_method": "pass_ratio_summary"},
        )


# =============================================================================
# Reward Shapers
# =============================================================================


class RewardShaper(ABC):
    """
    Abstract base class for computing shaped rewards from parsed test results.

    Shapers convert ParsedTestResult into a reward value in [0, 1].
    """

    @classmethod
    @abstractmethod
    def name(cls) -> str:
        """Return the shaper name for registry lookup."""
        pass

    @abstractmethod
    def shape(
        self,
        parsed: ParsedTestResult,
        original_reward: float,
    ) -> float:
        """
        Compute shaped reward from parsed test results.

        Args:
            parsed: Structured test results from an OutputParser
            original_reward: The original reward from the verifier

        Returns:
            Shaped reward value in [0, 1]
        """
        pass


class PassRatioShaper(RewardShaper):
    """
    Simple pass ratio shaper: reward = passed / total.

    This is the most straightforward approach - partial credit proportional
    to the fraction of tests passed.
    """

    def __init__(self, **kwargs):
        # Accept and ignore any kwargs for uniform interface
        pass

    @classmethod
    def name(cls) -> str:
        return "pass_ratio"

    def shape(
        self,
        parsed: ParsedTestResult,
        original_reward: float,
    ) -> float:
        """Return simple pass ratio."""
        return parsed.pass_ratio


class EffectivePassRatioShaper(RewardShaper):
    """
    Effective pass ratio shaper: reward = effective_passed / runnable_total.

    Treats xfailed (expected failures) as passes since the test behaved
    as expected. Excludes skipped tests from the denominator.
    """

    def __init__(self, **kwargs):
        # Accept and ignore any kwargs for uniform interface
        pass

    @classmethod
    def name(cls) -> str:
        return "effective_pass_ratio"

    def shape(
        self,
        parsed: ParsedTestResult,
        original_reward: float,
    ) -> float:
        """Return effective pass ratio."""
        return parsed.effective_pass_ratio


class WeightedShaper(RewardShaper):
    """
    Weighted shaper with configurable weights for different outcomes.

    reward = (w_pass * passed + w_xfail * xfailed - w_fail * failed - w_error * errors) / total

    Allows penalizing errors more heavily than failures, or giving
    partial credit for xfailed tests.
    """

    def __init__(
        self,
        weight_pass: float = 1.0,
        weight_xfail: float = 1.0,
        weight_fail: float = 0.0,
        weight_error: float = 0.0,
        weight_xpass: float = 0.5,  # Unexpected pass - partial credit
        **kwargs,  # Accept and ignore extra kwargs
    ):
        self.weight_pass = weight_pass
        self.weight_xfail = weight_xfail
        self.weight_fail = weight_fail
        self.weight_error = weight_error
        self.weight_xpass = weight_xpass

    @classmethod
    def name(cls) -> str:
        return "weighted"

    def shape(
        self,
        parsed: ParsedTestResult,
        original_reward: float,
    ) -> float:
        """Compute weighted reward."""
        if parsed.runnable_total == 0:
            return 0.0

        score = (
            self.weight_pass * parsed.passed
            + self.weight_xfail * parsed.xfailed
            + self.weight_fail * parsed.failed
            + self.weight_error * parsed.errors
            + self.weight_xpass * parsed.xpassed
        )

        # Normalize to [0, 1]
        max_score = self.weight_pass * parsed.runnable_total
        if max_score <= 0:
            return 0.0

        return max(0.0, min(1.0, score / max_score))


class ThresholdShaper(RewardShaper):
    """
    Threshold-based shaper with configurable pass threshold.

    Returns 1.0 if pass_ratio >= threshold, else returns scaled pass_ratio.
    Useful for "almost passing" scenarios where you want to reward
    getting close to full success.
    """

    def __init__(
        self,
        threshold: float = 1.0,
        below_threshold_scale: float = 0.5,
        **kwargs,  # Accept and ignore extra kwargs
    ):
        """
        Args:
            threshold: Pass ratio threshold for full reward (default 1.0 = all tests)
            below_threshold_scale: Scale factor for rewards below threshold
        """
        self.threshold = threshold
        self.below_threshold_scale = below_threshold_scale

    @classmethod
    def name(cls) -> str:
        return "threshold"

    def shape(
        self,
        parsed: ParsedTestResult,
        original_reward: float,
    ) -> float:
        """Apply threshold-based shaping."""
        ratio = parsed.effective_pass_ratio

        if ratio >= self.threshold:
            return 1.0

        # Scale the ratio for below-threshold results
        return ratio * self.below_threshold_scale


class BinaryWithPartialCreditShaper(RewardShaper):
    """
    Binary reward with optional partial credit for near-successes.

    - If all tests pass: reward = 1.0
    - If >= partial_threshold pass: reward = partial_credit
    - Otherwise: reward = 0.0

    Useful when you want mostly binary rewards but give some credit
    for getting close.
    """

    def __init__(
        self,
        partial_threshold: float = 0.9,
        partial_credit: float = 0.5,
        **kwargs,  # Accept and ignore extra kwargs
    ):
        self.partial_threshold = partial_threshold
        self.partial_credit = partial_credit

    @classmethod
    def name(cls) -> str:
        return "binary_partial"

    def shape(
        self,
        parsed: ParsedTestResult,
        original_reward: float,
    ) -> float:
        """Apply binary with partial credit shaping."""
        ratio = parsed.effective_pass_ratio

        if ratio >= 1.0:
            return 1.0
        elif ratio >= self.partial_threshold:
            return self.partial_credit
        else:
            return 0.0


class OriginalRewardShaper(RewardShaper):
    """
    Pass-through shaper that returns the original reward unchanged.

    Useful as a no-op option when reward shaping is disabled.
    """

    def __init__(self, **kwargs):
        # Accept and ignore any kwargs for uniform interface
        pass

    @classmethod
    def name(cls) -> str:
        return "original"

    def shape(
        self,
        parsed: ParsedTestResult,
        original_reward: float,
    ) -> float:
        """Return original reward unchanged."""
        return original_reward


# =============================================================================
# Trajectory-Based Reward Shapers
# =============================================================================
#
# These shapers operate on the agent's conversation trajectory rather than
# (or in addition to) test output. They reward behavioral qualities like
# thinking before acting and producing well-formatted responses.


# Shared regex for extracting <think>...</think> blocks from content
_THINK_BLOCK_PATTERN = re.compile(r"<think>(.*?)</think>", re.DOTALL)


def _extract_chat_history(chat_history: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Safely extract and validate chat history."""
    if not chat_history or not isinstance(chat_history, list):
        return []
    return chat_history


def _get_assistant_messages(chat_history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Extract assistant messages from chat history."""
    return [m for m in chat_history if m.get("role") == "assistant"]


# Strip <think>...</think> reasoning so two turns that *act* identically but
# reason differently still collapse to the same action payload (and, conversely,
# so a turn's reasoning never spuriously makes two identical actions look
# distinct). Anti-thrash keys on the ACTION, not the prose around it.
_THINK_STRIP_PATTERN = _THINK_BLOCK_PATTERN  # alias for intent at the call site


def _normalize_action_payload(text: str) -> str:
    """Normalize an action payload for byte-identity comparison.

    Anti-thrash keys on the exact *payload* (the command string + any heredoc
    body), normalized for **trivial whitespace only** so that two emissions that
    differ solely by incidental indentation / trailing spaces / blank-line
    padding still compare equal, while any real content difference (a different
    file body, a changed flag, a different command) does NOT.

    Normalization (whitespace-only, content-preserving):
      - strip a leading/trailing whitespace,
      - collapse runs of spaces/tabs *within a line* to a single space,
      - drop trailing whitespace on each line,
      - drop blank lines.
    The per-line structure (and thus the heredoc body content) is preserved, so
    e.g. a `cat > f.py <<EOF ... EOF` with one set of bytes is distinct from the
    same heredoc with different bytes.
    """
    if not text:
        return ""
    lines = []
    for raw_line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        # Collapse internal runs of spaces/tabs and trim the line.
        collapsed = re.sub(r"[ \t]+", " ", raw_line).strip()
        if collapsed:
            lines.append(collapsed)
    return "\n".join(lines)


def _extract_action_payload(msg: Dict[str, Any]) -> str:
    """Extract a single assistant turn's *action payload* as a normalized string.

    The payload is what the agent actually *did* this turn: the command /
    heredoc text it emitted (assistant ``content``, with any ``<think>`` block
    removed) plus a canonical serialization of any structured ``tool_calls``
    (function name AND its arguments — keyed on the full payload, not just the
    tool type, so re-running the same tool with *different* arguments is NOT a
    repeat). Returns ``""`` for a turn that carries no actionable payload (those
    are ignored by the detector — an empty action is never a "thrash").
    """
    parts: List[str] = []

    content = msg.get("content") or ""
    if not isinstance(content, str):
        content = str(content)
    # The action is the command/heredoc the model emits; its reasoning is not
    # part of the action payload.
    content_no_think = _THINK_STRIP_PATTERN.sub("", content)
    content_norm = _normalize_action_payload(content_no_think)
    if content_norm:
        parts.append(content_norm)

    tool_calls = msg.get("tool_calls") or []
    if isinstance(tool_calls, list):
        for tc in tool_calls:
            if not isinstance(tc, dict):
                continue
            name = tc.get("function_name") or (tc.get("function") or {}).get("name") or ""
            # Arguments may live under tc["arguments"] or tc["function"]["arguments"]
            # and be a dict or a (possibly already JSON) string. Serialize
            # canonically so identical argument *content* compares equal
            # regardless of dict key order / incidental whitespace.
            args = tc.get("arguments")
            if args is None:
                args = (tc.get("function") or {}).get("arguments")
            args_repr = _canonical_args(args)
            parts.append(f"{name}\n{args_repr}")

    return "\n".join(p for p in parts if p)


def _canonical_args(args: Any) -> str:
    """Canonicalize tool-call arguments to a stable string for comparison."""
    if args is None:
        return ""
    if isinstance(args, str):
        # Might be a JSON string; if so, re-serialize sorted for stability.
        try:
            import json

            parsed = json.loads(args)
            return json.dumps(parsed, sort_keys=True, separators=(",", ":"))
        except (ValueError, TypeError):
            return _normalize_action_payload(args)
    try:
        import json

        return json.dumps(args, sort_keys=True, separators=(",", ":"), default=str)
    except (TypeError, ValueError):
        return _normalize_action_payload(str(args))


def detect_repeated_actions(trajectory_or_chat: Optional[List[Dict[str, Any]]]) -> int:
    """Count byte-identical *consecutive* action repeats in a trajectory.

    The anti-thrash signal (Stage 2 / M3). For each assistant turn we extract a
    normalized **action payload** (the command/heredoc text + a canonical
    serialization of any structured tool-call name+arguments — see
    ``_extract_action_payload``), considering only turns that carry an actionable
    payload (empty / non-action turns are skipped and do NOT break a run). We
    then count repeats as the number of action turns whose payload is identical
    to the *immediately preceding action turn's* payload — i.e. the length of
    each maximal run of byte-identical actions minus one, summed over runs.

    KEY (the load-bearing correctness property): dedup is keyed on the action's
    exact PAYLOAD (content), NOT the action type, and a repeat only counts when
    it is **consecutive** (an uninterrupted re-emission of the same payload — the
    stuck-in-a-loop thrash). This is what makes the GOOD loop safe:

      - ``cat > f.py <<EOF …same bytes…`` ×4 in a row -> a run of 4 -> 3 repeats.
      - ``editA, pytest, editB, pytest`` -> the two ``pytest`` turns are NOT
        consecutive (a *different* edit sits between them), and ``editA`` /
        ``editB`` differ -> 0 repeats. Re-running the test after real new work is
        the loop we WANT, never a thrash.

    Returns the integer repeat count (0 if no repeats / empty trajectory).
    """
    messages = _extract_chat_history(trajectory_or_chat)
    if not messages:
        return 0

    repeats = 0
    prev_payload: Optional[str] = None
    for msg in messages:
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        payload = _extract_action_payload(msg)
        if not payload:
            # A turn with no actionable payload neither counts nor breaks the run.
            continue
        if prev_payload is not None and payload == prev_payload:
            repeats += 1
        prev_payload = payload
    return repeats


class TrajectoryRewardShaper(ABC):
    """
    Abstract base class for trajectory-based reward shapers.

    These shapers compute reward signals from the agent's conversation
    trajectory (chat_history) rather than test output. They can be used
    standalone or combined with verifier-based shapers via CompositeShaper.
    """

    @classmethod
    @abstractmethod
    def name(cls) -> str:
        """Return the shaper name for registry lookup."""
        pass

    @abstractmethod
    def shape(
        self,
        chat_history: List[Dict[str, Any]],
        original_reward: float,
    ) -> float:
        """
        Compute shaped reward from agent trajectory.

        Args:
            chat_history: List of message dicts with 'role' and 'content' keys.
            original_reward: The original reward from the verifier.

        Returns:
            Shaped reward value in [0, 1].
        """
        pass


class ThinkingLengthShaper(TrajectoryRewardShaper):
    """
    Rewards thinking blocks of moderate length, penalizing too-short
    (rushing) or too-long (wasting context) thinking.

    Uses a Gaussian curve centered on `target_tokens` with width `sigma`.
    The reward is averaged across all assistant turns that contain thinking.
    Turns without thinking blocks get reward 0.

    Config params:
        target_tokens: Center of the Gaussian (default: 750)
        sigma_tokens: Width of the Gaussian (default: 250)
        min_thinking_turns_ratio: Minimum fraction of assistant turns that
            should contain thinking (default: 0.5). If fewer turns think,
            the reward is scaled down proportionally.
    """

    def __init__(
        self,
        target_tokens: int = 750,
        sigma_tokens: int = 250,
        min_thinking_turns_ratio: float = 0.5,
        **kwargs,
    ):
        self.target_tokens = target_tokens
        self.sigma_tokens = sigma_tokens
        self.min_thinking_turns_ratio = min_thinking_turns_ratio

    @classmethod
    def name(cls) -> str:
        return "thinking_length"

    def _count_think_tokens_approx(self, content: str) -> int:
        """Count approximate tokens in <think> blocks (chars / 4)."""
        total_chars = 0
        for match in _THINK_BLOCK_PATTERN.finditer(content):
            total_chars += len(match.group(1))
        return total_chars // 4  # rough char-to-token ratio

    def shape(
        self,
        chat_history: List[Dict[str, Any]],
        original_reward: float,
    ) -> float:
        assistant_msgs = _get_assistant_messages(_extract_chat_history(chat_history))
        if not assistant_msgs:
            return 0.0

        turn_rewards = []
        turns_with_thinking = 0

        for msg in assistant_msgs:
            content = msg.get("content", "") or ""
            think_tokens = self._count_think_tokens_approx(content)

            if think_tokens > 0:
                turns_with_thinking += 1
                # Gaussian reward centered on target
                exponent = -0.5 * ((think_tokens - self.target_tokens) / self.sigma_tokens) ** 2
                turn_rewards.append(math.exp(exponent))
            else:
                turn_rewards.append(0.0)

        if not turn_rewards:
            return 0.0

        # Average turn reward
        avg_reward = sum(turn_rewards) / len(turn_rewards)

        # Penalize if too few turns have thinking
        thinking_ratio = turns_with_thinking / len(assistant_msgs)
        if thinking_ratio < self.min_thinking_turns_ratio:
            avg_reward *= thinking_ratio / self.min_thinking_turns_ratio

        return min(1.0, max(0.0, avg_reward))


class FormatQualityShaper(TrajectoryRewardShaper):
    """
    Rewards well-formed JSON responses from the Terminus-2 agent.

    Parses each assistant message to check if it contains valid JSON with
    the expected structure (analysis, plan, commands). Rewards the fraction
    of turns with clean, parseable responses.

    This shaper does a lightweight structural check — it doesn't re-run
    the full Terminus parser, but checks for the key structural elements
    that indicate a well-formed response.

    Config params:
        required_fields: Fields that must be present for a "clean" parse
            (default: ["analysis", "plan", "commands"])
        penalize_truncated_json: Whether to penalize responses where JSON
            appears truncated / auto-closed (default: True)
    """

    # Match a JSON object (greedy, may need cleanup)
    _JSON_PATTERN = re.compile(r"\{.*\}", re.DOTALL)

    def __init__(
        self,
        required_fields: Optional[List[str]] = None,
        penalize_truncated_json: bool = True,
        **kwargs,
    ):
        self.required_fields = required_fields or ["analysis", "plan", "commands"]
        self.penalize_truncated_json = penalize_truncated_json

    @classmethod
    def name(cls) -> str:
        return "format_quality"

    def _score_message(self, content: str) -> float:
        """Score a single assistant message for JSON format quality.

        Returns:
            1.0: Clean, valid JSON with all required fields
            0.5: Valid JSON but missing some fields or truncated
            0.0: No valid JSON found
        """
        if not content:
            return 0.0

        # Strip thinking blocks before checking JSON
        content_no_think = _THINK_BLOCK_PATTERN.sub("", content).strip()
        if not content_no_think:
            return 0.0

        # Try to find JSON object
        json_match = self._JSON_PATTERN.search(content_no_think)
        if not json_match:
            return 0.0

        json_str = json_match.group(0)

        # Try to parse
        try:
            import json

            parsed = json.loads(json_str)
        except json.JSONDecodeError:
            # Check if it looks like truncated JSON (common failure mode)
            if self.penalize_truncated_json and json_str.count("{") != json_str.count("}"):
                return 0.0
            # Try adding closing braces
            try:
                fixed = json_str
                while fixed.count("{") > fixed.count("}"):
                    fixed += "}"
                parsed = json.loads(fixed)
            except (json.JSONDecodeError, Exception):
                return 0.0
            # Parsed with auto-fix — partial credit
            return self._check_fields(parsed, partial=True)

        return self._check_fields(parsed, partial=False)

    def _check_fields(self, parsed: dict, partial: bool) -> float:
        """Check if parsed JSON has required fields."""
        if not isinstance(parsed, dict):
            return 0.0

        present = sum(1 for f in self.required_fields if f in parsed)
        total = len(self.required_fields)

        if total == 0:
            return 1.0

        field_ratio = present / total

        if partial:
            # Auto-fixed JSON — cap at 0.5
            return min(0.5, field_ratio * 0.5)

        return field_ratio

    def shape(
        self,
        chat_history: List[Dict[str, Any]],
        original_reward: float,
    ) -> float:
        assistant_msgs = _get_assistant_messages(_extract_chat_history(chat_history))
        if not assistant_msgs:
            return 0.0

        scores = [self._score_message(msg.get("content", "")) for msg in assistant_msgs]
        return sum(scores) / len(scores)


class CommandQualityShaper(TrajectoryRewardShaper):
    """
    Rewards episodes where the agent's commands produce clean execution
    (no errors) and penalizes episodes with many invalid commands.

    Scores each user message (environment response) for signs of command
    errors: shell error messages, help/usage dumps (indicating wrong syntax),
    Python tracebacks, permission errors, etc.

    The score per turn is:
        1.0: Clean output — no error indicators detected
        0.0: Clear error — error pattern matched in environment output

    Episode score is the ratio of clean turns to total command turns.

    Config params:
        error_penalty_weight: How much to weight error turns vs clean turns
            (default: 1.0, meaning errors count equally against clean turns)
        min_turns: Minimum number of command turns required to produce a
            non-trivial score (default: 2). Episodes with fewer turns get 0.5.
    """

    # Shell error patterns (matched against user messages = environment output)
    _ERROR_PATTERNS = [
        # Command not found / bad syntax
        re.compile(r"command not found", re.IGNORECASE),
        re.compile(r"No such file or directory", re.IGNORECASE),
        re.compile(r"Permission denied", re.IGNORECASE),
        re.compile(r"not recognized as an? (internal|external) command", re.IGNORECASE),
        re.compile(r"cannot execute binary file", re.IGNORECASE),
        # Help/usage dumps (tool printed usage = wrong invocation)
        re.compile(r"^[Uu]sage:\s+\S+", re.MULTILINE),
        re.compile(r"^Try '.*--help'", re.MULTILINE),
        re.compile(r"unrecognized option", re.IGNORECASE),
        re.compile(r"invalid option", re.IGNORECASE),
        re.compile(r"unknown option", re.IGNORECASE),
        re.compile(r"illegal option", re.IGNORECASE),
        # Python tracebacks
        re.compile(r"^Traceback \(most recent call last\)", re.MULTILINE),
        re.compile(r"^SyntaxError:", re.MULTILINE),
        re.compile(r"^NameError:", re.MULTILINE),
        re.compile(r"^TypeError:", re.MULTILINE),
        re.compile(r"^ModuleNotFoundError:", re.MULTILINE),
        re.compile(r"^ImportError:", re.MULTILINE),
        re.compile(r"^FileNotFoundError:", re.MULTILINE),
        # Git errors
        re.compile(r"fatal: not a git repository", re.IGNORECASE),
        re.compile(r"fatal: ", re.IGNORECASE),
        re.compile(r"error: pathspec .* did not match", re.IGNORECASE),
        # General error markers
        re.compile(r"^ERROR:", re.MULTILINE),
        re.compile(r"^Error:", re.MULTILINE),
        re.compile(r"^\[ERROR\]", re.MULTILINE),
        # Segfault / killed
        re.compile(r"Segmentation fault", re.IGNORECASE),
        re.compile(r"^Killed$", re.MULTILINE),
    ]

    def __init__(
        self,
        error_penalty_weight: float = 1.0,
        min_turns: int = 2,
        **kwargs,
    ):
        self.error_penalty_weight = error_penalty_weight
        self.min_turns = min_turns

    @classmethod
    def name(cls) -> str:
        return "command_quality"

    def _score_turn(self, env_output: str) -> float:
        """Score a single environment response (user message).

        Returns:
            1.0: No error patterns detected (clean execution)
            0.0: Error pattern matched
        """
        if not env_output or not env_output.strip():
            # Empty output — not an error, but not informative either
            return 1.0

        for pattern in self._ERROR_PATTERNS:
            if pattern.search(env_output):
                return 0.0

        return 1.0

    def shape(
        self,
        chat_history: List[Dict[str, Any]],
        original_reward: float,
    ) -> float:
        messages = _extract_chat_history(chat_history)
        if not messages:
            return 0.5

        # User messages after the first one are environment responses to commands.
        # The first user message is the task prompt — skip it.
        user_msgs = [m for m in messages if m.get("role") == "user"]
        command_turns = user_msgs[1:] if len(user_msgs) > 1 else []

        if len(command_turns) < self.min_turns:
            return 0.5  # Not enough data to judge

        scores = [self._score_turn(m.get("content", "")) for m in command_turns]
        n_clean = sum(s for s in scores)
        n_error = len(scores) - n_clean

        # Weighted ratio: clean / (clean + weight * error)
        denominator = n_clean + self.error_penalty_weight * n_error
        if denominator == 0:
            return 0.5

        return n_clean / denominator


class CompositeShaper:
    """
    Combines multiple reward signals (verifier-based + trajectory-based)
    into a single weighted reward.

    Each component is identified by name and assigned a weight. The final
    reward is a weighted average of all components, normalized to [0, 1].

    Config params:
        components: Dict mapping component name to weight, e.g.:
            {"verifier": 0.5, "thinking_length": 0.3, "format_quality": 0.2}
        verifier_shaper: Name of the verifier-based shaper to use for the
            "verifier" component (default: "pass_ratio")
        trajectory_shapers: Dict mapping trajectory shaper names to their
            kwargs, e.g.: {"thinking_length": {"target_tokens": 750}}
    """

    def __init__(
        self,
        components: Optional[Dict[str, float]] = None,
        verifier_shaper: str = "pass_ratio",
        trajectory_shaper_kwargs: Optional[Dict[str, Dict]] = None,
        **kwargs,
    ):
        self.components = components or {
            "verifier": 0.55,
            "thinking_length": 0.15,
            "format_quality": 0.15,
            "command_quality": 0.15,
        }
        self.verifier_shaper_name = verifier_shaper
        self.trajectory_shaper_kwargs = trajectory_shaper_kwargs or {}

        # Instantiate trajectory shapers
        self._trajectory_shapers: Dict[str, TrajectoryRewardShaper] = {}
        for comp_name in self.components:
            if comp_name == "verifier":
                continue
            if comp_name in _TRAJECTORY_SHAPER_REGISTRY:
                shaper_kwargs = self.trajectory_shaper_kwargs.get(comp_name, {})
                self._trajectory_shapers[comp_name] = _TRAJECTORY_SHAPER_REGISTRY[comp_name](**shaper_kwargs)
            else:
                logger.warning(
                    f"CompositeShaper: unknown trajectory shaper '{comp_name}', "
                    f"available: {list(_TRAJECTORY_SHAPER_REGISTRY.keys())}"
                )

        # Normalize weights
        total_weight = sum(self.components.values())
        if total_weight > 0:
            self._normalized_weights = {k: v / total_weight for k, v in self.components.items()}
        else:
            self._normalized_weights = self.components

    @classmethod
    def name(cls) -> str:
        return "composite"

    def shape(
        self,
        parsed: Optional[ParsedTestResult],
        original_reward: float,
        chat_history: Optional[List[Dict[str, Any]]] = None,
    ) -> float:
        """
        Compute composite reward from verifier results and trajectory.

        Args:
            parsed: Parsed test results (may be None if parsing failed)
            original_reward: Original binary reward from verifier
            chat_history: Agent conversation trajectory

        Returns:
            Weighted composite reward in [0, 1]
        """
        final_reward, _ = self.shape_with_components(parsed, original_reward, chat_history)
        return final_reward

    def shape_with_components(
        self,
        parsed: Optional[ParsedTestResult],
        original_reward: float,
        chat_history: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute composite reward and return per-component breakdown.

        Returns:
            Tuple of (final_reward, component_rewards_dict)
            where component_rewards_dict maps component name to its raw
            (unweighted) reward value in [0, 1].
        """
        component_rewards = {}

        # Verifier component
        if "verifier" in self._normalized_weights:
            if parsed is not None:
                verifier_shaper = get_reward_shaper(self.verifier_shaper_name)
                component_rewards["verifier"] = verifier_shaper.shape(parsed, original_reward)
            else:
                component_rewards["verifier"] = original_reward

        # Trajectory components
        for comp_name, shaper in self._trajectory_shapers.items():
            component_rewards[comp_name] = shaper.shape(chat_history or [], original_reward)

        # Weighted sum
        final_reward = 0.0
        for comp_name, weight in self._normalized_weights.items():
            reward = component_rewards.get(comp_name, 0.0)
            final_reward += weight * reward

        return min(1.0, max(0.0, final_reward)), component_rewards


# =============================================================================
# Composite Loop Shaper (loop-behavior reward shaping — Stage 0 scaffold)
# =============================================================================
#
# `composite_loop` is a *container* shaper for the loop-behavior reward-shaping
# program (notes/RL/skyrl/loop_behavior_reward_plan.md). It sums an outcome term
# (the existing binary / pass_ratio reward) plus zero-or-more bounded behavioral
# components (termination, anti-thrash, ... — added in Stages 1-2).
#
# STAGE 0 INVARIANT (G1): with all components disabled and outcome_weight=1.0,
# `composite_loop` reproduces today's reward *bit-for-bit* — it is exactly the
# value of the configured outcome shaper (default `pass_ratio`). No component is
# wired yet; this lands the container, the config surface, the G2 clamp, and the
# fully-populated `reward_components` dict (every key present, value 0.0).
#
# G2 GROUND-TRUTH ANCHOR: the summed *shaped* delta (everything except the
# outcome term) is clamped to +/- total_shaping_cap, and a failing trajectory
# (verifier reward == 0) can never be moved net-positive by shaping alone.


# Default loop-shaping config (the no-op). Mirrors the yaml block in the Stage-0
# spec. Every component is disabled with zero weights; with these defaults the
# composite reduces to the outcome term.
DEFAULT_LOOP_SHAPING_CONFIG: Dict[str, Any] = {
    "outcome_weight": 1.0,
    # Stage 1 — termination as a high-stakes learned action.
    # Disabled by default with zero magnitudes (G1 byte-identity). The spec
    # opt-in magnitudes (green_bonus 0.3, red_penalty 0.3, noterm_penalty 0.2)
    # are applied as fallbacks inside the component logic when a config sets
    # ``enabled: true`` but omits an explicit magnitude (see
    # ``_TERMINATE_DEFAULTS`` / ``_compute_terminate``).
    "terminate": {
        "enabled": False,
        "green_bonus": 0.0,
        "red_penalty": 0.0,
        "noterm_penalty": 0.0,
    },
    # Stage 2 — anti-thrash penalty (repeated byte-identical actions).
    "antithrash": {
        "enabled": False,
        "per_repeat_penalty": 0.0,
        "cap": 0.0,
    },
    # G2: |sum of shaped components| is clamped to this.
    "total_shaping_cap": 0.3,
}

# The set of behavioral component keys the composite always reports (even at 0.0
# in Stage 0). The outcome term is reported separately under "outcome".
_LOOP_COMPONENT_KEYS: Tuple[str, ...] = ("terminate", "antithrash")

# Stage 1 opt-in magnitude fallbacks for the terminate component. Applied only
# when ``terminate.enabled`` is True AND the config left the magnitude at 0.0
# (the disabled default). A config that explicitly sets a magnitude (including
# 0.0 while enabled, via a sentinel != the default) keeps its value — see
# ``_resolve_terminate_magnitude``.
_TERMINATE_DEFAULTS: Dict[str, float] = {
    "green_bonus": 0.3,
    "red_penalty": 0.3,
    "noterm_penalty": 0.2,
}

# Stage 2 opt-in magnitude fallbacks for the antithrash component. Same pattern
# as ``_TERMINATE_DEFAULTS``: applied only when ``antithrash.enabled`` is True
# AND the config left the magnitude at its 0.0 disabled default, so "just enable
# antithrash" yields the spec behaviour (0.02 per repeat, clamped at 0.1) without
# re-stating the magnitudes. An explicit positive value is honoured verbatim.
_ANTITHRASH_DEFAULTS: Dict[str, float] = {
    "per_repeat_penalty": 0.02,
    "cap": 0.1,
}


def _deep_merge_loop_config(overrides: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Merge user loop_shaping overrides onto DEFAULT_LOOP_SHAPING_CONFIG.

    A fresh copy of the defaults is taken; nested component dicts are merged
    key-by-key so a partial override (e.g. only ``terminate.enabled``) keeps the
    other defaulted fields. Unknown top-level keys are passed through.
    """
    merged: Dict[str, Any] = {
        "outcome_weight": DEFAULT_LOOP_SHAPING_CONFIG["outcome_weight"],
        "total_shaping_cap": DEFAULT_LOOP_SHAPING_CONFIG["total_shaping_cap"],
    }
    for comp_key in _LOOP_COMPONENT_KEYS:
        merged[comp_key] = dict(DEFAULT_LOOP_SHAPING_CONFIG[comp_key])

    if not overrides:
        return merged

    for key, value in overrides.items():
        if key in _LOOP_COMPONENT_KEYS and isinstance(value, dict):
            merged[key].update(value)
        else:
            merged[key] = value
    return merged


class CompositeLoopShaper:
    """Container shaper that sums an outcome term + bounded behavioral components.

    Stage 0: the container is wired to *nothing* — all components are disabled by
    default, so the reward equals ``outcome_weight * <outcome shaper>`` which, with
    ``outcome_weight == 1.0`` and the default ``pass_ratio`` outcome shaper, is
    bit-identical to today's reward.

    The shaped delta (sum of behavioral components) is clamped to
    ``+/- total_shaping_cap`` (G2) and may never flip a failing trajectory
    (``original_reward == 0``) to a net-positive total.

    Config (passed via ``loop_shaping`` kwarg, merged onto the defaults):
        loop_shaping:
          outcome_weight: 1.0
          terminate:  {enabled: false, green_bonus: 0.0, red_penalty: 0.0, noterm_penalty: 0.0}
          antithrash: {enabled: false, per_repeat_penalty: 0.0, cap: 0.0}
          total_shaping_cap: 0.3
        outcome_shaper: pass_ratio   # which verifier-based shaper computes the outcome term
    """

    def __init__(
        self,
        loop_shaping: Optional[Dict[str, Any]] = None,
        outcome_shaper: str = "pass_ratio",
        outcome_parser: Optional[str] = None,
        outcome_fallback_to_original: bool = True,
        outcome_shaper_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,  # tolerate (and ignore) extra kwargs for a uniform interface
    ):
        self.config = _deep_merge_loop_config(loop_shaping)
        self.outcome_shaper_name = outcome_shaper
        self.outcome_parser = outcome_parser
        self.outcome_fallback_to_original = outcome_fallback_to_original
        self.outcome_shaper_kwargs = outcome_shaper_kwargs or {}

    @classmethod
    def name(cls) -> str:
        return "composite_loop"

    def _compute_outcome_reward(
        self,
        stdout: Optional[str],
        original_reward: float,
        chat_history: Optional[List[Dict[str, Any]]],
    ) -> float:
        """Outcome term = exactly the existing single-shaper reward path.

        Delegates to ``shape_reward_from_output`` with the configured outcome
        shaper so the value is, by construction, identical to running that shaper
        standalone (the byte-identity guarantee for Stage 0).
        """
        return shape_reward_from_output(
            stdout=stdout,
            original_reward=original_reward,
            parser_name=self.outcome_parser,
            shaper_name=self.outcome_shaper_name,
            shaper_kwargs=self.outcome_shaper_kwargs,
            fallback_to_original=self.outcome_fallback_to_original,
            chat_history=chat_history,
        )

    @staticmethod
    def _resolve_terminate_magnitude(cfg: Dict[str, Any], key: str) -> float:
        """Resolve a terminate magnitude: explicit positive config value, else
        the Stage-1 spec fallback (``_TERMINATE_DEFAULTS``).

        The merged default for each magnitude is 0.0 (the disabled value). When a
        config opts in (``enabled: true``) but leaves a magnitude at the 0.0
        default, we substitute the spec fallback so "just enable terminate"
        yields the intended +0.3 / -0.3 / -0.2 behaviour. A config wanting a
        custom magnitude simply sets it; any value > 0 is honoured verbatim.
        """
        try:
            val = abs(float(cfg.get(key, 0.0)))
        except (TypeError, ValueError):
            val = 0.0
        if val > 0.0:
            return val
        return _TERMINATE_DEFAULTS.get(key, 0.0)

    def _compute_terminate(
        self,
        trajectory_context: Optional[Dict[str, Any]],
    ) -> float:
        """Stage 1 (M2) "only-then-complete" termination component.

        Returns a raw *signed* delta (clamped downstream by the G2 cap):
          - mark_complete AND verifier green (>0): ``+green_bonus``
          - mark_complete AND verifier failing (==0): ``-red_penalty``
          - never terminated (premature_stop / wall): ``-noterm_penalty``

        The green bonus only fires on an already-green verifier, so it can never
        rescue a failing trajectory (G2). When no trajectory context is supplied
        (Stage-0 call sites), the component contributes 0.0 — preserving
        byte-identity for callers that don't thread the signals.
        """
        terminate_cfg = self.config.get("terminate", {})
        if not terminate_cfg.get("enabled", False):
            return 0.0
        if not trajectory_context:
            # Enabled but no signals available -> no contribution (and no crash).
            return 0.0

        mark_complete = bool(trajectory_context.get("mark_complete", False))
        premature_stop = bool(trajectory_context.get("premature_stop", False))
        try:
            verifier_reward = float(trajectory_context.get("verifier_reward", 0.0))
        except (TypeError, ValueError):
            verifier_reward = 0.0

        green_bonus = self._resolve_terminate_magnitude(terminate_cfg, "green_bonus")
        red_penalty = self._resolve_terminate_magnitude(terminate_cfg, "red_penalty")
        noterm_penalty = self._resolve_terminate_magnitude(terminate_cfg, "noterm_penalty")

        if mark_complete:
            if verifier_reward > 0.0:
                # Finished, and the hidden tests agree it's green -> reward it.
                return +green_bonus
            # Claimed done but the verifier is red -> discourage premature "done".
            return -red_penalty

        # Did not mark complete. If it ran to the wall (never terminated), penalize
        # the over-iterate-never-stop mode. (premature_stop is set by the caller
        # whenever the trajectory ended without a confirmed completion.)
        if premature_stop:
            return -noterm_penalty

        # Terminated for some other reason without mark_complete and not flagged
        # as a wall stop -> no terminate signal.
        return 0.0

    @staticmethod
    def _resolve_antithrash_magnitude(cfg: Dict[str, Any], key: str) -> float:
        """Resolve an antithrash magnitude: explicit positive config value, else
        the Stage-2 spec fallback (``_ANTITHRASH_DEFAULTS``).

        Mirrors ``_resolve_terminate_magnitude``: the merged default is 0.0 (the
        disabled value); when a config opts in (``enabled: true``) but leaves a
        magnitude at the 0.0 default, the spec fallback is substituted.
        """
        try:
            val = abs(float(cfg.get(key, 0.0)))
        except (TypeError, ValueError):
            val = 0.0
        if val > 0.0:
            return val
        return _ANTITHRASH_DEFAULTS.get(key, 0.0)

    def _compute_antithrash(
        self,
        chat_history: Optional[List[Dict[str, Any]]],
    ) -> float:
        """Stage 2 (M3) anti-thrash component.

        Returns a raw *signed* (negative) delta: ``-per_repeat_penalty`` per
        byte-identical *consecutive* action repeat beyond the first
        (``detect_repeated_actions``), summed and clamped to ``-cap``. The signal
        is computed from ``chat_history`` ALONE (the assistant turns carry the
        action payloads) — no ``trajectory_context`` needed.

        Content-keyed dedup (not type-keyed) + consecutive-only counting means a
        legitimate re-run of a command after a *different* edit (the GOOD loop) is
        never penalized — only genuine identical-payload churn is. When disabled,
        or with no repeats, the component contributes 0.0 (preserving Stage-0/1
        byte-identity).
        """
        antithrash_cfg = self.config.get("antithrash", {})
        if not antithrash_cfg.get("enabled", False):
            return 0.0

        repeats = detect_repeated_actions(chat_history)
        if repeats <= 0:
            return 0.0

        per_repeat = self._resolve_antithrash_magnitude(antithrash_cfg, "per_repeat_penalty")
        cap = self._resolve_antithrash_magnitude(antithrash_cfg, "cap")

        raw = per_repeat * repeats
        # Component-level clamp at -cap (the G2 total clamp applies again downstream).
        penalty = min(raw, cap)
        return -penalty

    def _compute_components(
        self,
        stdout: Optional[str],
        original_reward: float,
        chat_history: Optional[List[Dict[str, Any]]],
        trajectory_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        """Compute each behavioral component's *raw signed delta* (pre-clamp).

        Stage 1: the ``terminate`` component is live (gated on
        ``terminate.enabled`` and on a supplied ``trajectory_context``); all
        other components remain disabled -> 0.0. The dict shape (one key per
        component, always present) is fixed here so runs are inspectable.
        """
        components: Dict[str, float] = {}

        # Stage 1 — terminate (M2 "only-then-complete").
        components["terminate"] = self._compute_terminate(trajectory_context)

        # Stage 2 — antithrash (M3, repeated byte-identical actions). Gated on
        # ``antithrash.enabled``; computed from chat_history alone. Disabled -> 0.0.
        components["antithrash"] = self._compute_antithrash(chat_history)

        return components

    def shape_with_components(
        self,
        stdout: Optional[str],
        original_reward: float,
        chat_history: Optional[List[Dict[str, Any]]] = None,
        trajectory_context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[float, Dict[str, float]]:
        """Compute the composite reward and a per-component breakdown.

        ``trajectory_context`` (Stage 1+) carries the finished-trajectory signals
        the behavioral components need but that are NOT derivable from
        ``stdout``/``chat_history`` alone:
          - ``mark_complete`` (bool): did the trajectory end in a confirmed
            ``mark_task_complete``?
          - ``verifier_reward`` (float): the ground-truth verifier verdict.
          - ``premature_stop`` (bool): did it run to the agent/step wall without
            confirming completion (never-terminated)?
        When ``trajectory_context`` is None (Stage-0 call sites), every behavioral
        component contributes 0.0 -> byte-identical to Stage 0 (G1).

        Returns ``(final_reward, reward_components)`` where ``reward_components``
        carries ``outcome``, the clamped ``shaping_total``, and every behavioral
        component key (raw signed value).
        """
        outcome_weight = float(self.config.get("outcome_weight", 1.0))
        total_shaping_cap = abs(float(self.config.get("total_shaping_cap", 0.3)))

        outcome_reward = self._compute_outcome_reward(stdout, original_reward, chat_history)
        components = self._compute_components(stdout, original_reward, chat_history, trajectory_context)

        # G2 clamp: sum the behavioral component deltas, clamp to +/- cap.
        raw_shaping = sum(components.values())
        shaping_total = max(-total_shaping_cap, min(total_shaping_cap, raw_shaping))

        final_reward = outcome_weight * outcome_reward + shaping_total

        # G2 ground-truth anchor: a failing trajectory (no outcome credit) can
        # never be moved net-positive by shaping alone. When the outcome
        # contribution is <= 0, shaping may not push the total above 0.
        outcome_contribution = outcome_weight * outcome_reward
        if outcome_contribution <= 0.0 and final_reward > 0.0:
            final_reward = min(final_reward, outcome_contribution)

        reward_components: Dict[str, float] = {"outcome": outcome_reward}
        reward_components.update(components)
        reward_components["shaping_total"] = shaping_total

        return final_reward, reward_components

    def shape(
        self,
        stdout: Optional[str],
        original_reward: float,
        chat_history: Optional[List[Dict[str, Any]]] = None,
        trajectory_context: Optional[Dict[str, Any]] = None,
    ) -> float:
        final_reward, _ = self.shape_with_components(stdout, original_reward, chat_history, trajectory_context)
        return final_reward


# =============================================================================
# Trajectory Shaper Registry
# =============================================================================

_TRAJECTORY_SHAPER_REGISTRY: Dict[str, Type[TrajectoryRewardShaper]] = {}


def register_trajectory_shaper(shaper_cls: Type[TrajectoryRewardShaper]) -> Type[TrajectoryRewardShaper]:
    """Register a trajectory-based shaper class."""
    _TRAJECTORY_SHAPER_REGISTRY[shaper_cls.name()] = shaper_cls
    return shaper_cls


register_trajectory_shaper(ThinkingLengthShaper)
register_trajectory_shaper(FormatQualityShaper)
register_trajectory_shaper(CommandQualityShaper)


def get_trajectory_shaper(name: str, **kwargs) -> TrajectoryRewardShaper:
    """Get a trajectory-based reward shaper by name."""
    if name not in _TRAJECTORY_SHAPER_REGISTRY:
        available = ", ".join(_TRAJECTORY_SHAPER_REGISTRY.keys())
        raise ValueError(f"Unknown trajectory shaper '{name}'. Available: {available}")
    return _TRAJECTORY_SHAPER_REGISTRY[name](**kwargs)


def list_trajectory_shapers() -> List[str]:
    """List all registered trajectory shaper names."""
    return list(_TRAJECTORY_SHAPER_REGISTRY.keys())


# =============================================================================
# Registry
# =============================================================================

# Parser registry
_PARSER_REGISTRY: Dict[str, Type[OutputParser]] = {}

# Shaper registry
_SHAPER_REGISTRY: Dict[str, Type[RewardShaper]] = {}


def register_parser(parser_cls: Type[OutputParser]) -> Type[OutputParser]:
    """Register a parser class in the registry."""
    _PARSER_REGISTRY[parser_cls.name()] = parser_cls
    return parser_cls


def register_shaper(shaper_cls: Type[RewardShaper]) -> Type[RewardShaper]:
    """Register a shaper class in the registry."""
    _SHAPER_REGISTRY[shaper_cls.name()] = shaper_cls
    return shaper_cls


# Register built-in parsers
register_parser(PytestOutputParser)
register_parser(UnittestOutputParser)
register_parser(JestOutputParser)
register_parser(PassRatioSummaryOutputParser)
register_parser(GenericOutputParser)

# Register built-in shapers
register_shaper(PassRatioShaper)
register_shaper(EffectivePassRatioShaper)
register_shaper(WeightedShaper)
register_shaper(ThresholdShaper)
register_shaper(BinaryWithPartialCreditShaper)
register_shaper(OriginalRewardShaper)


def get_output_parser(name: str) -> OutputParser:
    """
    Get an output parser by name.

    Args:
        name: Parser name ("pytest", "unittest", "generic")

    Returns:
        Instantiated OutputParser

    Raises:
        ValueError: If parser name not found
    """
    if name not in _PARSER_REGISTRY:
        available = ", ".join(_PARSER_REGISTRY.keys())
        raise ValueError(f"Unknown parser '{name}'. Available: {available}")
    return _PARSER_REGISTRY[name]()


def get_reward_shaper(name: str, **kwargs) -> RewardShaper:
    """
    Get a reward shaper by name.

    Args:
        name: Shaper name ("pass_ratio", "effective_pass_ratio", "weighted", etc.)
        **kwargs: Additional arguments passed to shaper constructor

    Returns:
        Instantiated RewardShaper

    Raises:
        ValueError: If shaper name not found
    """
    if name not in _SHAPER_REGISTRY:
        available = ", ".join(_SHAPER_REGISTRY.keys())
        raise ValueError(f"Unknown shaper '{name}'. Available: {available}")
    return _SHAPER_REGISTRY[name](**kwargs)


def list_parsers() -> List[str]:
    """List all registered parser names."""
    return list(_PARSER_REGISTRY.keys())


def list_shapers() -> List[str]:
    """List all registered shaper names."""
    return list(_SHAPER_REGISTRY.keys())


# =============================================================================
# Convenience Functions
# =============================================================================


def auto_detect_parser(output: str) -> Optional[OutputParser]:
    """
    Auto-detect the appropriate parser for the given output.

    Tries parsers in order of specificity (pytest, unittest, generic).

    Args:
        output: Test output string

    Returns:
        Appropriate OutputParser or None if no parser matches
    """
    # Try in order of specificity
    parser_order = ["pytest", "unittest", "pass_ratio_summary", "generic"]

    for parser_name in parser_order:
        parser = get_output_parser(parser_name)
        if parser.can_parse(output):
            return parser

    return None


def parse_test_output(
    output: str,
    parser_name: Optional[str] = None,
) -> Optional[ParsedTestResult]:
    """
    Parse test output using specified or auto-detected parser.

    Args:
        output: Test output string
        parser_name: Parser to use, or None for auto-detection

    Returns:
        ParsedTestResult or None if parsing failed
    """
    parsed, _ = parse_test_output_with_parser(output, parser_name)
    return parsed


def parse_test_output_with_parser(
    output: str,
    parser_name: Optional[str] = None,
) -> tuple[Optional[ParsedTestResult], Optional[str]]:
    """Parse test output and report which parser handled it."""
    if parser_name:
        parser = get_output_parser(parser_name)
    else:
        parser = auto_detect_parser(output)
        if parser is None:
            return None, None
    parsed = parser.parse(output)
    if parsed is not None:
        parsed = _with_identified_tests(parsed, _identified_tests(output, parser.name()))
    return parsed, parser.name()


def shape_reward_from_output(
    stdout: Optional[str],
    original_reward: float,
    parser_name: Optional[str] = None,
    shaper_name: str = "pass_ratio",
    shaper_kwargs: Optional[Dict] = None,
    fallback_to_original: bool = True,
    chat_history: Optional[List[Dict[str, Any]]] = None,
) -> float:
    """
    Parse test output and compute shaped reward in one call.

    This is the main entry point for reward shaping. Supports three modes:

    1. Verifier-based shapers (pass_ratio, threshold, etc.) — use test output
    2. Trajectory-based shapers (thinking_length, format_quality) — use chat history
    3. Composite shaper — weighted combination of verifier + trajectory signals

    Args:
        stdout: Test output string (verifier stdout)
        original_reward: Original reward from verifier
        parser_name: Parser to use (None for auto-detection)
        shaper_name: Shaper to use (default: "pass_ratio")
        shaper_kwargs: Additional kwargs for shaper
        fallback_to_original: Return original_reward on parse failure when True; raise ValueError when False
        chat_history: Agent conversation trajectory (for trajectory-based shapers)

    Returns:
        Shaped reward value in [0, 1]

    Raises:
        ValueError: The verifier output cannot be parsed and fallback_to_original is False.
    """
    kwargs = shaper_kwargs or {}

    # Handle composite shaper separately — it combines verifier + trajectory
    if shaper_name == "composite":
        parsed = parse_test_output(stdout, parser_name) if stdout else None
        composite = CompositeShaper(**kwargs)
        shaped, component_rewards = composite.shape_with_components(parsed, original_reward, chat_history)
        logger.debug(f"Composite shaped reward: {original_reward:.3f} -> {shaped:.3f} (components={component_rewards})")
        return shaped

    # Handle trajectory-based shapers
    if shaper_name in _TRAJECTORY_SHAPER_REGISTRY:
        shaper = get_trajectory_shaper(shaper_name, **kwargs)
        shaped = shaper.shape(chat_history or [], original_reward)
        logger.debug(f"Trajectory shaped reward: {original_reward:.3f} -> {shaped:.3f} (shaper={shaper_name})")
        return shaped

    # Verifier-based shapers — need test output
    if not stdout:
        if fallback_to_original:
            return original_reward
        raise RewardOutputParseError("could not parse verifier output: stdout is empty")

    # Parse output
    parsed = parse_test_output(stdout, parser_name)

    if parsed is None:
        if fallback_to_original:
            logger.debug(
                f"Could not parse test output with parser={parser_name or 'auto'}. "
                f"Falling back to original reward: {original_reward}"
            )
            return original_reward
        raise RewardOutputParseError(f"could not parse verifier output with parser={parser_name or 'auto'}")

    # Log parse results
    logger.debug(
        f"Parsed test results: passed={parsed.passed}, failed={parsed.failed}, "
        f"errors={parsed.errors}, total={parsed.total}, "
        f"effective_pass_ratio={parsed.effective_pass_ratio:.3f}"
    )

    # Shape reward
    shaper = get_reward_shaper(shaper_name, **kwargs)
    shaped = shaper.shape(parsed, original_reward)

    logger.debug(f"Shaped reward: {original_reward:.3f} -> {shaped:.3f} (shaper={shaper_name})")

    return shaped


def shape_reward_with_components(
    stdout: Optional[str],
    original_reward: float,
    parser_name: Optional[str] = None,
    shaper_kwargs: Optional[Dict] = None,
    chat_history: Optional[List[Dict[str, Any]]] = None,
    shaper_name: str = "composite",
    trajectory_context: Optional[Dict[str, Any]] = None,
) -> Tuple[float, Dict[str, float]]:
    """
    Component-returning variant for container shapers.

    Dispatches on ``shaper_name``:
      - ``"composite"``       -> the weighted-average trajectory composite shaper.
      - ``"composite_loop"``  -> the loop-behavior container shaper (Stage 0+):
        outcome term + bounded behavioral components, with the G2 clamp.

    ``trajectory_context`` (Stage 1+) carries finished-trajectory signals
    (``mark_complete`` / ``verifier_reward`` / ``premature_stop``) consumed by the
    ``composite_loop`` terminate component. It is ignored by the legacy
    ``composite`` shaper. When None, ``composite_loop`` is byte-identical to
    Stage 0 (G1).

    Returns:
        Tuple of (final_reward, component_rewards) where component_rewards maps
        each component name to its (raw) reward value. The breakdown shape
        depends on the container shaper.
    """
    kwargs = shaper_kwargs or {}

    if shaper_name == "composite_loop":
        shaper = CompositeLoopShaper(**kwargs)
        return shaper.shape_with_components(stdout, original_reward, chat_history, trajectory_context)

    # Default: weighted-average composite shaper.
    parsed = parse_test_output(stdout, parser_name) if stdout else None
    composite = CompositeShaper(**kwargs)
    return composite.shape_with_components(parsed, original_reward, chat_history)
