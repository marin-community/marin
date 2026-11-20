# Log parsers and test-name matching shared by the standalone Harbor graders.
# task_files.grader_source() copies the definitions that a parser needs into tests/grade.py.
from __future__ import annotations

from bisect import bisect_left
from enum import Enum
import contextlib
import fnmatch
import re
import xml.etree.ElementTree as ET


class TestStatus(str, Enum):
    FAILED = "FAILED"
    PASSED = "PASSED"
    SKIPPED = "SKIPPED"
    ERROR = "ERROR"

    def __str__(self):
        return self.value


def ansi_escape(text: str) -> str:
    """
    Remove ANSI escape sequences from text
    """
    return re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~]|[()][A-Z0-9])").sub(
        "", text
    )


def parse_log_maven(log: str) -> ParsedTestLog:
    """
    Parser for test logs generated with 'mvn test'.
    Annoyingly maven will not print the tests that have succeeded. For this log
    parser to work, each test must be run individually, and then we look for
    BUILD (SUCCESS|FAILURE) in the logs.

    Args:
        log (str): log content
    Returns:
        dict: test case to test status mapping
    """
    test_status_map: ParsedTestLog = {}
    current_test_name = "---NO TEST NAME FOUND YET---"

    test_name_pattern = r"^.*-Dtest=(\S+).*$"
    running_pattern = r"^\[INFO\]\s+Running\s+(\S+)$"
    result_pattern = r"^.*BUILD (SUCCESS|FAILURE)$"

    for line in log.split("\n"):
        test_name_match = re.match(test_name_pattern, line.strip())
        if test_name_match:
            current_test_name = test_name_match.groups()[0]
            continue

        running_match = re.match(running_pattern, line.strip())
        if running_match:
            current_test_name = running_match.groups()[0]
            continue

        result_match = re.match(result_pattern, line.strip())
        if result_match:
            status = result_match.groups()[0]
            if status == "SUCCESS":
                test_status_map[current_test_name] = TestStatus.PASSED
            elif status == "FAILURE":
                test_status_map[current_test_name] = TestStatus.FAILED

    return test_status_map


def _mvn_failure_status(line: str) -> TestStatus:
    if "Exception" in line and "AssertionError" not in line:
        return TestStatus.ERROR
    return TestStatus.FAILED


def _mvn_summary_class(line: str) -> str | None:
    if " in " in line:
        return line.rsplit(" in ", 1)[-1].strip().rstrip(".:;")
    return None


def _mvn_status_from_summary_counts(
    tests_run: int, failures: int, errors: int, skipped: int
) -> TestStatus:
    if failures == 0 and errors == 0:
        if tests_run > skipped or skipped == 0:
            return TestStatus.PASSED
        return TestStatus.SKIPPED
    return TestStatus.FAILED


def _record_mvn_gradle_test_status(
    results: ParsedTestLog, current_gradle_class: str | None, match: re.Match[str]
) -> None:
    if not current_gradle_class:
        return
    method, status_word = match.groups()
    status = TestStatus.PASSED if status_word == "PASSED" else TestStatus.FAILED
    results[f"{current_gradle_class} {method}"] = status
    if status is TestStatus.FAILED:
        results[current_gradle_class] = status
    else:
        results.setdefault(current_gradle_class, status)


def _record_mvn_failure(
    results: ParsedTestLog, line: str, match: re.Match[str]
) -> None:
    clazz = match.group("class")
    method = match.group("method")
    test_id = f"{clazz}.{method}"
    results[test_id] = _mvn_failure_status(line)
    results[clazz] = TestStatus.FAILED


def _record_mvn_summary(
    results: ParsedTestLog,
    line: str,
    match: re.Match[str],
    current_running: str | None,
) -> None:
    tests_run, failures, errors, skipped = map(int, match.groups())
    clazz = _mvn_summary_class(line)
    result_name = clazz or current_running
    if result_name:
        status = _mvn_status_from_summary_counts(tests_run, failures, errors, skipped)
        if status is TestStatus.FAILED:
            results[result_name] = status
        else:
            results.setdefault(result_name, status)


def _nested_statuses(
    names: list[str], results: ParsedTestLog, parent: str
) -> set[TestStatus]:
    prefix = f"{parent}$"
    statuses: set[TestStatus] = set()
    index = bisect_left(names, prefix)
    while index < len(names) and names[index].startswith(prefix):
        statuses.add(results[names[index]])
        index += 1
    return statuses


def _promote_nested_class_parents(results: ParsedTestLog) -> None:
    names = sorted(results)
    promoted: dict[str, TestStatus] = {}
    for parent in names:
        if " " in parent:
            continue
        statuses = _nested_statuses(names, results, parent)
        if statuses & {TestStatus.FAILED, TestStatus.ERROR}:
            promoted[parent] = TestStatus.FAILED
        elif results[parent] is TestStatus.SKIPPED and TestStatus.PASSED in statuses:
            promoted[parent] = TestStatus.PASSED
    results.update(promoted)


def parse_java_mvn(log: str) -> ParsedTestLog:
    """Parse Maven logs into {identifier: status}."""
    results: ParsedTestLog = {}

    failure_re = re.compile(
        r"^\[ERROR\]\s+(?P<class>[A-Za-z0-9_$.]+)\.(?P<method>[A-Za-z0-9_$.]+?)(?::(?P<line>\d+))?\b"
    )
    running_re = re.compile(r"^(?:\[INFO\]\s+)?Running\s+([A-Za-z0-9_$.]+)$")
    gradle_class_re = re.compile(r"^([A-Za-z0-9_$.]+(?:Test|Tests|IT|ITCase))$")
    gradle_test_status_re = re.compile(
        r"^Test\s+(.+?)\s+(PASSED|FAILED)(?:\s+\(\d+(?:\.\d+)?\s*m?s\))?$"
    )
    summary_re = re.compile(
        r"Tests run:\s*(\d+),\s*Failures:\s*(\d+),\s*Errors:\s*(\d+),"
        r"\s*Skipped:\s*(\d+)"
    )

    seen_running: set[str] = set()
    current_running: str | None = None
    current_gradle_class: str | None = None

    for raw in log.splitlines():
        line = raw.strip()
        if not line:
            continue
        if rm := running_re.match(line):
            running_name = rm.group(1)
            current_running = running_name
            current_gradle_class = running_name
            seen_running.add(running_name)
            continue
        if gm := gradle_class_re.match(line):
            current_gradle_class = gm.group(1)
            seen_running.add(current_gradle_class)
            continue
        if tm := gradle_test_status_re.match(line):
            _record_mvn_gradle_test_status(results, current_gradle_class, tm)
            continue
        if fm := failure_re.match(line):
            _record_mvn_failure(results, line, fm)
            continue

        if sm := summary_re.search(line):
            _record_mvn_summary(results, line, sm, current_running)
            current_running = None

    for test_name, status in parse_log_junit(log).items():
        results.setdefault(test_name, status)

    for test_name, status in parse_log_maven(log).items():
        results.setdefault(test_name, status)

    if not results and seen_running:
        for clazz in seen_running:
            results.setdefault(clazz, TestStatus.PASSED)

    _promote_nested_class_parents(results)
    return results


def _parse_junit_testcases_from_text(log: str) -> ParsedTestLog:
    results: ParsedTestLog = {}
    open_tag_re = re.compile(r"<testcase\b([^>]*)>", re.DOTALL)
    attr_re = re.compile(r'(\w+)="([^"]*)"')

    pos = 0
    while True:
        match = open_tag_re.search(log, pos)
        if match is None:
            break

        raw_attrs = match.group(1) or ""
        attrs = dict(attr_re.findall(raw_attrs))
        name = attrs.get("name")
        classname = attrs.get("classname", "")

        if not name:
            pos = match.end()
            continue

        is_self_closing = raw_attrs.strip().endswith("/")
        content = ""
        if is_self_closing:
            pos = match.end()
        else:
            close_pos = log.find("</testcase>", match.end())
            if close_pos != -1:
                content = log[match.end() : close_pos]
                pos = close_pos + len("</testcase>")
            else:
                content = log[match.end() : match.end() + 4096]
                pos = match.end()

        full_name = f"{classname} {name}".strip() if classname else name

        if "<failure" in content:
            status = TestStatus.FAILED
        elif "<error" in content:
            status = TestStatus.ERROR
        elif "<skipped" in content:
            status = TestStatus.SKIPPED
        else:
            status = TestStatus.PASSED

        results[full_name] = status

    return results


def parse_log_junit(log: str) -> ParsedTestLog:
    """Parse Scala sbt test output (JUnit XML format).

    Returns {full_test_name: status}.

    Rules:
      * Parse XML testsuite elements from the log
      * Extract testcase elements with their name and classname
      * Determine status: PASSED (no failure/error/skipped), FAILED, ERROR,
        or SKIPPED
    """
    return _parse_junit_testcases_from_text(log)


def parse_java_mvn_v2(log: str) -> ParsedTestLog:
    """Parse Maven Surefire / Failsafe style logs and extract module status."""
    results: ParsedTestLog = {}
    line_re = re.compile(
        r"^\[INFO\]\s+(?P<name>.+?)\s+\.\.+\s+(?P<status>SUCCESS|FAILURE|SKIPPED)(?:\s|\[|$)"
    )
    summary_re = re.compile(
        r"Tests run:\s*(\d+),\s*Failures:\s*(\d+),\s*Errors:\s*(\d+),"
        r"\s*Skipped:\s*(\d+)"
    )

    for raw in log.splitlines():
        line = raw.strip()
        if not line:
            continue
        if m := line_re.match(line):
            status_word = m.group("status")
            if status_word == "SUCCESS":
                status = TestStatus.PASSED
            elif status_word == "FAILURE":
                status = TestStatus.FAILED
            elif status_word == "SKIPPED":
                status = TestStatus.SKIPPED
            else:
                status = TestStatus.ERROR
            results[m.group("name")] = status
            continue

        if sm := summary_re.search(line):
            tests_run, failures, errors, skipped = map(int, sm.groups())
            aggregate_status = TestStatus.PASSED
            if failures > 0:
                aggregate_status = TestStatus.FAILED
            elif errors > 0:
                aggregate_status = TestStatus.ERROR
            elif skipped == tests_run:
                aggregate_status = TestStatus.SKIPPED
            results.setdefault("__suite__", aggregate_status)

    return results


def parse_log_ant(log: str) -> ParsedTestLog:
    test_status_map: ParsedTestLog = {}

    pattern = r"^\s*\[junit\]\s+\[(PASS|FAIL|ERR)\]\s+(.*)$"

    for line in log.split("\n"):
        match = re.match(pattern, line.strip())
        if match:
            status, test_name = match.groups()
            if status == "PASS":
                test_status_map[test_name] = TestStatus.PASSED
            elif status in ["FAIL", "ERR"]:
                test_status_map[test_name] = TestStatus.FAILED

    return test_status_map


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


CARGO_TEST_RE = re.compile(r"^test\s+(.+?)\s+\.\.\.\s*(.*)$")


NEXTEST_RE = re.compile(r"^\s*(PASS|FAIL)\s+\[[^\]]+\]\s+\S+\s+(.+?)\s*$")


CARGO_BUILD_FAILURE_RE = re.compile(
    r"^error: could not compile `(?P<crate>[^`]+)`"
    r"(?: \([^)]+\))? due to (?:\d+ )?previous errors?"
    r"(?:; \d+ warnings? emitted)?$"
)


def _cargo_outcome(output: str) -> TestStatus | None:
    output = output.strip()
    if output.startswith("ok") or output.endswith(" ok"):
        return TestStatus.PASSED
    if output.startswith("FAILED") or output.endswith(" FAILED"):
        return TestStatus.FAILED
    return None


def _pending_outcome(line: str) -> TestStatus | None:
    if line.startswith("ok"):
        return TestStatus.PASSED
    failed_suffix = line[len("FAILED") : len("FAILED") + 1]
    if line == "FAILED" or (line.startswith("FAILED") and not failed_suffix.isalnum()):
        return TestStatus.FAILED
    return None


def _parse_cargo_tests(
    line: str,
    test_status_map: ParsedTestLog,
    pending_test_names: list[str],
) -> bool:
    found_cargo_test = False
    start_index = 0
    while True:
        test_marker_index = line.find("test ", start_index)
        if test_marker_index < 0:
            return found_cargo_test
        match = CARGO_TEST_RE.match(line[test_marker_index:])
        if match:
            found_cargo_test = True
            test_name, output = match.groups()
            if "test " in test_name:
                test_name = test_name.rsplit("test ", 1)[1]
            outcome = _cargo_outcome(output)
            if outcome is not None:
                test_status_map[test_name] = outcome
            else:
                pending_test_names.append(test_name)
        start_index = test_marker_index + len("test ")


def parse_log_cargo(log: str) -> ParsedTestLog:
    """
    Args:
        log (str): log content
    Returns:
        dict: test case to test status mapping
    """
    test_status_map = {}
    pending_test_names = []

    for raw_line in log.split("\n"):
        line = ANSI_RE.sub("", raw_line).strip()
        build_failure = CARGO_BUILD_FAILURE_RE.match(line)
        if build_failure:
            test_status_map[f"{build_failure.group('crate')}::<build>"] = (
                TestStatus.ERROR
            )
            continue
        pending_outcome = _pending_outcome(line)
        if pending_outcome is not None and pending_test_names:
            test_name = pending_test_names.pop()
            test_status_map[test_name] = pending_outcome
            continue

        if _parse_cargo_tests(line, test_status_map, pending_test_names):
            continue

        match = NEXTEST_RE.match(line)
        if match:
            outcome, test_name = match.groups()
            if outcome == "PASS":
                test_status_map[test_name] = TestStatus.PASSED
            elif outcome == "FAIL":
                test_status_map[test_name] = TestStatus.FAILED

    return test_status_map


def parse_log_cpp_v3(log: str) -> ParsedTestLog:
    """Parse ``... OK`` / ``... FAILED`` style C++ test output."""
    results: ParsedTestLog = {}
    line_re = re.compile(r"^(?:\[[0-9]+/[0-9]+\]\s*)?(.*?)\s*\.\.\.\s*(\w+)$")
    failed_re = re.compile(r"\bFAILED\b|\bFAIL\b|\bFailure\b", re.IGNORECASE)
    skipped_re = re.compile(r"\bSKIPPED\b", re.IGNORECASE)
    passed_token_re = re.compile(r"\bOK\b|\bPASSED\b", re.IGNORECASE)

    for raw in log.splitlines():
        line = raw.strip()
        if not line:
            continue
        if m := line_re.match(line):
            name = m.group(1).strip()
            status_token = m.group(2).upper()
            if skipped_re.search(status_token) or skipped_re.search(line):
                results[name] = TestStatus.SKIPPED
            elif failed_re.search(status_token) or failed_re.search(line):
                results[name] = TestStatus.FAILED
            elif passed_token_re.search(status_token) or passed_token_re.search(line):
                results[name] = TestStatus.PASSED
            else:
                results[name] = TestStatus.ERROR
            continue

        if skipped_re.search(line):
            results[line] = TestStatus.SKIPPED
            continue
        if failed_re.search(line):
            results[line] = TestStatus.FAILED
            continue
        if passed_token_re.search(line):
            results[line] = TestStatus.PASSED

    return results


def parse_log_csharp(log: str) -> ParsedTestLog:
    """Parse xUnit.net output."""
    results: ParsedTestLog = {}
    result_re = re.compile(r"\b(Passed|Failed|Skipped)\s+(.+?)\s+\[[^\]]+?\]")
    loose_passed_re = re.compile(
        r"(?=(?<![A-Za-z0-9_.])([A-Za-z_]\w*(?:\.[A-Za-z_]\w*){2,}[^\[\]\n]*?)"
        r"\s+\[[^\]]+?\])"
    )
    joined_passed_re = re.compile(
        r"(?<=[a-z])([A-Z][A-Za-z_]\w*(?:\.[A-Za-z_]\w*){2,}[^\[\]\n]*?)"
        r"\s+\[[^\]]+?\]"
    )
    greedy_passed_re = re.compile(r"\bPassed\s+(.+)\s+\[[^\]]+?\]")
    embedded_joined_result_re = re.compile(
        r"(?<=[a-z])[A-Z][A-Za-z_]\w*(?:\.[A-Za-z_]\w*){2,}[^\[\]\n]*?"
        r"\s+\[[^\]]+?\]"
    )
    skipped_re = re.compile(r"^\s+Skipped\s+(.+?)(?:\s+\[.+?\])?$")
    xunit_fail_re = re.compile(r"^\[xUnit\.net\s+[\d:\.]+\]\s+(.+?)\s+\[FAIL\]$")
    xunit_finished_re = re.compile(r"^\[xUnit\.net\s+[\d:\.]+\]\s+Finished:\s+")
    dotnet_success_prefix_re = re.compile(r"^(?:Test Run )?Successful\.")
    duplicated_assembly_prefix_re = re.compile(r"^([A-Za-z_]\w*)\.Tests(?=\1\.)")

    def clean_name(name: str) -> str:
        name = re.sub(r"^(?:Passed|Failed|Skipped|Usage:)\s*", "", name).strip()
        name = xunit_finished_re.sub("", name)
        name = dotnet_success_prefix_re.sub("", name)
        name = embedded_joined_result_re.sub("", name)
        return duplicated_assembly_prefix_re.sub("", name)

    for raw in log.splitlines():
        line = raw.rstrip()
        if not line:
            continue
        if m := xunit_fail_re.match(line):
            results[clean_name(m.group(1))] = TestStatus.FAILED
            continue
        if m := skipped_re.match(line):
            results[clean_name(m.group(1))] = TestStatus.SKIPPED
            continue
        for m in result_re.finditer(line):
            status = m.group(1)
            name = clean_name(m.group(2))
            if status == "Failed":
                results[name] = TestStatus.FAILED
            elif status == "Skipped":
                results[name] = TestStatus.SKIPPED
            else:
                results.setdefault(name, TestStatus.PASSED)
        for m in loose_passed_re.finditer(line):
            name = clean_name(m.group(1))
            results.setdefault(name, TestStatus.PASSED)
        for m in joined_passed_re.finditer(line):
            name = clean_name(m.group(1))
            results.setdefault(name, TestStatus.PASSED)
        for m in greedy_passed_re.finditer(line):
            name = clean_name(m.group(1))
            results.setdefault(name, TestStatus.PASSED)

    return results


def _iter_json_objects(log: str) -> Iterator[dict[str, Any]]:
    for line in log.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except (ValueError, RecursionError):
            continue
        if isinstance(event, dict):
            yield event


def _dart_test_start(event: dict[str, Any]) -> tuple[int, str] | None:
    test_info = event.get("test")
    if not isinstance(test_info, dict):
        return None
    test_id = test_info.get("id")
    test_name = test_info.get("name")
    if (
        type(test_id) is int
        and isinstance(test_name, str)
        and test_name
        and not test_name.startswith("loading ")
    ):
        return test_id, test_name
    return None


def _dart_done_status(event: dict[str, Any]) -> TestStatus | None:
    result = event.get("result")
    if result == "success":
        return TestStatus.PASSED
    if result == "failure":
        return TestStatus.FAILED
    if result == "error":
        return TestStatus.ERROR
    if event.get("skipped"):
        return TestStatus.SKIPPED
    return None


def parse_log_dart(log: str) -> ParsedTestLog:
    """Parse Dart test output and return {test_name: status}.

    Dart test output is a stream of JSON objects with events.
    - testStart events have test.id and test.name
    - testDone events have testID and result (success, failure, error, or skipped)

    We map test IDs to names, then to their final result status.
    """
    results: ParsedTestLog = {}
    test_id_to_name: dict[int, str] = {}
    errored: set[int] = set()

    for event in _iter_json_objects(log):
        event_type = event.get("type")
        if event_type == "testStart":
            if started := _dart_test_start(event):
                test_id_to_name[started[0]] = started[1]
            continue

        test_id = event.get("testID")
        if type(test_id) is not int or test_id not in test_id_to_name:
            continue
        test_name = test_id_to_name[test_id]
        if event_type == "error":
            failed = event.get("isFailure", True)
            results[test_name] = TestStatus.FAILED if failed else TestStatus.ERROR
            errored.add(test_id)
        elif event_type == "testDone" and not event.get("hidden", False):
            status = _dart_done_status(event)
            if status is not None and test_id not in errored:
                results[test_name] = status

    return results


def parse_log_dart_v3(log: str) -> ParsedTestLog:
    """Parse Dart/Flutter test output with ``[E]`` markers."""
    results: ParsedTestLog = {}
    test_line_re = re.compile(
        r"^\d+:\d+\s+\+\d+(?:\s+-\d+)?:\s+(/[^:]+\.dart):\s+(.+?)(?:\s+\[E\])?$"
    )
    loading_error_re = re.compile(
        r"^\d+:\d+\s+\+\d+\s+-\d+:\s+loading\s+(/[^:]+\.dart)\s+\[E\]$"
    )

    for raw in log.splitlines():
        line = raw.strip()
        if not line:
            continue
        if m := loading_error_re.match(line):
            results[m.group(1)] = TestStatus.ERROR
            continue
        if m := test_line_re.match(line):
            full_test_name = f"{m.group(1)}: {m.group(2).strip()}"
            if "[E]" in line:
                results[full_test_name] = TestStatus.FAILED
            elif " -" in line.split(":")[0]:
                results.setdefault(full_test_name, TestStatus.FAILED)
            else:
                results.setdefault(full_test_name, TestStatus.PASSED)

    for test_name, status in parse_log_dart(log).items():
        results.setdefault(test_name, status)
    return results


def parse_log_elixir(log: str) -> ParsedTestLog:
    """Parse ExUnit output and return {full_test_name: status}.

    Rules:
      * Lines like: "* test <name> [L#42]" or with timing "(12.3ms)"
        -> PASSED (tentative)
      * Lines like: "* test <name> (skipped) [L#42]" -> SKIPPED
      * Failure headers: "1) test <name> (<Module>)"
        -> FAILED (overrides prior PASS)
    """
    results: ParsedTestLog = {}

    # Regexes
    skipped_re = re.compile(r"^\*\s+test\s+(.*?)\s+\(skipped\)\s+\[L#\d+\]$")
    passed_timed_re = re.compile(
        r"^\*\s+test\s+(.*?)\s+\([0-9]+(?:\.[0-9]+)?ms\)\s+\[L#\d+\]$"
    )
    passed_basic_re = re.compile(r"^\*\s+test\s+(.*?)\s+\[L#\d+\]$")
    failure_header_re = re.compile(r"^\d+\)\s+test\s+(.*?)\s+\([^)]+\)$")

    for raw in log.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.count("* test ") > 1:
            line = "* test " + line.rsplit("* test ", 1)[1]
        if m := skipped_re.match(line):
            results[m.group(1)] = TestStatus.SKIPPED
            continue
        if m := failure_header_re.match(line):
            results[m.group(1)] = TestStatus.FAILED
            continue
        if m := passed_timed_re.match(line):
            results.setdefault(m.group(1), TestStatus.PASSED)
            continue
        if m := passed_basic_re.match(line):
            results.setdefault(m.group(1), TestStatus.PASSED)
            continue
    return results


GO_TEST_RE = re.compile(r"^--- (PASS|FAIL|SKIP): (.+) \((.+)\)$")


GO_TEST_STATUSES = {
    "PASS": TestStatus.PASSED,
    "FAIL": TestStatus.FAILED,
    "SKIP": TestStatus.SKIPPED,
}


GO_JSON_STATUSES = {
    "pass": TestStatus.PASSED,
    "fail": TestStatus.FAILED,
    "skip": TestStatus.SKIPPED,
}


def _parse_go_json_event(event: dict[str, object]) -> tuple[str, TestStatus] | None:
    action = event.get("Action")
    test_name = event.get("Test")
    if isinstance(test_name, str) and test_name:
        package = event.get("Package")
        if isinstance(package, str) and package:
            test_name = f"{package}::{test_name}"
        status = GO_JSON_STATUSES.get(action) if isinstance(action, str) else None
        return (test_name, status) if status is not None else None

    package = event.get("Package")
    failed_build = event.get("FailedBuild")
    if (
        action == "fail"
        and isinstance(package, str)
        and package.strip()
        and isinstance(failed_build, str)
        and failed_build.strip()
    ):
        return f"{package.strip()}::<build>", TestStatus.ERROR
    return None


def parse_log_gotest(log: str) -> ParsedTestLog:
    """
    Parser for test logs generated with 'go test'

    Args:
        log (str): log content
        test_spec (TestSpec): test spec (unused)
    Returns:
        dict: test case to test status mapping
    """
    test_status_map = {}

    for line in log.split("\n"):
        stripped = line.strip()
        match = GO_TEST_RE.match(stripped)
        if match:
            status, test_name, _duration = match.groups()
            test_status_map[test_name] = GO_TEST_STATUSES[status]
            continue

        if not stripped.startswith("{"):
            continue
        try:
            event = json.loads(stripped)
        except json.JSONDecodeError:
            continue

        result = _parse_go_json_event(event)
        if result is not None:
            test_name, status = result
            test_status_map[test_name] = status

    return test_status_map


JAVA_OPTIONS_RE = re.compile(r"\s*Picked up _JAVA_OPTIONS:.*")


GRADLE_CUSTOM_STATUS_RE = re.compile(
    r"^(.+)\s+(PASSED|FAILED)(?:$|\s|[a-z]|[^A-Za-z0-9_])"
)


GRADLE_CUSTOM_PARTIAL_STATUS_RE = re.compile(
    r"^(.+?)\s+(P|PA|PAS|PASS|PASSE|F|FA|FAI|FAIL|FAILE)$"
)


GRADLE_BUILD_SUMMARY_RE = re.compile(r"^BUILD (SUCCESSFUL|FAILED)(?:$|\s)")


def _record_gradle_custom_status(
    test_status_map: ParsedTestLog, test_name: str, status: str
) -> bool:
    if status == "PASSED":
        test_status_map[test_name] = TestStatus.PASSED
        return True
    if status == "FAILED":
        test_status_map[test_name] = TestStatus.FAILED
        return True
    return False


def _record_gradle_custom_status_line(
    test_status_map: ParsedTestLog, line: str
) -> bool:
    if not (match := GRADLE_CUSTOM_STATUS_RE.match(line)):
        return False
    test_name, status = match.groups()
    return _record_gradle_custom_status(test_status_map, test_name, status)


def _consume_gradle_custom_status_prefix(
    test_status_map: ParsedTestLog,
    pending_status_prefix: tuple[str, str] | None,
    line: str,
) -> tuple[tuple[str, str] | None, bool]:
    if pending_status_prefix is None:
        return None, False

    pending_name, status_prefix = pending_status_prefix
    consumed = _record_gradle_custom_status(
        test_status_map, pending_name, f"{status_prefix}{line}"
    )
    return None, consumed


def _consume_gradle_custom_test_name(
    test_status_map: ParsedTestLog, pending_test_name: str | None, line: str
) -> tuple[str | None, bool]:
    if not pending_test_name or line not in {"PASSED", "FAILED"}:
        return pending_test_name, False

    _record_gradle_custom_status(test_status_map, pending_test_name, line)
    return None, True


def _gradle_custom_status_prefix(
    line: str, had_java_options: bool
) -> tuple[str, str] | None:
    if not had_java_options:
        return None
    if partial_match := GRADLE_CUSTOM_PARTIAL_STATUS_RE.match(line):
        test_name, status_prefix = partial_match.groups()
        return test_name, status_prefix
    return None


def parse_log_gradle_custom(log: str) -> ParsedTestLog:
    """
    Parser for test logs generated with 'gradle test'. Assumes that the
    pre-install script to update the gradle config has run.
    """
    test_status_map: ParsedTestLog = {}
    pending_test_name: str | None = None
    pending_status_prefix: tuple[str, str] | None = None
    pending_name_prefix: str | None = None

    for raw_line in log.split("\n"):
        raw_line = raw_line.strip()
        had_java_options = "Picked up _JAVA_OPTIONS:" in raw_line
        java_options_prefix = raw_line.split("Picked up _JAVA_OPTIONS:", 1)[0]
        line = JAVA_OPTIONS_RE.sub("", raw_line).strip()
        if not line or line.startswith(">"):
            continue
        if GRADLE_BUILD_SUMMARY_RE.match(line):
            continue

        if pending_name_prefix is not None:
            _record_gradle_custom_status_line(
                test_status_map, f"{pending_name_prefix}{line}"
            )
            pending_name_prefix = None

        pending_status_prefix, consumed = _consume_gradle_custom_status_prefix(
            test_status_map, pending_status_prefix, line
        )
        if consumed:
            continue

        pending_test_name, consumed = _consume_gradle_custom_test_name(
            test_status_map, pending_test_name, line
        )
        if consumed:
            continue

        if _record_gradle_custom_status_line(test_status_map, line):
            pending_test_name = None
            continue

        if pending_status_prefix := _gradle_custom_status_prefix(
            line, had_java_options
        ):
            continue

        if had_java_options:
            if ">" in line:
                pending_test_name = line
            pending_name_prefix = java_options_prefix

    return test_status_map


def parse_log_gradlew_v1(log: str) -> ParsedTestLog:
    """Parse Gradle JUnit XML output and return {full_test_name: status}.

    Parses XML testsuite elements and extracts testcase results.
    Test name format: "testname (classname)" or just "testname" if no class.

    Status mapping:
      * <testcase> with no <failure>, <error>, or <skipped> -> PASSED
      * <testcase> with <skipped> -> SKIPPED
      * <testcase> with <failure> -> FAILED
      * <testcase> with <error> -> ERROR
    """
    results: ParsedTestLog = {}

    # Find all XML blocks in the log
    lines = log.split("\n")
    xml_blocks = []
    current_block = []
    in_xml = False

    for line in lines:
        if line.strip().startswith("<?xml"):
            in_xml = True
            current_block = [line]
        elif in_xml:
            current_block.append(line)
            if line.strip().startswith("</testsuite>"):
                xml_blocks.append("\n".join(current_block))
                current_block = []
                in_xml = False

    # Parse each XML block
    for xml_block in xml_blocks:
        try:
            root = ET.fromstring(xml_block)

            # Process each testcase
            for testcase in root.findall(".//testcase"):
                name = testcase.get("name", "")
                classname = testcase.get("classname", "")

                # Format: "testname (classname)" to match common test output format
                full_name = f"{name} ({classname})" if classname and name else name

                # Determine status
                if testcase.find("skipped") is not None:
                    status = TestStatus.SKIPPED
                elif testcase.find("failure") is not None:
                    status = TestStatus.FAILED
                elif testcase.find("error") is not None:
                    status = TestStatus.ERROR
                else:
                    status = TestStatus.PASSED

                results[full_name] = status

        except ET.ParseError:
            # Skip malformed XML blocks
            continue

    return results


def parse_log_jq(log: str) -> ParsedTestLog:
    """
    Args:
        log (str): log content
    Returns:
        dict: test case to test status mapping
    """
    test_status_map = {}

    pattern = r"^\s*(PASS|FAIL):\s(.+)$"

    for line in log.split("\n"):
        match = re.match(pattern, line.strip())
        if match:
            status, test_name = match.groups()
            if status == "PASS":
                test_status_map[test_name] = TestStatus.PASSED
            elif status == "FAIL":
                test_status_map[test_name] = TestStatus.FAILED
    return test_status_map


_JS_DURATION_SUFFIX_RE = re.compile(
    r"(?:\s+\d+(?:\.\d+)?\s*(?:ms|s)|\s*\(\s*\d+(?:\.\d+)?\s*(?:ms|s)\s*\))\s*$",
    re.IGNORECASE,
)


def _strip_js_duration_suffix(name: str) -> str:
    """Remove trailing timing information such as ``(53ms)`` from a test name."""

    return _JS_DURATION_SUFFIX_RE.sub("", name).strip()


def _is_js_diff_marker_name(name: str) -> bool:
    return name.strip().lower() in {"true", "false", "null", "undefined"}


def parse_log_js(log: str) -> ParsedTestLog:
    """Parse Mocha/Jest output and return {full_test_name: status}.

    Rules:
      * Lines like: "    ✔ should send results to beats" -> PASSED
      * Lines like: "    - test-summary for single suite - teams" -> SKIPPED
      * Lines like: "    [W] 1) should mention group name in slack:" -> FAILED
    """
    results: ParsedTestLog = {}

    # Regexes
    passed_re = re.compile(r"^\s*(?:\[[A-Z]\]\s*)?[✔✓]\s+(.*?)$")
    skipped_re = re.compile(r"^\s*-\s+(.*?)$")
    failed_re = re.compile(r"^\s*\[W\]\s*\d+\)\s+(.*?)$")
    failed_header_re = re.compile(r"^\s*\d+\)\s+(.*?):$")

    for raw in log.splitlines():
        line = raw.strip()
        if not line:
            continue
        if m := skipped_re.match(line):
            name = _strip_js_duration_suffix(m.group(1))
            if name and not _is_js_diff_marker_name(name):
                results[name] = TestStatus.SKIPPED
            continue
        if m := failed_header_re.match(line):
            name = _strip_js_duration_suffix(m.group(1))
            if name:
                results[name] = TestStatus.FAILED
            continue
        if m := failed_re.match(line):
            name = _strip_js_duration_suffix(m.group(1).rstrip()).rstrip(":").rstrip()
            if name:
                results[name] = TestStatus.FAILED
            continue
        if m := passed_re.match(line):
            name = _strip_js_duration_suffix(m.group(1))
            if name:
                results.setdefault(name, TestStatus.PASSED)
            continue
    return results


def parse_log_js_2(log: str) -> ParsedTestLog:
    """Parse Mocha output and return {full_test_name: status}.

    Rules:
      * Lines like: "  ✔ <name>" -> PASSED (tentative)
      * Lines like: "  1) <name>" -> FAILED (overrides prior PASS)
      * Lines like: "  - <name>" -> SKIPPED (overrides prior PASS)
    """
    results: ParsedTestLog = {}

    # Regexes
    passed_re = re.compile(r"^\s*[✔✓]\s+(.*?)$")
    failed_re = re.compile(r"^\s*\d+\)\s+(.*?)$")
    skipped_re = re.compile(r"^\s*-\s+(.*?)$")

    for raw in log.splitlines():
        line = raw.strip()
        if not line:
            continue
        if m := skipped_re.match(line):
            name = _strip_js_duration_suffix(m.group(1))
            if name and not _is_js_diff_marker_name(name):
                results[name] = TestStatus.SKIPPED
            continue
        if m := failed_re.match(line):
            name = _strip_js_duration_suffix(m.group(1))
            if name:
                results[name] = TestStatus.FAILED
            continue
        if m := passed_re.match(line):
            name = _strip_js_duration_suffix(m.group(1))
            if name:
                results.setdefault(name, TestStatus.PASSED)
            continue
    return results


def parse_log_js_3(log: str) -> ParsedTestLog:
    """Parse TAP-formatted JS test output into {full_test_name: status}.

    The parser tracks nested suites by concatenating names with ``::`` so that
    leaf test names remain unique even when repeated across suites.
    """
    results: ParsedTestLog = {}
    stack: list[str] = []

    tap_line_re = re.compile(
        r"^(?P<status>not ok|ok)\s+\d+\s+-\s+(?P<name>.*?)(?:\s+#.*)?$"
    )

    for raw in log.splitlines():
        line = raw.strip()
        if not line:
            continue

        # Handle closing braces signalling the end of a nested block.
        if line.replace("}", "") == "":
            for _ in range(line.count("}")):
                if stack:
                    stack.pop()
            continue

        opens_context = line.endswith("{")
        if opens_context:
            line = line[:-1].rstrip()

        match = tap_line_re.match(line)
        if not match:
            continue

        status_word = match.group("status")
        raw_name = match.group("name")
        name, *_ = re.split(r"\s+#", raw_name, maxsplit=1)
        name = _strip_js_duration_suffix(name.strip())
        if not name:
            continue
        skip_marker = any(
            token in raw_name.lower() for token in ("# skip", "# skipped", "# todo")
        )

        full_name = " :: ".join((*stack, name)) if stack else name

        if skip_marker:
            results[full_name] = TestStatus.SKIPPED
        else:
            status_value = (
                TestStatus.PASSED if status_word == "ok" else TestStatus.FAILED
            )
            if status_value == TestStatus.FAILED or full_name not in results:
                results[full_name] = status_value

        if opens_context:
            stack.append(name)

    return results


def _strip_js_4_tag(payload: str) -> str:
    payload = payload.strip()
    if (payload.startswith("[") and "]: " in payload) or (
        payload.startswith("[") and "]:" in payload
    ):
        payload = payload.split("]:", 1)[1].strip()
    if payload.startswith(":"):
        payload = payload[1:].strip()
    return payload


def _strip_js_4_skip_markers(candidate: str, skip_markers: tuple[str, ...]) -> str:
    for marker in skip_markers:
        candidate = candidate.replace(marker, "")
    return candidate


def _record_js_4_result(
    results: ParsedTestLog,
    line: str,
    symbol: str,
    payload: str,
    pass_symbols: tuple[str, ...],
    fail_symbols: tuple[str, ...],
    skip_symbols: tuple[str, ...],
    skip_markers: tuple[str, ...],
) -> None:
    if symbol in pass_symbols:
        name = _strip_js_duration_suffix(_strip_js_4_tag(payload).strip())
        if name:
            results.setdefault(name, TestStatus.PASSED)
        return

    if symbol in fail_symbols:
        name = _strip_js_duration_suffix(_strip_js_4_tag(payload).strip())
        if name:
            results[name] = TestStatus.FAILED
        return

    if symbol in skip_symbols:
        candidate = _strip_js_4_skip_markers(_strip_js_4_tag(payload), skip_markers)
        name = _strip_js_duration_suffix(candidate.strip())
        if name:
            results[name] = TestStatus.SKIPPED
        return

    if any(marker in line.lower() for marker in skip_markers):
        candidate = _strip_js_4_skip_markers(line, skip_markers)
        name = _strip_js_duration_suffix(candidate.strip())
        if name:
            results[name] = TestStatus.SKIPPED


def parse_log_js_4(log: str) -> ParsedTestLog:
    """Parse JS test runner output and return {full_test_name: status}."""
    results: ParsedTestLog = {}
    mult_sign = chr(0x00D7)
    pass_symbols = ("✔", "✓")
    fail_symbols = ("✘", "✖", "✕", "✗", mult_sign)
    skip_symbols = ("○", "◌", "◦", "⚪")
    skip_markers = ("(skipped)", "[skip]", "[skipped]", "[pending]", "[todo]")

    status_symbols = pass_symbols + fail_symbols + skip_symbols
    status_re = re.compile(
        r"(?:^|\s)([" + re.escape("".join(status_symbols)) + r"])(?:\s+)(.+)$"
    )

    for raw in log.splitlines():
        line = raw.strip()
        if not line:
            continue

        match = None
        for candidate in status_re.finditer(line):
            match = candidate
        symbol = match.group(1) if match else line[0]
        payload = match.group(2).strip() if match else line[1:].strip()

        _record_js_4_result(
            results,
            line,
            symbol,
            payload,
            pass_symbols,
            fail_symbols,
            skip_symbols,
            skip_markers,
        )

    return results


JULIA_SUMMARY_COLUMNS = frozenset({"Pass", "Fail", "Error", "Broken", "Total"})


def _julia_header_columns(header: str) -> list[tuple[str, int]] | None:
    if "|" not in header:
        return None
    bar = header.index("|")
    columns = [
        (match.group(), bar + 1 + match.end())
        for match in re.finditer(r"\S+", header[bar + 1 :])
        if match.group() in JULIA_SUMMARY_COLUMNS
    ]
    return columns or None


def _julia_row_status(
    raw: str, columns: list[tuple[str, int]] | None
) -> TestStatus | None:
    if not columns or "|" not in raw:
        return None
    bar = raw.index("|")
    counts: dict[str, int] = {}
    for match in re.finditer(r"\S+", raw[bar + 1 :]):
        if not match.group().isdigit():
            continue
        end = bar + 1 + match.end()
        name, column_end = min(columns, key=lambda column: abs(column[1] - end))
        if column_end != end or name in counts:
            return None
        counts[name] = int(match.group())
    if "Total" not in counts:
        return None
    if counts.get("Error", 0) > 0:
        return TestStatus.ERROR
    if counts.get("Fail", 0) > 0:
        return TestStatus.FAILED
    return TestStatus.PASSED


def parse_log_julia(log: str) -> ParsedTestLog:  # noqa: PLR0912, PLR0915
    """Parse Julia Test output and return {test_name: status}.

    Parses the test summary table that appears at the end of Julia test runs.
    Format: "Test Summary: | Pass Fail Error Broken Total Time"

    Rules:
      * If a test has Error count > 0 -> ERROR
      * If a test has Fail count > 0 -> FAILED
      * Otherwise (only Pass or no failures) -> PASSED
      * Priority: ERROR > FAILED > PASSED
      * Tests can be hierarchical (indented with spaces)
    """
    results: ParsedTestLog = {}

    # Regex to match test summary lines that look like "  name  | numbers ..."
    test_line_re = re.compile(r"^(\s*)(.+?)\s+\|(.+)$")

    in_summary = False
    has_fail_column = False
    has_error_column = False
    columns: list[tuple[str, int]] | None = None

    for raw in log.splitlines():
        # Detect start of test summary section and column headers
        if "Test Summary:" in raw:
            in_summary = True
            columns = _julia_header_columns(raw)
            # Check which columns are present in the header
            if "| Pass" in raw:
                header_part = raw.split("|")[1] if "|" in raw else raw
                has_fail_column = "Fail" in header_part
                has_error_column = "Error" in header_part
            continue

        if not in_summary:
            continue

        # Match test result lines
        if match := test_line_re.match(raw):
            test_name = match.group(2).strip()
            columns_text = match.group(3)

            if (row_status := _julia_row_status(raw, columns)) is not None:
                results[test_name] = row_status
                continue

            # Parse columns in order: Pass, Fail, Error, Broken, Total, Time
            # Split by whitespace and filter out time values
            parts = columns_text.split()

            # Filter out time values (contain 's', 'm', or ':')
            numeric_parts = [p for p in parts if p.isdigit()]

            if len(numeric_parts) < 2:
                continue

            # The last numeric value is Total
            total = int(numeric_parts[-1])

            # Determine status based on what failures are present
            status = TestStatus.PASSED

            if len(numeric_parts) == 2:
                # Two numbers: either [Pass, Total] or [Fail/Error, Total]
                first_num = int(numeric_parts[0])

                # Check if Pass column has a value (look for leading spaces pattern)
                # When Pass is missing, there are many leading spaces (typically 10+)
                leading_spaces = len(columns_text) - len(columns_text.lstrip())

                if leading_spaces >= 10:
                    # Pass column is empty, so first number is Error/Fail count
                    # We can't distinguish between Error and Fail in this format
                    # Default to ERROR for this case
                    status = TestStatus.ERROR
                elif first_num != total:
                    # Pass is present but doesn't equal Total.
                    # This shouldn't happen in valid output.
                    status = TestStatus.ERROR

            elif len(numeric_parts) == 3:
                # Three numbers: Could be [Pass, Fail, Total] OR [Pass, Error, Total]
                # Depends on which columns are present in the header
                middle_count = int(numeric_parts[1])

                if middle_count > 0:
                    if has_error_column and not has_fail_column:
                        # Header shows Error but no Fail column: [Pass, Error, Total]
                        status = TestStatus.ERROR
                    elif has_fail_column and not has_error_column:
                        # Header shows Fail but no Error column: [Pass, Fail, Total]
                        status = TestStatus.FAILED
                    else:
                        # Both or neither - default to ERROR (more severe)
                        status = TestStatus.ERROR

            elif len(numeric_parts) == 4:
                # Four numbers: [Pass, Fail, Error, Total]
                fail_count = int(numeric_parts[1])
                error_count = int(numeric_parts[2])

                if error_count > 0:
                    status = TestStatus.ERROR
                elif fail_count > 0:
                    status = TestStatus.FAILED

            elif len(numeric_parts) >= 5:
                # Five or more: [Pass, Fail, Error, Broken, Total, ...]
                fail_count = int(numeric_parts[1])
                error_count = int(numeric_parts[2])

                if error_count > 0:
                    status = TestStatus.ERROR
                elif fail_count > 0:
                    status = TestStatus.FAILED

            results[test_name] = status

    return results


def parse_log_lein(log: str) -> ParsedTestLog:  # noqa: PLR0912
    """Parse Leiningen (Clojure) test output into {test_namespace: status}."""
    results: ParsedTestLog = {}
    current_namespace: str | None = None
    lein_re = re.compile(r"^lein test (.+)$")

    for raw in log.splitlines():
        line = raw.strip()
        if not line:
            continue

        if m := lein_re.match(line):
            payload = m.group(1).strip()
            if not payload:
                current_namespace = None
                continue

            tokens = payload.split()
            if tokens and tokens[0] == ":only":
                only_target = " ".join(tokens[1:]).strip()
                if not only_target:
                    current_namespace = None
                    continue
                base = only_target.split("/", 1)[0].strip()
                if base:
                    results.setdefault(base, TestStatus.PASSED)
                    current_namespace = base
                else:
                    current_namespace = None
            else:
                for token in tokens:
                    base = token.strip()
                    if base:
                        results.setdefault(base, TestStatus.PASSED)
                        current_namespace = base
            continue

        if line.startswith("FAIL in") and current_namespace:
            results[current_namespace] = TestStatus.FAILED
            continue
        if line.startswith("ERROR in") and current_namespace:
            results[current_namespace] = TestStatus.ERROR
            continue

    return results


def parse_log_ocaml(log: str) -> ParsedTestLog:
    """Parse Alcotest/QCheck output."""
    results: ParsedTestLog = {}
    status_map = {
        "OK": TestStatus.PASSED,
        "PASS": TestStatus.PASSED,
        "FAIL": TestStatus.FAILED,
        "FAILED": TestStatus.FAILED,
        "ERROR": TestStatus.ERROR,
        "ERR": TestStatus.ERROR,
        "SKIP": TestStatus.SKIPPED,
        "SKIPPED": TestStatus.SKIPPED,
        "TODO": TestStatus.SKIPPED,
    }
    entry_re = re.compile(
        r"^\s*\[(?P<status>[A-Z]+)\]\s+(?P<suite>.*?)\s+(?P<index>\d+)\s+(?P<name>.+)$"
    )

    for raw in log.splitlines():
        line = raw.rstrip()
        if not (match := entry_re.match(line)):
            continue
        test_name = " ".join(
            part
            for part in (
                match.group("suite").strip(),
                match.group("index").strip(),
                match.group("name").strip(),
            )
            if part
        )
        mapped_status = status_map.get(match.group("status"), TestStatus.ERROR)
        if mapped_status == TestStatus.PASSED:
            results.setdefault(test_name, mapped_status)
        else:
            results[test_name] = mapped_status

    return results


OCAML_STATUS_PRECEDENCE = {
    TestStatus.PASSED: 0,
    TestStatus.SKIPPED: 1,
    TestStatus.FAILED: 2,
    TestStatus.ERROR: 3,
}


class _OcamlDuneLogParser:
    running_re = re.compile(r"^Running\[(?P<id>\d+)\]:\s+\((?P<command>.+)\)$")
    output_re = re.compile(r"^Output\[(?P<id>\d+)]:")
    fail_tokens = ("FAIL", "ERROR", "EXCEPTION", "CRASH", "FATAL")
    skip_tokens = ("SKIP", "SKIPPED")

    def __init__(self) -> None:
        self.results: ParsedTestLog = {}
        self.run_lookup: dict[int, str] = {}
        self.active_test: str | None = None
        self.seen_output_tests: set[str] = set()

    @staticmethod
    def _extract_test_name(command_segment: str) -> str:
        command = command_segment.split("&&")[-1].strip()
        if command.startswith("exec "):
            command = command[5:].strip()
        if command.startswith("./"):
            command = command[2:]
        parts = command.split()
        binary = parts[0] if parts else ""
        return binary.rsplit("/", 1)[-1] if binary else ""

    def _update_status(self, name: str, status: TestStatus) -> None:
        if not name:
            return
        current = self.results.get(name)
        if current is None:
            self.results[name] = status
            return
        if OCAML_STATUS_PRECEDENCE.get(status, -1) >= OCAML_STATUS_PRECEDENCE.get(
            current, -1
        ):
            self.results[name] = status

    def handle_line(self, raw_line: str) -> None:
        line = raw_line.strip()
        if not line:
            return
        if match := self.running_re.match(line):
            run_id = int(match.group("id"))
            test_name = self._extract_test_name(match.group("command"))
            self.run_lookup[run_id] = test_name or f"run_{run_id}"
            self.active_test = None
            return
        if match := self.output_re.match(line):
            self.active_test = self.run_lookup.get(int(match.group("id")))
            return
        if not self.active_test:
            return
        self.seen_output_tests.add(self.active_test)
        upper = line.upper()
        if any(token in upper for token in self.fail_tokens):
            self._update_status(self.active_test, TestStatus.FAILED)
        elif any(token in upper for token in self.skip_tokens):
            self._update_status(self.active_test, TestStatus.SKIPPED)
        elif upper == "OK":
            self._update_status(self.active_test, TestStatus.PASSED)

    def finalize(self) -> ParsedTestLog:
        for test_name in self.seen_output_tests:
            self.results.setdefault(test_name, TestStatus.PASSED)
        return self.results


def parse_log_ocaml_v2(log: str) -> ParsedTestLog:
    """Parse dune test logs and return {test_binary: status}."""
    parser = _OcamlDuneLogParser()
    for raw_line in log.splitlines():
        parser.handle_line(raw_line)
    return parser.finalize()


def parse_log_php_v1(log: str) -> ParsedTestLog:  # noqa: PLR0912
    """Parse PHPUnit style test execution logs."""
    results: ParsedTestLog = {}
    cross_mark = chr(0x2A2F)

    def clean(name: str) -> str:
        name = name.rstrip()
        name = re.sub(r"\s{2,}\d+(?:\.\d+)?s\s*$", "", name)
        name = re.sub(r"\s+\d+(?:\.\d+)?s\s*$", "", name)
        return name.strip()

    pass_line_re = re.compile(r"^\s*(?:✓|✔)\s+(?P<name>.+?)\s{2,}\d+\.?\d*s\b.*$")
    pass_line_no_timing_re = re.compile(r"^\s*(?:✓|✔)\s+(?P<name>.+?)\s*$")
    fail_line_re = re.compile(
        rf"^\s*(?:{cross_mark}|x)\s+(?P<name>.+?)\s{{2,}}\d+\.?\d*s\b.*$",
        re.IGNORECASE,
    )
    fail_line_no_timing_re = re.compile(
        rf"^\s*(?:{cross_mark}|x)\s+(?P<name>.+?)\s*$", re.IGNORECASE
    )
    suite_fail_re = re.compile(r"^\s*FAIL(?:ED)?\s+(?P<name>\S.+)$", re.IGNORECASE)
    skipped_line_re = re.compile(r"^\s*-\s+(?P<name>.+?)\s{2,}\d+\.?\d*s\b.*$")
    skipped_line_no_timing_re = re.compile(r"^\s*-\s+(?P<name>.+?)\s*$")
    inline_skipped_marker = re.compile(r"\(skipped\)|\bSKIPPED\b", re.IGNORECASE)

    for raw in log.splitlines():
        line = raw.rstrip()
        if not line:
            continue
        stripped = line.strip()

        if stripped.startswith(("___", "---", "Tests:", "Duration:")):
            continue

        if m := suite_fail_re.match(line):
            results[clean(m.group("name"))] = TestStatus.FAILED
            continue

        for rx in (fail_line_re, fail_line_no_timing_re):
            if m := rx.match(line):
                name = clean(m.group("name"))
                if name:
                    results[name] = TestStatus.FAILED
                break
        else:
            matched_skip = False
            for rx in (skipped_line_re, skipped_line_no_timing_re):
                if m := rx.match(line):
                    name = clean(m.group("name"))
                    if name:
                        results[name] = TestStatus.SKIPPED
                    matched_skip = True
                    break
            if matched_skip:
                continue

            for rx in (pass_line_re, pass_line_no_timing_re):
                if m := rx.match(line):
                    name = clean(m.group("name"))
                    if name:
                        results.setdefault(name, TestStatus.PASSED)
                    break
            else:
                if inline_skipped_marker.search(line):
                    token = re.sub(rf"^\s*(?:✓|✔|{cross_mark}|x|-)\s+", "", line)
                    token = clean(token)
                    if token and token not in results:
                        results[token] = TestStatus.SKIPPED

    return results


def parse_log_phpunit(log: str) -> ParsedTestLog:
    """
    Parser for phpunit logs with the --testdox option.
    Args:
        log (str): log content
        test_spec (TestSpec): test spec (unused)
    Returns:
        dict: test case to test status mapping
    """
    test_status_map = {}
    suite = None

    suite_pattern = r"^(\w.+) \(.+\)$"
    test_pattern = r"^\s*([✔✘↩])\s*(.*)$"

    for line in log.split("\n"):
        suite_match = re.match(suite_pattern, line)
        if suite_match:
            suite = suite_match.groups()[0]
            continue

        test_match = re.match(test_pattern, line)
        if test_match:
            status, test_name = test_match.groups()
            full_test_name = f"{suite} > {test_name}"

            if status == "✔":
                test_status_map[full_test_name] = TestStatus.PASSED
            elif status == "✘":
                test_status_map[full_test_name] = TestStatus.FAILED
            elif status == "↩":
                test_status_map[full_test_name] = TestStatus.SKIPPED

    return test_status_map


def _parse_pytest_status_line(line: str) -> tuple[str, TestStatus] | None:
    line = line.strip()
    if not line:
        return None

    for status in TestStatus:
        status_value = status.value
        if line.startswith(status_value):
            normalized = (
                line.replace(" - ", " ") if status is TestStatus.FAILED else line
            )
            test_case = normalized.split()
            if len(test_case) > 1:
                return test_case[1], status

        match = re.match(
            rf"^(?P<test>\S+::\S.*?)\s+{re.escape(status_value)}(?:\s|$)",
            line,
        )
        if match:
            return match.group("test"), status

    return None


def _pytest_node_id(classname: str, name: str) -> str | None:
    parts = classname.split(".")
    module_indexes = [
        index
        for index, part in enumerate(parts)
        if part.startswith("test_") or part.endswith("_test")
    ]
    if not module_indexes:
        return None
    module_index = module_indexes[-1]
    module = "/".join(parts[: module_index + 1]) + ".py"
    return "::".join([module, *parts[module_index + 1 :], name])


def _parse_appended_junit(log: str) -> ParsedTestLog | None:
    start = log.rfind("<testsuites")
    closing = "</testsuites>"
    end = log.find(closing, start)
    if start < 0 or end < 0:
        return None
    try:
        root = ET.fromstring(log[start : end + len(closing)])
    except ET.ParseError:
        return None

    parsed: ParsedTestLog = {}
    for testcase in root.iter("testcase"):
        classname = testcase.get("classname")
        name = testcase.get("name")
        if not classname or not name:
            return None
        node_id = _pytest_node_id(classname, name)
        if node_id is None:
            return None
        if testcase.find("failure") is not None:
            status = TestStatus.FAILED
        elif testcase.find("error") is not None:
            status = TestStatus.ERROR
        elif testcase.find("skipped") is not None:
            status = TestStatus.SKIPPED
        else:
            status = TestStatus.PASSED
        parsed[node_id] = status
    return parsed or None


def parse_log_pytest(log: str) -> ParsedTestLog:
    """
    Parser for test logs generated with PyTest framework

    Args:
        log (str): log content
    Returns:
        dict: test case to test status mapping
    """
    junit_results = _parse_appended_junit(log)
    if junit_results is not None:
        return junit_results

    test_status_map: ParsedTestLog = {}
    for line in log.split("\n"):
        parsed = _parse_pytest_status_line(line)
        if parsed is None:
            continue
        test_name, status = parsed
        test_status_map[test_name] = status
    return test_status_map


# V1 addition: lines whose status word is not a TestStatus are skipped (the original parser raised ValueError)
def parse_log_pytest_v2(log: str) -> ParsedTestLog:
    """
    Parser for test logs generated with PyTest framework (Later Version)

    Args:
        log (str): log content
    Returns:
        dict: test case to test status mapping
    """
    test_status_map = {}
    escapes = "".join(map(chr, range(1, 32)))
    for line in log.split("\n"):
        line = re.sub(r"\[(\d+)m", "", line)
        translator = str.maketrans("", "", escapes)
        line = line.translate(translator)
        if any(line.startswith(x.value) for x in TestStatus):
            if line.startswith(TestStatus.FAILED.value):
                line = line.replace(" - ", " ")
            test_case = line.split()
            if len(test_case) >= 2 and test_case[0] in {x.value for x in TestStatus}:
                test_status_map[test_case[1]] = TestStatus(test_case[0])
        # Support older pytest versions by checking if line ends with the test status
        elif any(line.endswith(x.value) for x in TestStatus):
            test_case = line.split()
            if len(test_case) >= 2 and test_case[1] in {x.value for x in TestStatus}:
                test_status_map[test_case[0]] = TestStatus(test_case[1])
    return test_status_map


def parse_log_r(log: str) -> ParsedTestLog:
    """Parse R testthat output."""
    results: ParsedTestLog = {}
    passed_context_re = re.compile(r"^✔\s+\|[^|]+\|\s+(.+)$")
    failed_context_re = re.compile(r"^✖\s+\|[^|]+\|\s+(.+)$")
    failure_re = re.compile(r"^Failure\s+\([^)]+\):\s+(.+)$")
    error_re = re.compile(r"^Error\s+\([^)]+\):\s+(.+)$")
    timing_suffix_re = re.compile(r"\s*\[\s*\d+(?:\.\d+)?\s*[a-zA-Z]+\]$")

    def strip_timing_suffix(name: str) -> str:
        return timing_suffix_re.sub("", name).strip()

    for raw in log.splitlines():
        line = raw.strip()
        if m := passed_context_re.match(line):
            results[strip_timing_suffix(m.group(1).strip())] = TestStatus.PASSED
            continue
        if m := failed_context_re.match(line):
            results[strip_timing_suffix(m.group(1).strip())] = TestStatus.FAILED

    in_failed_section = False
    for raw in log.splitlines():
        line = raw.strip()
        if line.startswith(("── Failed tests ──", "== Failed tests ==")):
            in_failed_section = True
            continue
        if in_failed_section and line.startswith(("[ FAIL", "══ Results ══")):
            break
        if in_failed_section and line:
            if m := failure_re.match(line):
                results[strip_timing_suffix(m.group(1).strip())] = TestStatus.FAILED
                continue
            if m := error_re.match(line):
                results[strip_timing_suffix(m.group(1).strip())] = TestStatus.ERROR

    return results


def parse_log_scala_v2(log: str) -> ParsedTestLog:
    """Parse ANSI-colored Scala test output."""
    results: ParsedTestLog = {}
    ansi_escape = re.compile(r"\x1b\[[0-9;]*m|\[(?:[0-9]{1,2}m)")
    passed_re = re.compile(r"^\s*\+\s+(.*?)\s+[\d.]+[mμn]?s\s*$")
    failed_re = re.compile(r"^\s*x\s+(.*?)\s+[\d.]+[mμn]?s\s*$")
    passed_color_re = re.compile(
        r"^\s*\+\s+\[0m\[32m(.*?)\[0m\s+\[90m[\d.]+[mμn]?s\[0m\s*$"
    )
    failed_color_re = re.compile(
        r"^\s*x\s+\[0m\[31m(.*?)\[0m\s+\[90m[\d.]+[mμn]?s\[0m\s*$"
    )

    for raw in log.splitlines():
        line = raw.strip()
        if not line:
            continue
        clean_line = ansi_escape.sub("", line)
        if m := failed_color_re.match(line):
            results[m.group(1)] = TestStatus.FAILED
            continue
        if m := failed_re.match(clean_line):
            results[m.group(1)] = TestStatus.FAILED
            continue
        if m := passed_color_re.match(line):
            results[m.group(1)] = TestStatus.PASSED
            continue
        if m := passed_re.match(clean_line):
            results[m.group(1)] = TestStatus.PASSED

    return results


def parse_log_scala_v3(log: str) -> ParsedTestLog:
    """Parse ScalaTest ``[info]`` lines and combine suites with test names."""
    results: ParsedTestLog = {}
    status_map = {
        "PASSED": TestStatus.PASSED,
        "FAILED": TestStatus.FAILED,
        "ERROR": TestStatus.ERROR,
        "ABORTED": TestStatus.ERROR,
        "SKIPPED": TestStatus.SKIPPED,
        "IGNORED": TestStatus.SKIPPED,
        "CANCELED": TestStatus.SKIPPED,
        "CANCELLED": TestStatus.SKIPPED,
        "PENDING": TestStatus.SKIPPED,
    }
    test_line_re = re.compile(r"^\[info\]\s+-\s+(.*?)(?:\s+\*{3}\s+([A-Z]+)\s+\*{3})?$")
    suite_line_re = re.compile(r"^\[info\]\s+([^\-].*)$")
    current_suite = None

    for raw in log.splitlines():
        line = raw.strip()
        if not line.startswith("[info]"):
            continue
        if line.startswith("[info] -"):
            if match := test_line_re.match(line):
                test_name = re.sub(r"\s*\(\d+\s+\w+\)$", "", match.group(1).strip())
                full_test_name = (
                    f"{current_suite}: {test_name}" if current_suite else test_name
                )
                if status_token := match.group(2):
                    results[full_test_name] = status_map.get(
                        status_token.upper(), TestStatus.ERROR
                    )
                else:
                    results.setdefault(full_test_name, TestStatus.PASSED)
        elif match := suite_line_re.match(line):
            potential_suite = match.group(1).strip()
            if potential_suite and not any(
                skip in potential_suite
                for skip in [
                    "Run completed",
                    "Total number",
                    "Suites:",
                    "Tests:",
                    "All tests",
                    "compiling",
                    "done compiling",
                    "loading",
                    "welcome to",
                    "set current",
                ]
            ):
                current_suite = potential_suite

    return results


def parse_log_swift(log: str) -> ParsedTestLog:
    """Parse Swift XCTest output."""
    results: ParsedTestLog = {}
    record_start_re = re.compile(r"(?=Test (?:Case|Suite) ')")
    test_result_re = re.compile(
        r"Test Case '([^']+)'\s+(passed|failed)\s+\([0-9.]+\s+seconds\)\s*"
    )

    for raw in log.splitlines():
        line = raw.strip()
        if not line.startswith(("Test Case '", "Test Suite '")):
            continue
        for record in record_start_re.split(line):
            if m := test_result_re.fullmatch(record):
                if m.group(2) == "passed":
                    results[m.group(1)] = TestStatus.PASSED
                else:
                    results[m.group(1)] = TestStatus.FAILED

    return results


def parse_logs_r_junit(log: str) -> ParsedTestLog:
    """Parse R testthat JUnit XML output and return {full_test_name: status}.

    Parses JUnit XML format from testthat::test_local(..., reporter = 'junit').
    Returns a dict mapping test names to their status (PASSED, FAILED, SKIPPED, ERROR).
    """
    results: ParsedTestLog = {}

    try:
        # Find the XML content in the log
        xml_start = log.find("<?xml")
        if xml_start == -1:
            return results

        xml_content = log[xml_start:]
        # Try to find the end of XML (closing </testsuites>)
        xml_end = xml_content.find("</testsuites>")
        if xml_end != -1:
            xml_content = xml_content[: xml_end + len("</testsuites>")]

        root = ET.fromstring(xml_content)

        # Iterate through all testsuites and testcases
        for testsuite in root.findall(".//testsuite"):
            for testcase in testsuite.findall("testcase"):
                test_name = testcase.get("name", "")
                classname = testcase.get("classname", "")

                # Create full test name (classname::test_name)
                full_name = f"{classname}::{test_name}" if classname else test_name

                # Check for failure, error, or skipped elements
                if testcase.find("failure") is not None:
                    results[full_name] = TestStatus.FAILED
                elif testcase.find("error") is not None:
                    results[full_name] = TestStatus.ERROR
                elif testcase.find("skipped") is not None:
                    results[full_name] = TestStatus.SKIPPED
                else:
                    # No failure/error/skipped element means the test passed
                    results[full_name] = TestStatus.PASSED

    except ET.ParseError:
        # If XML parsing fails, return empty results
        pass

    return results


def parse_lue_nvim(log: str) -> ParsedTestLog:
    results = {}

    # Precompiled regexes
    ansi_re = re.compile(r"\x1b\[[0-9;]*m")
    success_re = re.compile(r"^Success\s*\|\|\s*(.+?)\s*$")
    fail_re = re.compile(r"^(?:Fail|Failed)\s*\|\|\s*(.+?)\s*$")
    skip_re = re.compile(r"^(?:Skip|Skipped)\s*\|\|\s*(.+?)\s*$", re.IGNORECASE)

    for raw in log.splitlines():
        if not raw.strip():
            continue
        # Remove ANSI escapes then trim whitespace & trailing tabs
        line = ansi_re.sub("", raw).rstrip()

        # Ignore summary counters (they have a colon right after keyword and no '||')
        if ":" in line.split("\t")[0] and "||" not in line:
            # e.g., "Success:" or "Failed :" or "Errors :"
            # ensure we don't skip real test lines which always have '||'
            continue

        # Match order: fail overrides, skip, success (success uses setdefault)
        m_fail = fail_re.match(line)
        if m_fail:
            name = m_fail.group(1).strip()
            results[name] = TestStatus.FAILED
            continue

        m_skip = skip_re.match(line)
        if m_skip:
            name = m_skip.group(1).strip()
            # Only set if not already failed
            results.setdefault(name, TestStatus.SKIPPED)
            continue

        m_success = success_re.match(line)
        if m_success:
            name = m_success.group(1).strip()
            # Don't overwrite FAIL (or SKIPPED) with PASS
            results.setdefault(name, TestStatus.PASSED)
            continue

    return results


def _names_failed_on_any_line(test_log: str, parse_fn: TestParser) -> set[str]:
    failed: set[str] = set()
    for line in test_log.splitlines():
        if not line.strip():
            continue
        try:
            parsed_line = parse_fn(line)
        except Exception:  # noqa: BLE001
            continue
        failed.update(
            name
            for name, status in parsed_line.items()
            if status in (TestStatus.FAILED, TestStatus.ERROR)
        )
    return failed


PLACEHOLDER_TEST_NAMES = {"---NO TEST NAME FOUND YET---", "ok"}


MAVEN_SELECTOR_PLACEHOLDER_RE = re.compile(
    r'^"(?:[A-Za-z_$][\w$]*(?:Test)?(?:,[A-Za-z_$][\w$]*(?:Test)?){2,}'
    r"|[\w.*$-]*\*Test)$"
)


BUILD_SUMMARY_PLACEHOLDER_RE = re.compile(r"^\d+ modules transformed\.$")


SERVER_START_PLACEHOLDER_RE = re.compile(r"^=== \([^)]*\) Starting server\b")


JAVA_CLASS_SELECTOR_RE = re.compile(
    r"^(?:[A-Za-z_$][\w$]*\.)*[A-Z_$][\w$]*(?:Test|Tests|IT|ITCase)$"
)


JAVA_CLASS_GLOB_SELECTOR_RE = re.compile(
    r"^(?:[A-Za-z_$][\w$]*\.)*[A-Z_$*][\w$*]*(?:Test|Tests|IT|ITCase)$"
)


TEST_NAME_NORMALIZERS = (
    (
        re.compile(
            r"^\s*\d+/\d+\s+Test\s+#\d+:\s+(\S.*?)\s+\.+\s*(?:\*{3})?"
            r"(?:Passed|Failed|Not Run|Skipped|Timeout|Exception)\b.*$"
        ),
        r"\1",
    ),
    (
        re.compile(
            r"(?:\s+\[\d+(?:\.\d+)?\s*ms\]"
            r"|\s+\(\d+(?:\.\d+)?\s*m?s\)"
            r"|\s+\(\d+(?:\.\d+)?\s*ms\s+and\s+\d+\s+assertions\)"
            r"|\s+\d+(?:\.\d+)?m?s)$"
        ),
        "",
    ),
    (re.compile(r"\$\$Lambda/0x[0-9a-fA-F]+"), "$$Lambda/0x<id>"),
    (re.compile(r"@[0-9a-fA-F]{4,}\b"), "@<id>"),
    (re.compile(r"0x[0-9a-fA-F]{7,}"), "0x<addr>"),
    (re.compile(r"&\{\[[0-9_ ]{9,}\]\}"), "&{[<addr>]}"),
    (re.compile(r"\b(127\.0\.0\.1|localhost|0\.0\.0\.0):\d+\b"), r"\1:<port>"),
    (
        re.compile(
            r"\b\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?"
            r"(?:Z(?:\[[^\]]+\])?|[+-]\d{2}:\d{2})?\b"
        ),
        "<timestamp>",
    ),
    (
        re.compile(
            r"\b(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun) "
            r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec) "
            r"\d{2} \d{4} \d{2}:\d{2}:\d{2} GMT[+-]\d{4} \([^)]*\)"
        ),
        "<date>",
    ),
    (
        re.compile(
            r"(?<=\[)(?:\d{10}\.\d+|\d{2}-\d{2}-\d{4}|\d{2}:\d{2}:\d{2})(?=\]|-)"
        ),
        "<param>",
    ),
    (re.compile(r"\b(Compiled in )\d+(?:\.\d+)?s\b"), r"\1<duration>s"),
    (re.compile(r"^\d+(?:\.\d+)?%\s+passed$"), "<percent> passed"),
    (re.compile(r"/tmp/[^/\s,)]+"), "/tmp/<tmp>"),
    (re.compile(r"\s+ran (?:\d+|ZERO) tests?\b.*$"), ""),
    (re.compile(r"^\[\s*#\d+\]\s+"), ""),
    (re.compile(r"^(?!\[ok\]: ).+?(?=\[ok\]: )"), ""),
    (re.compile(r"\s+\(cached\)$"), ""),
    (re.compile(r"^\d+\s+(?=\[[\w.-]+\] \u203a )"), ""),
    (re.compile(r"(?<![\w.])\d+\.\d{6,}(?:e-?\d+)?\b"), "<float>"),
)


def _strip_shell_quotes(value: str) -> str:
    value = value.strip()
    while len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1].strip()
    return value.strip("'\"").strip()


def _normalize_test_name_for_compare(value: str) -> str:
    with contextlib.suppress(UnicodeError):
        value = value.encode("latin1").decode("utf-8")
    value = _strip_shell_quotes(ansi_escape(value))
    for pattern, replacement in TEST_NAME_NORMALIZERS:
        value = pattern.sub(replacement, value)
    return _strip_shell_quotes(value)


def _is_placeholder_test_name(name: str) -> bool:
    if (
        name in PLACEHOLDER_TEST_NAMES
        or BUILD_SUMMARY_PLACEHOLDER_RE.fullmatch(name)
        or SERVER_START_PLACEHOLDER_RE.search(name)
    ):
        return True
    return MAVEN_SELECTOR_PLACEHOLDER_RE.fullmatch(
        name
    ) is not None and not _is_java_class_glob_selector(name)


def _normalized_passed_names(names: set[str]) -> set[str]:
    normalized = {_normalize_test_name_for_compare(name) for name in names}
    return {name for name in normalized if name and not _is_placeholder_test_name(name)}


def _is_java_class_selector(name: str) -> bool:
    return JAVA_CLASS_SELECTOR_RE.fullmatch(name) is not None


def _is_java_class_glob_selector(name: str) -> bool:
    return "*" in name and JAVA_CLASS_GLOB_SELECTOR_RE.fullmatch(name) is not None


def _split_java_class_selectors(name: str) -> list[str] | None:
    if "," not in name or "(" in name or any(char.isspace() for char in name):
        return None
    parts = [part.strip() for part in name.split(",") if part.strip()]
    if len(parts) < 2 or not all(
        _is_java_class_selector(part) or _is_java_class_glob_selector(part)
        for part in parts
    ):
        return None
    return parts


def _java_class_aliases(class_name: str) -> set[str]:
    return {class_name, class_name.rsplit(".", 1)[-1]}


def _java_class_name_from_test_name(name: str) -> str | None:
    if _is_java_class_selector(name):
        return name
    return None


def _actual_passed_aliases(names: set[str]) -> set[str]:
    aliases: set[str] = set()
    for name in names:
        aliases.add(name)
        if class_name := _java_class_name_from_test_name(name):
            aliases.update(_java_class_aliases(class_name))
        if " " in name:
            class_name, method_name = name.split(None, 1)
            if _is_java_class_selector(class_name):
                simple_class = class_name.rsplit(".", 1)[-1]
                aliases.add(f"{simple_class} {method_name}")
    return aliases


def _java_method_selector_aliases(name: str) -> set[str]:
    if "#" not in name:
        return set()
    class_name, method_name = name.split("#", 1)
    method_name = method_name.rstrip("()")
    aliases = {f"{class_name} {method_name}", f"{class_name} {method_name}()"}
    if _is_java_class_selector(class_name):
        simple_class = class_name.rsplit(".", 1)[-1]
        aliases.update(
            {f"{simple_class} {method_name}", f"{simple_class} {method_name}()"}
        )
    return aliases


def _expected_selector_satisfied(actual_passed: set[str], expected_name: str) -> bool:
    if _is_java_class_glob_selector(expected_name):
        return any(fnmatch.fnmatchcase(name, expected_name) for name in actual_passed)

    aliases = {expected_name}
    aliases.update(_java_method_selector_aliases(expected_name))
    if _is_java_class_selector(expected_name):
        aliases.update(_java_class_aliases(expected_name))
    return not aliases.isdisjoint(actual_passed)


def _expected_name_satisfied(actual_passed: set[str], expected_name: str) -> bool:
    selector_parts = _split_java_class_selectors(expected_name) or [expected_name]
    return all(
        _expected_selector_satisfied(actual_passed, selector)
        for selector in selector_parts
    )


def normalized_passed_subset(actual: TestResults, expected: TestResults) -> bool:
    actual_passed = _actual_passed_aliases(_normalized_passed_names(actual["PASSED"]))
    expected_passed = _normalized_passed_names(expected["PASSED"])
    return bool(expected_passed) and all(
        _expected_name_satisfied(actual_passed, expected_name)
        for expected_name in expected_passed
    )
