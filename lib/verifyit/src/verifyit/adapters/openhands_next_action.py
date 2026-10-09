# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Functional next-action grading for OpenHands agent turns.

An expert turn is one tool call (``execute_bash``, ``str_replace_editor``, ``think``,
``task_tracker`` or ``finish``). :func:`grade_next_action` accepts a student's call when it does
the same kind of thing to the same target, rather than when its arguments match exactly. Each call
is normalized into an :class:`Action`: an operation (view, test, search, ...) and the repository
paths it targets, resolved against the repository root and the working directory at that turn.

Tier 1 settles a pair by rules per operation. Pairs the rules leave open (different but possibly
equivalent operations, scratch scripts, ``python -c`` snippets) go to tier 2, an LLM judge, only
when a :class:`NextActionJudge` is given. Without one they score 0 with reason ``undecided``, so the
undecided fraction can be measured before paying for a judge. A judge failure is an
``infra_error`` verdict, never a 0.
"""

import json
import posixpath
import re
import shlex
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from verifyit.grade import InvalidTask, Reward, infra_error, scored
from verifyit.modes.grade_judge import JudgeConnection, grade_judge_candidate
from verifyit.spec import RUBRIC_CHECKLIST, JudgeSpec


class Operation(StrEnum):
    VIEW = "view"
    TEST = "test"
    SEARCH = "search"
    RUN_SCRIPT = "run_script"
    PYTHON_SNIPPET = "python_snippet"
    CREATE = "create"
    EDIT = "edit"
    SHELL = "shell"
    FINISH = "finish"
    THINK = "think"
    TRACKER = "task_tracker"
    UNPARSEABLE = "unparseable"


@dataclass(frozen=True)
class TurnState:
    """Where the agent stands before the turn: the repository root and its working directory."""

    repo_root: str
    cwd: str


@dataclass(frozen=True)
class Action:
    operation: Operation
    targets: frozenset[str] = frozenset()
    """Paths relative to the repository root ("." for the root), or absolute outside it."""
    detail: tuple[str, ...] = ()
    """Operation-specific: search tokens, a shell program and subcommand, an edited old_str, ..."""
    text: str = ""
    """Free text the rules compare loosely: a created file's content or a ``python -c`` snippet."""


@dataclass(frozen=True)
class NextActionJudge:
    """Tier 2. Credentials live in ``connection``, never in a task specification."""

    model: str
    connection: JudgeConnection = field(repr=False)
    max_completion_tokens: int = 2048


def grade_next_action(
    expected_call: dict[str, Any],
    message: dict[str, Any],
    state: TurnState,
    *,
    observation: str = "",
    judge: NextActionJudge | None = None,
) -> Reward:
    """Score a student's assistant message against the expert's next tool call.

    Args:
        expected_call: The expert's ``{"name": ..., "arguments": <JSON string>}``.
        message: The student's OpenAI-style assistant message.
        state: Repository root and working directory before the turn.
        observation: What the expert's call returned; context for the judge.
        judge: Tier 2. ``None`` scores undecided pairs 0 with reason ``undecided``.

    ``detail["components"]`` records ``tool_name`` (the expert's tool), ``same_operation``,
    ``functional`` (equal to the reward), and ``undecided`` (the rules left the pair to the judge).
    """
    expert = call_action(expected_call, state)
    if expert.operation in (Operation.THINK, Operation.TRACKER, Operation.UNPARSEABLE):
        raise InvalidTask(f"expert turn is not a gradable pivot: {expert.operation}")
    tool_calls = message.get("tool_calls") or []
    if len(tool_calls) != 1:
        return _next_action_verdict(0.0, "not_one_tool_call", tool_calls=len(tool_calls))
    student_call = tool_calls[0].get("function") or {}
    student = call_action(student_call, state)
    detail = {
        "expert_operation": expert.operation,
        "student_operation": student.operation,
        "tool_name": student_call.get("name") == expected_call.get("name"),
        "same_operation": student.operation == expert.operation,
    }

    verdict = match_actions(expert, student)
    if verdict is not None:
        accepted, reason = verdict
        return _next_action_verdict(float(accepted), reason, **detail)
    if judge is None:
        return _next_action_verdict(0.0, "undecided", undecided=True, judge="disabled", **detail)
    return _judged(expected_call, student_call, state, observation, judge, detail)


def _next_action_verdict(
    reward: float,
    reason: str,
    *,
    tool_name: bool = False,
    same_operation: bool = False,
    undecided: bool = False,
    **detail: Any,
) -> Reward:
    components = {
        "tool_name": float(tool_name),
        "same_operation": float(same_operation),
        "functional": reward,
        "undecided": float(undecided),
    }
    return scored(reward, reason=reason, components=components, **detail)


def match_actions(expert: Action, student: Action) -> tuple[bool, str] | None:
    """Tier 1: ``(accepted, reason)``, or ``None`` when the rules cannot settle the pair."""
    if student.operation is Operation.UNPARSEABLE:
        return False, "unparseable"
    if student.operation in (Operation.THINK, Operation.TRACKER):
        return False, "deliberation"
    if Operation.FINISH in (expert.operation, student.operation):
        same = expert.operation == student.operation
        return same, "finish" if same else "finish_mismatch"
    if expert.operation != student.operation:
        # Operations that only inspect or run things; two different ones may still serve the same purpose.
        investigative = {
            Operation.VIEW,
            Operation.TEST,
            Operation.SEARCH,
            Operation.RUN_SCRIPT,
            Operation.PYTHON_SNIPPET,
            Operation.SHELL,
        }
        if expert.operation not in investigative or student.operation not in investigative:
            return False, "different_operation"
        if expert.targets and student.targets and not _scopes_overlap(expert.targets, student.targets):
            return False, "different_target"
        return None

    match expert.operation:
        case Operation.VIEW | Operation.RUN_SCRIPT:
            return _decided(bool(expert.targets & student.targets), "same_target", "different_target")
        case Operation.TEST:
            return _match_tests(expert, student)
        case Operation.SEARCH:
            return _match_search(expert, student)
        case Operation.CREATE:
            if expert.targets == student.targets:
                return True, "same_target"
            if all(_is_scratch(path) for path in expert.targets | student.targets):
                return None
            return False, "different_target"
        case Operation.EDIT:
            return _match_edit(expert, student)
        case Operation.PYTHON_SNIPPET:
            return (True, "same_snippet") if _squash(expert.text) == _squash(student.text) else None
        case Operation.SHELL:
            return _decided(expert.detail == student.detail, "same_command", "different_command")
    raise AssertionError(f"unhandled operation {expert.operation}")


def call_action(call: dict[str, Any], state: TurnState) -> Action:
    """Normalize one tool call into an :class:`Action`."""
    name = call.get("name")
    try:
        arguments = json.loads(call.get("arguments") or "{}")
    except json.JSONDecodeError:
        return Action(Operation.UNPARSEABLE)
    if not isinstance(arguments, dict):
        return Action(Operation.UNPARSEABLE)
    match name:
        case "execute_bash":
            return bash_action(str(arguments.get("command", "")), state)
        case "str_replace_editor":
            return _editor_action(arguments, state)
        case "think":
            return Action(Operation.THINK)
        case "task_tracker":
            return Action(Operation.TRACKER)
        case "finish":
            return Action(Operation.FINISH)
    return Action(Operation.UNPARSEABLE)


def bash_action(command: str, state: TurnState) -> Action:
    """Classify the first real command of a bash line, after any leading ``cd``."""
    try:
        segments = _command_segments(command)
    except ValueError:
        return Action(Operation.UNPARSEABLE)
    cwd = state.cwd
    while segments and segments[0][:1] == ["cd"]:
        words = segments.pop(0)
        cwd = _absolute(words[1] if len(words) > 1 else state.repo_root, cwd)
    if not segments:
        return Action(Operation.SHELL, detail=("cd",))
    words = _strip_wrappers(segments[0])
    if not words:
        return Action(Operation.UNPARSEABLE)
    here = TurnState(state.repo_root, cwd)
    program = posixpath.basename(words[0])
    args = words[1:]
    if program.startswith("python"):
        return _python_action(args, here)
    if program in ("pytest", "py.test"):
        return _test_action(args, here)
    if program in ("cat", "head", "tail", "less", "more", "nl", "bat", "wc"):
        return Action(Operation.VIEW, _paths(_operands(args, values=("-n", "-c")), here))
    if program in ("ls", "tree"):
        return Action(Operation.VIEW, _paths(_operands(args) or ["."], here))
    if program == "sed":
        return _sed_action(args, here)
    if program in ("grep", "egrep", "fgrep", "rg", "ag"):
        return _grep_action(args, here)
    if program == "find":
        return _find_action(args, here)
    if program == "git":
        subcommand = next((arg for arg in args if not arg.startswith("-")), "")
        if subcommand == "grep":
            return _grep_action(args[args.index("grep") + 1 :], here)
        return Action(Operation.SHELL, detail=("git", subcommand))
    first = next((arg for arg in args if not arg.startswith("-")), "")
    return Action(Operation.SHELL, detail=(program, first))


def _judged(
    expected_call: dict[str, Any],
    student_call: dict[str, Any],
    state: TurnState,
    observation: str,
    judge: NextActionJudge,
    detail: dict[str, Any],
) -> Reward:
    spec = JudgeSpec(
        rubric=RUBRIC_CHECKLIST,
        criteria=(
            "The candidate's next action serves the same purpose as the expert's next action at this point in "
            "the task: it gathers the same information or makes the same change, even if it uses a different "
            "tool or different arguments.",
        ),
        question="An agent is fixing a GitHub issue in a Python repository. Should its next action be accepted?",
        model=judge.model,
        max_completion_tokens=judge.max_completion_tokens,
    )
    context = (
        f"Repository root: {state.repo_root}\nWorking directory: {state.cwd}\n"
        f"Expert next action: {json.dumps(expected_call)}\n"
        f"What the expert action returned:\n{observation}"
    )
    try:
        verdict = grade_judge_candidate(
            spec, json.dumps(student_call), connection=judge.connection, runtime=None, context=context
        )
    except InvalidTask:
        raise
    except Exception as error:
        # A judge outage must exclude the row, not score the student 0.
        return infra_error(f"judge failed: {error}", **detail)
    return _next_action_verdict(verdict.reward, "judged", undecided=True, judge=verdict.detail, **detail)


def _decided(accepted: bool, accept_reason: str, reject_reason: str) -> tuple[bool, str]:
    return accepted, accept_reason if accepted else reject_reason


def _match_tests(expert: Action, student: Action) -> tuple[bool, str] | None:
    if not expert.targets or not student.targets:
        # A whole-suite run covers any selected file.
        if expert.detail and student.detail and not set(expert.detail) & set(student.detail):
            return None
        return True, "overlapping_tests"
    return _decided(bool(expert.targets & student.targets), "overlapping_tests", "different_tests")


def _match_search(expert: Action, student: Action) -> tuple[bool, str]:
    # Patterns without identifiers (punctuation, short words) compare by scope alone.
    if (expert.detail or student.detail) and not set(expert.detail) & set(student.detail):
        return False, "different_terms"
    return _decided(_scopes_overlap(expert.targets, student.targets), "same_search", "different_scope")


def _match_edit(expert: Action, student: Action) -> tuple[bool, str] | None:
    if not expert.targets & student.targets:
        return False, "different_file"
    if not expert.text or not student.text:
        return None
    return _decided(_texts_overlap(expert.text, student.text), "overlapping_edit", "different_edit")


def _editor_action(arguments: dict[str, Any], state: TurnState) -> Action:
    targets = _paths([str(arguments.get("path", ""))], state)
    match arguments.get("command"):
        case "view":
            return Action(Operation.VIEW, targets)
        case "create":
            return Action(Operation.CREATE, targets, text=str(arguments.get("file_text") or ""))
        case "str_replace":
            return Action(Operation.EDIT, targets, text=str(arguments.get("old_str") or ""))
        case "insert" | "undo_edit":
            return Action(Operation.EDIT, targets)
    return Action(Operation.UNPARSEABLE)


def _python_action(args: list[str], state: TurnState) -> Action:
    if "-c" in args:
        index = args.index("-c")
        return Action(Operation.PYTHON_SNIPPET, text=args[index + 1] if index + 1 < len(args) else "")
    if "-m" in args:
        index = args.index("-m")
        module = args[index + 1] if index + 1 < len(args) else ""
        if module in ("pytest", "unittest"):
            return _test_action(args[index + 2 :], state)
        return Action(Operation.SHELL, detail=("python -m", module))
    script = next((arg for arg in args if not arg.startswith("-")), "")
    if script.endswith(".py"):
        return Action(Operation.RUN_SCRIPT, _paths([script], state))
    return Action(Operation.SHELL, detail=("python", script))


def _test_action(args: list[str], state: TurnState) -> Action:
    keywords: list[str] = []
    files: list[str] = []
    skip_next = False
    for index, arg in enumerate(args):
        if skip_next:
            skip_next = False
            continue
        if arg == "-k" and index + 1 < len(args):
            keywords.extend(_identifiers(args[index + 1]))
            skip_next = True
        elif not arg.startswith("-"):
            files.append(arg.split("::", 1)[0])
    return Action(Operation.TEST, _paths(files, state), detail=tuple(sorted(set(keywords))))


def _sed_action(args: list[str], state: TurnState) -> Action:
    operands = _operands(args, values=("-e",))
    files = operands[1:] if operands else []
    if any(arg.startswith("-i") for arg in args):
        return Action(Operation.EDIT, _paths(files, state))
    return Action(Operation.VIEW, _paths(files, state))


def _grep_action(args: list[str], state: TurnState) -> Action:
    valued = ("-e", "--regexp", "-A", "-B", "-C", "-m", "--max-count", "--include", "--exclude", "--exclude-dir", "-g")
    patterns = [args[index + 1] for index, arg in enumerate(args[:-1]) if arg in ("-e", "--regexp")]
    operands = _operands(args, values=valued)
    if not patterns and operands:
        patterns, operands = [operands[0]], operands[1:]
    tokens = tuple(sorted({token for pattern in patterns for token in _identifiers(pattern)}))
    return Action(Operation.SEARCH, _paths(operands or ["."], state), detail=tokens)


def _find_action(args: list[str], state: TurnState) -> Action:
    first_predicate = next((index for index, arg in enumerate(args) if arg.startswith("-")), len(args))
    roots = args[:first_predicate] or ["."]
    names = [
        args[index + 1]
        for index, arg in enumerate(args[:-1])
        if arg in ("-name", "-iname", "-path", "-ipath", "-regex", "-iregex")
    ]
    tokens = tuple(sorted({token for name in names for token in _identifiers(name)} - {"py"}))
    if not tokens:
        return Action(Operation.VIEW, _paths(roots, state))
    return Action(Operation.SEARCH, _paths(roots, state), detail=tokens)


def _command_segments(command: str) -> list[list[str]]:
    """Split a bash line into commands at ``&&``, ``||``, ``;`` and ``|``; drop redirections."""
    lexer = shlex.shlex(command, posix=True, punctuation_chars=True)
    lexer.whitespace_split = True
    segments: list[list[str]] = [[]]
    skip = False
    for token in lexer:
        if skip:
            skip = False
        elif token in ("&&", "||", ";", "|", "&"):
            segments.append([])
        elif set(token) <= set("<>&"):
            # A redirection: drop it, its target, and a file-descriptor number before it (2>&1).
            if segments[-1] and segments[-1][-1].isdigit():
                segments[-1].pop()
            skip = True
        else:
            segments[-1].append(token)
    return [segment for segment in segments if segment]


def _strip_wrappers(words: list[str]) -> list[str]:
    """Drop environment assignments and wrappers such as ``timeout 60`` or ``env``."""
    words = list(words)
    while words:
        head = words[0]
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*=.*", head) or head in ("env", "nohup", "time", "sudo"):
            words.pop(0)
        elif head == "timeout":
            words = words[2:]
        else:
            return words
    return words


def _operands(args: list[str], values: tuple[str, ...] = ()) -> list[str]:
    """Non-flag arguments, skipping the value after any flag in ``values``."""
    operands: list[str] = []
    skip = False
    for arg in args:
        if skip:
            skip = False
            continue
        if arg in values:
            skip = True
        elif not arg.startswith("-") and not arg.isdigit():
            operands.append(arg)
    return operands


def _absolute(path: str, cwd: str) -> str:
    return posixpath.normpath(path if path.startswith("/") else posixpath.join(cwd, path))


def _paths(paths: list[str], state: TurnState) -> frozenset[str]:
    resolved = set()
    for path in paths:
        if not path:
            continue
        absolute = _absolute(path, state.cwd)
        if absolute == state.repo_root or absolute.startswith(state.repo_root + "/"):
            resolved.add(posixpath.relpath(absolute, state.repo_root))
        else:
            resolved.add(absolute)
    return frozenset(resolved)


def _scopes_overlap(left: frozenset[str], right: frozenset[str]) -> bool:
    def contains(outer: str, inner: str) -> bool:
        return outer == "." or inner == outer or inner.startswith(outer.rstrip("/") + "/")

    return any(contains(a, b) or contains(b, a) for a in left for b in right)


def _is_scratch(path: str) -> bool:
    """A throwaway script: outside the repository, or a repro/debug/test script at its top level."""
    if path.startswith("/"):
        return True
    name = posixpath.basename(path)
    top_level = "/" not in path
    scratch_name = re.match(
        r"(test_|reproduce|repro|debug|verify|check|demo|example|comprehensive|edge|final|simple|minimal|quick|run_)",
        name,
    )
    return top_level and bool(scratch_name) and name.endswith((".py", ".sh"))


def _identifiers(text: str) -> list[str]:
    return [token.lower() for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", text)]


def _squash(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _lines(text: str) -> set[str]:
    return {line.strip() for line in text.splitlines() if line.strip()}


def _texts_overlap(left: str, right: str) -> bool:
    """One edit's old text contains the other's, or they share a non-blank line."""
    if _squash(left) in _squash(right) or _squash(right) in _squash(left):
        return True
    return bool(_lines(left) & _lines(right))
