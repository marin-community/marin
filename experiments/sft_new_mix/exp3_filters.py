# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Reversible, source-aware scripted filters for the SFT candidate pipeline."""

import json
import re
import zlib
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, replace
from enum import StrEnum
from functools import cached_property

BOS = "<|begin_of_text|>"
EOS = "<|end_of_text|>"
HEADER = "<|start_header_id|>"
END_HEADER = "<|end_header_id|>"
EOT = "<|eot_id|>"
THINK = "<|start_think|>"
END_THINK = "<|end_think|>"
MAX_TOKENS = 262_144
REASONING_TAIL_CHARS = 3000
MESSAGE = re.compile(
    re.escape(HEADER) + r"(system|user|assistant|tool|ipython)" + re.escape(END_HEADER) + r"\n?(.*?)" + re.escape(EOT),
    re.S,
)
THOUGHT = re.compile(re.escape(THINK) + r"(.*?)" + re.escape(END_THINK), re.S)
CALL = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.S)
GIVE_UP = re.compile(
    r"\b(?:placeholder|dummy|not a correct|cannot solve|I give up|I'll guess|given the (?:lack of )?time)\b", re.I
)
EXCUSE = re.compile(
    r"(?:fail(?:ed|ing|ure|ures)?|test.{0,20}fail).{0,120}"
    r"(?:pre-existing|preexisting|unrelated|out of scope|not related)"
    r"|(?:pre-existing|preexisting|unrelated).{0,120}(?:fail|test)",
    re.I | re.S,
)
HIDDEN_TEST = re.compile(r"hidden tests?|\bgrader\b|SWE-bench style task", re.I)
PATCH_PATH = re.compile(r"^(?:diff --git a/\S+ b/|\+\+\+ b/)([^\n]+)", re.M)
LEAN_DECLARATION = re.compile(
    r"^\s*(?:(?:private|noncomputable|protected)\s+)?(?:def|theorem|lemma|instance|abbrev)\s+([\w'.]+)", re.M
)
EMAIL = re.compile(r"\b[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")


class Stage(StrEnum):
    SOURCE_EXCLUSIONS = "source_exclusions"
    OUTCOME = "recorded_pass"
    TRACE_ENDING = "complete_trace_ending"
    TEST_TAMPERING = "test_tampering"
    TEST_EXCUSES = "failing_test_excuses"
    HIDDEN_TESTS = "qwen38_hidden_tests"
    REAL_TOOLS = "real_harness_tools"
    GIVE_UP = "give_up_endings"
    THINK_LENGTH = "think_length"
    LEAN = "lean_structural"
    SYSTEM_TURNS = "merge_system_turns"
    WARNINGS = "contradictory_parse_warnings"
    THINK_MODE = "spanless_think_mode"
    GENERATOR_LINES = "generator_lines"
    MINI_JSON = "mini_swe_double_encoded_json"
    REPLAY_HEURISTICS = "replay_heuristics"
    REPLAY_LINE_LENGTH = "replay_mean_line_length"
    REPLAY_EMAILS = "replay_email_scrub"
    CONTEXT_LIMIT = "context_limit"


@dataclass(frozen=True)
class Turn:
    role: str
    content: str


@dataclass(frozen=True)
class Document:
    source: str
    id: str
    text: str
    # Only source-adapter outcomes or checked receipts may set this flag.
    passed: bool = False

    @cached_property
    def messages(self) -> list[Turn]:
        return turns(self.text)


@dataclass(frozen=True)
class FilterConfig:
    enabled: frozenset[Stage]
    # Required, measured per-source limits; a guessed p90 must not enter here.
    think_limits: dict[str, int]
    tools_by_source: dict[str, frozenset[str]]
    give_up_sources: frozenset[str] = frozenset()
    protected_paths: tuple[str, ...] = ("tests", "test", "/tests", ".github", "grader", "verifier")


@dataclass(frozen=True)
class Decision:
    document: Document
    reason: str = ""

    @property
    def keep(self) -> bool:
        return not self.reason


def turns(text: str) -> list[Turn]:
    """Parse complete native transcripts, refusing unparsed non-whitespace text."""
    result = []
    cursor = 0
    for match in MESSAGE.finditer(text):
        prefix = text[cursor : match.start()]
        if prefix.strip() not in ("", BOS):
            raise ValueError("Unparsed content outside native chat turns")
        result.append(Turn(match[1], match[2]))
        cursor = match.end()
    if not result or text[cursor:].strip() not in ("", EOS):
        raise ValueError("Incomplete native chat transcript")
    return result


def with_turns(document: Document, before: list[Turn], after: list[Turn]) -> Document:
    """Splice edited turns back into the original rendered text.

    Turn framing must keep matching MARIN_CHAT_TEMPLATE, which ends assistant
    turns with <|eot_id|> and no newline. Each kept turn reuses its original
    framing bytes; turns dropped or merged away vanish together with their
    framing, so an assistant end-of-turn never gains a newline.
    """
    if before == after:
        return document
    return replace(document, text=splice_turn_bodies(document.text, before, after))


def splice_turn_bodies(text: str, before: list[Turn], after: list[Turn]) -> str:
    """Rebuild native text with edited turn bodies spliced into their framing.

    Each after turn consumes the next unconsumed before turn of its own role
    in order, so surviving turns keep their original relative position;
    skipped turns were dropped or merged away and leave no trace. Every kept
    turn keeps its own header, end-of-turn token and original separator, so
    an assistant end-of-turn never gains a newline.
    """
    matches = list(MESSAGE.finditer(text))
    if len(matches) != len(before) or [match[1] for match in matches] != [turn.role for turn in before]:
        raise ValueError("Turn list does not match the native transcript")
    ends = [match.end() for match in matches]
    tail = text[ends[-1] :]
    eos = ""
    if tail.endswith(EOS):
        eos, tail = EOS, tail[: -len(EOS)]
    if tail.strip():
        raise ValueError("Incomplete native chat transcript")
    separators = [text[ends[j] : matches[j + 1].start()] for j in range(len(matches) - 1)] + [tail]
    pieces = [text[: matches[0].start()]]
    cursor = 0
    for turn in after:
        while cursor < len(before) and before[cursor].role != turn.role:
            cursor += 1
        if cursor >= len(before):
            raise ValueError("Turn edits add turns, which native splicing cannot represent")
        match = matches[cursor]
        pieces.append(text[match.start() : match.start(2)])
        pieces.append(turn.content)
        pieces.append(text[match.end(2) : match.end()])
        pieces.append(separators[cursor])
        cursor += 1
    pieces.append(eos)
    return "".join(pieces)


def unquote_opencode_prompt(content: str, question: str) -> str:
    """Remove only the raw question's generator wrapper and up to three notes."""
    prefix = '"' + question.replace('"', '\\"')
    if not content.startswith(prefix) or not content.endswith('"') or len(content) <= len(prefix):
        return content
    suffix = content[len(prefix) : -1]
    notes = r"(?:\s*<note>(?:(?!</?note>).)*</note>){0,3}\s*"
    if re.fullmatch(notes, suffix, re.S) is None:
        return content
    return question + suffix


def source_exclusion(source: str) -> str:
    """Return "" to keep the source, or the exclusion reason code that drops it."""
    if source.startswith("nemotron_sft_v3/multilingual_v1/"):
        return "multilingual_v1"
    if source == "nemotron_sft_v3/math_v2/medium":
        return "math_medium"
    if source == "agenttrove-glm53-compactions":
        return "compactions"
    if source.startswith("nemotron_sft_v3/agentic_") and source.rsplit("/", 1)[-1] in (
        "interactive_agent",
        "tool_calling",
        "search",
    ):
        return "unvetted_agentic_subset"
    if source.startswith("penfever-traces/"):
        if "qwen35-122b-131k-opencode/" in source:
            return "unverified_opencode"
        if source.rsplit("/", 1)[-1].startswith(("nemotron-gym-", "exp_rpt_e2egit-")):
            return "terminal_qa_or_low_diversity"
    return ""


def is_agent(source: str) -> bool:
    return (
        source.startswith(
            (
                "open_swe_traces/",
                "penfever-traces/",
                "calibforge-relay/",
                "nemotron_sft_v3/swe_",
                "nemotron_sft_v3/opencode_",
            )
        )
        or source == "agenttrove"
    )


def needs_pass(source: str) -> bool:
    return source.startswith(("open_swe_traces/", "penfever-traces/")) or source == "agenttrove"


def protected_path(path: str, prefixes: tuple[str, ...]) -> bool:
    path = path.strip().removeprefix("./")
    parts = path.split("/")
    return (
        any(part in prefixes for part in parts)
        or any(
            part.startswith("test_") or part.endswith(("_test.py", "_test.go", ".test.js", ".spec.js")) for part in parts
        )
        or any(path == prefix or path.startswith(prefix.rstrip("/") + "/") for prefix in prefixes)
    )


def replay_reason(text: str) -> str:
    """Heuristic gates; thresholds require calibration before publication.

    Returns "" to keep the row, or the reason code that drops it.
    """
    stripped = text.removeprefix(BOS).lstrip()
    if stripped.startswith(("This trajectory ", "# Repository Context")):
        return "embedded_trajectory_or_patch"
    if re.search(r"HumanEval[_/][0-9]+", text) and re.search(r"(?:passed|completion|test_results)", text):
        return "benchmark_results"
    raw = text.encode()
    if b"\x00" in raw or sum(ord(char) < 9 or 13 < ord(char) < 32 for char in text) > max(2, len(text) // 100):
        return "binary_controls"
    if len(raw) >= 1024 and len(zlib.compress(raw)) / len(raw) < 0.12:
        return "compression_ratio"
    if re.search(r"(?:[A-Za-z0-9+/]{1000,}={0,2})", text):
        return "base64_like_blob"
    lines = text.splitlines()
    # Normalize numeric columns without erasing alphabetic content in ordinary prose.
    templates = [re.sub(r"\d+", "#", line.strip()) for line in lines if line.strip()]
    if len(templates) >= 20:
        counts = Counter(templates)
        repeated = sum(count for template, count in counts.items() if count > 1 and len(template) >= 10)
        if repeated / len(templates) > 0.6:
            return "templated_lines"
    return ""


def apply_stage(document: Document, stage: Stage, config: FilterConfig, token_count: Callable[[str], int]) -> Decision:
    """Apply one named step. A reason rejects the whole document, never truncates it."""
    source, text = document.source, document.text
    if stage == Stage.REPLAY_LINE_LENGTH and source.startswith("replay/"):
        lines = text.splitlines()
        return Decision(document, "mean_line_length" if lines and sum(map(len, lines)) / len(lines) > 400 else "")
    if stage == Stage.SOURCE_EXCLUSIONS:
        return Decision(document, source_exclusion(source))
    if stage == Stage.OUTCOME:
        return Decision(document, "missing_or_failed_outcome" if needs_pass(source) and not document.passed else "")
    if stage == Stage.CONTEXT_LIMIT:
        return Decision(document, "over_context_limit" if token_count(text) > MAX_TOKENS else "")
    if source.startswith("replay/"):
        if stage == Stage.REPLAY_HEURISTICS:
            return Decision(document, replay_reason(text))
        if stage == Stage.REPLAY_EMAILS:
            return Decision(replace(document, text=EMAIL.sub("[email]", text)))
        return Decision(document)
    if stage in (Stage.REPLAY_HEURISTICS, Stage.REPLAY_LINE_LENGTH, Stage.REPLAY_EMAILS):
        return Decision(document)
    messages = document.messages
    assistant = [turn.content for turn in messages if turn.role == "assistant"]
    if not assistant:
        return Decision(document, "no_assistant_turn")
    if stage == Stage.TRACE_ENDING and is_agent(source):
        final = assistant[-1].strip()
        if messages[-1].role != "assistant" or final.endswith("(tool use)"):
            return Decision(document, "unfinished_trace")
        # Native submit/finish calls are terminal actions. Shell/edit calls are not.
        calls = [json.loads(call) for call in CALL.findall(final)]
        if calls and any(call.get("name") not in ("submit", "finish", "exit", "task_complete") for call in calls):
            return Decision(document, "ends_with_nonterminal_tool_call")
        if final.startswith("{"):
            try:
                action = json.loads(final)
            except json.JSONDecodeError:
                return Decision(document, "incomplete_terminal_json")
            if isinstance(action, dict) and action.get("commands") and not action.get("task_complete"):
                return Decision(document, "ends_with_commands")
    if stage == Stage.TEST_TAMPERING and is_agent(source):
        paths = PATCH_PATH.findall(text)
        if any(protected_path(path, config.protected_paths) for path in paths):
            return Decision(document, "protected_diff_path")
    if stage == Stage.TEST_EXCUSES and is_agent(source) and any(EXCUSE.search(content) for content in assistant):
        return Decision(document, "excuses_failed_tests")
    if stage == Stage.HIDDEN_TESTS and "open_swe_traces/v1_2/minisweagent/qwen38_27b/" in source:
        if any(HIDDEN_TEST.search(content) for content in assistant):
            return Decision(document, "hidden_test_reasoning")
    if stage == Stage.REAL_TOOLS:
        calls = [json.loads(call) for content in assistant for call in CALL.findall(THOUGHT.sub("", content))]
        allowed = config.tools_by_source.get(source, frozenset())
        unknown = sorted({call.get("name", "") for call in calls} - allowed)
        if unknown:
            return Decision(document, "unvetted_tools:" + ",".join(unknown))
    if stage == Stage.GIVE_UP and source in config.give_up_sources:
        for content in assistant:
            thoughts = THOUGHT.findall(content)
            tail = "\n".join(thought[-REASONING_TAIL_CHARS:] for thought in thoughts)
            answer = THOUGHT.sub("", content)
            if GIVE_UP.search(tail) or GIVE_UP.search(answer):
                return Decision(document, "give_up_pattern")
    if stage == Stage.THINK_LENGTH and source in config.think_limits:
        if any(
            token_count(thought) > config.think_limits[source]
            for content in assistant
            for thought in THOUGHT.findall(content)
        ):
            return Decision(document, "over_think_limit")
    if stage == Stage.LEAN and source.startswith("nemotron_sft_v3/math_proofs_v1/"):
        for content in assistant:
            answer = THOUGHT.sub("", content)
            if re.search(r"\bsorry\b", answer):
                return Decision(document, "lean_sorry")
            names = LEAN_DECLARATION.findall(answer)
            if len(names) != len(set(names)):
                return Decision(document, "duplicate_declaration_in_answer")
    if stage == Stage.SYSTEM_TURNS:
        cleaned: list[Turn] = []
        for turn in messages:
            if turn.role == "system" and not turn.content.strip():
                continue
            if turn.role == "system" and cleaned and cleaned[-1].role == "system":
                cleaned[-1] = Turn("system", cleaned[-1].content.rstrip() + "\n\n" + turn.content.lstrip())
            else:
                cleaned.append(turn)
        return Decision(with_turns(document, messages, cleaned))
    if stage == Stage.WARNINGS:
        cleaned = []
        for turn in messages:
            if cleaned and cleaned[-1].role == "assistant" and turn.role in ("tool", "user"):
                previous = THOUGHT.sub("", cleaned[-1].content).strip()
                try:
                    action = json.loads(previous)
                except json.JSONDecodeError:
                    action = None
                # Remove only the documented extra-prose warning. A valid JSON object
                # alone does not prove required fields or command semantics are valid.
                if isinstance(action, dict) and isinstance(action.get("commands"), list):
                    content = re.sub(r"(?m)^.*Extra text detected before JSON object[^\n]*(?:\n|$)", "", turn.content)
                    if content != turn.content:
                        content = re.sub(r"(?m)^WARNINGS:\s*$", "", content).strip()
                        if not content:
                            continue
                        turn = replace(turn, content=content)
            cleaned.append(turn)
        return Decision(with_turns(document, messages, cleaned))
    if stage == Stage.THINK_MODE:
        mode_line = re.compile(r"(?m)^Reasoning: /think(?=\s|$)")
        has_think_mode = any(turn.role == "system" and mode_line.search(turn.content) for turn in messages)
        if has_think_mode and any(THINK not in content for content in assistant):
            if any(THINK in content for content in assistant):
                return Decision(document, "mixed_think_mode")
            cleaned = [
                (
                    replace(turn, content=mode_line.sub("Reasoning: /nothink", turn.content))
                    if turn.role == "system"
                    else turn
                )
                for turn in messages
            ]
            return Decision(with_turns(document, messages, cleaned))
    if stage == Stage.GENERATOR_LINES and "opencode" in source:
        cleaned = [
            (
                replace(
                    turn,
                    content=re.sub(
                        r"(?im)^.*(?:powered by the model named|generator output path|"
                        r"/[^\s]*Qwen3-Coder-480B[^\s]*).*(?:\n|$)",
                        "",
                        turn.content,
                    ),
                )
                if turn.role == "system"
                else turn
            )
            for turn in messages
        ]
        return Decision(with_turns(document, messages, cleaned))
    if stage == Stage.MINI_JSON and "minisweagent" in source:
        cleaned = []
        for turn in messages:
            # Only double encoding is repaired. A native {returncode, output}
            # object stays in the source harness's JSON format.
            if turn.role == "tool":
                try:
                    value = json.loads(turn.content)
                except json.JSONDecodeError:
                    value = None
                if isinstance(value, str):
                    try:
                        inner = json.loads(value)
                    except json.JSONDecodeError:
                        inner = None
                    if isinstance(inner, dict) and "returncode" in inner and "output" in inner:
                        turn = replace(turn, content=value)
            cleaned.append(turn)
        return Decision(with_turns(document, messages, cleaned))
    return Decision(document)
