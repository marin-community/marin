# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The kinds of flaw the gallery shows, with the prevalence measured in the 10.05 data review (issue #11188)."""

from dataclasses import dataclass
from enum import StrEnum


class Severity(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass(frozen=True)
class Category:
    key: str
    title: str
    severity: Severity
    description: str
    prevalence: str  # as measured in the review sample; the gallery shows examples, not rates


@dataclass(frozen=True)
class Example:
    """One flawed document, cut to a window around the flaw; the match is ``text[match_start:][:match_length]``."""

    category: str
    source: str
    doc_index: int
    tokens: int
    note: str
    text: str
    match_start: int
    match_length: int


CATEGORIES = (
    Category(
        "give_up",
        "Reasoning gives up, answer hides it",
        Severity.HIGH,
        "Long reasoning ends in a guess or a placeholder, and the final answer states it without a caveat.",
        "4.4% of math_v3 documents (8.9% of its tokens); 5.7% of Python competitive coding; 45% of science_v2/vendor.",
    ),
    Category(
        "loop_assistant",
        "Loops in assistant text",
        Severity.HIGH,
        "The assistant repeats the same tokens back to back. A few are runaway loops inside reasoning; "
        "others are grids or tactic chains.",
        "21 of 17,096 SFT documents scanned at full length; 8 with a loop of 500 tokens or more.",
    ),
    Category(
        "reemission",
        "The same passage re-emitted many times",
        Severity.MEDIUM,
        "Reasoning writes out the same code or sentence again and again, with other text between the copies.",
        "131 of 276 OpenThoughts4 documents repeat one 32-token window ten or more times; 135 of 390 math_proofs_v2.",
    ),
    Category(
        "translator",
        "Translator commentary in the data",
        Severity.HIGH,
        "The machine translator's refusals and notes were kept as the prompt or the answer.",
        "21% of multilingual v1 math and 9% of v1 code documents.",
    ),
    Category(
        "broken_answer",
        "Translated answer lost its result",
        Severity.HIGH,
        "The final answer of a translated math document has no boxed result, or an empty box.",
        "13% of multilingual v1 math documents.",
    ),
    Category(
        "phantom_instructions",
        "Reasoning quotes instructions the prompt never gave",
        Severity.MEDIUM,
        "The prompt used at generation time was replaced, and the reasoning still refers to it.",
        "69% of Python competitive coding; 53% of C++; 22% of math_v3; 22% of instruction-following chat v3.",
    ),
    Category(
        "hidden_tests",
        "Agent reasons about hidden tests",
        Severity.HIGH,
        "The assistant talks about hidden tests, a grader or the benchmark, or recalls the upstream fix from memory.",
        "40% of open SWE traces from the qwen38_27b teacher; under 1% for other teachers.",
    ),
    Category(
        "excused_failures",
        "Failing tests explained away",
        Severity.HIGH,
        "The agent submits while tests fail, calling the failures pre-existing or unrelated without checking.",
        "About 13% of open SWE traces; 6 of 93 Nemotron SWE traces.",
    ),
    Category(
        "unfinished",
        "Rollout ends mid-task",
        Severity.HIGH,
        "The document stops while the agent is still issuing commands, or on a tool call that is never answered.",
        "47 of 412 penfever Terminus rollouts; 18 of 99 AgentTrove; all 60 ARC documents.",
    ),
    Category(
        "tool_use_placeholder",
        "Final answer is a placeholder",
        Severity.HIGH,
        "The last assistant turn is the literal text (tool use), with no work done.",
        "14 of 108 penfever OpenCode documents.",
    ),
    Category(
        "harness_warning",
        "Harness warning contradicts the turn",
        Severity.MEDIUM,
        "The harness complains about extra text or invalid JSON, but the assistant turn as rendered is clean JSON. "
        "The converter moved the reasoning and kept the complaint.",
        "30% of AgentTrove assistant turns; 25% of penfever Terminus documents.",
    ),
    Category(
        "grader_leak",
        "Prompt describes the grader",
        Severity.MEDIUM,
        "A question is answered through a terminal harness, and the prompt says how the answer will be graded.",
        "116 of 520 penfever documents.",
    ),
    Category(
        "cheerleading",
        "Cheerleading openers",
        Severity.MEDIUM,
        "Turn after turn opens with Perfect!, Excellent! or Great!, whatever the tool returned.",
        "26% of assistant turns from the qwen35 and qwen36 SWE teachers; about 1% for other teachers.",
    ),
    Category(
        "empty_system",
        "Second, empty system turn",
        Severity.LOW,
        "The document has two system turns and the second is empty.",
        "314 of 5,280 SFT documents; all of finance_v1.",
    ),
    Category(
        "loop_user",
        "Loops in the user turn",
        Severity.MEDIUM,
        "The prompt itself contains a long repetition: a translator loop, a pasted table, or a grid.",
        "38 of 17,096 SFT documents have a user-turn loop of 500 tokens or more.",
    ),
    Category(
        "loop_tool",
        "Loops in tool output",
        Severity.LOW,
        "Tool results full of one repeated line: recursive tracebacks, progress bars, escape codes.",
        "138 of 17,096 SFT documents have a tool-turn loop of 500 tokens or more, mostly open SWE traces.",
    ),
    Category(
        "loop_replay",
        "Loops in pretraining replay",
        Severity.HIGH,
        "Replay documents that are one line or record repeated: logs, spreadsheets, arrays, generated filler.",
        "8.9% of long replay documents are at least half repeated 32-token windows.",
    ),
    Category(
        "replay_junk",
        "Replay data dumps and machine output",
        Severity.HIGH,
        "Long replay documents that are binary data, serialised objects, numeric tables or other highly "
        "compressible machine output.",
        "50.8% of the tokens in replay documents over 64k tokens.",
    ),
    Category(
        "replay_agent",
        "Agent runs, gold patches and benchmark files in replay",
        Severity.HIGH,
        "Failed SWE agent trajectories in another tool syntax, issue-plus-patch documents, and benchmark result files.",
        "57 of 215 sampled documents in replay/c01/q3 and q4 over 64k tokens.",
    ),
    Category(
        "replay_synthetic",
        "Instruction data inside short replay",
        Severity.MEDIUM,
        "Replay documents that are synthetic instruction items in another format, or carry a generated "
        "question-and-answer tail.",
        "About 11% of short replay by share for instruction items; 65 of 825 documents carry a generated tail.",
    ),
)
