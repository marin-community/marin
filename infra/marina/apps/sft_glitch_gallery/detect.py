# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Find examples of known flaws in a sampled snapshot of the 10.05 SFT mix.

Reads a snapshot written by the SFT Mix Viewer sampler, runs one detector per flaw category over every
document, and writes up to ``--per-category`` examples of each to ``examples.parquet``, cut to a window
around the flaw. The detectors are deliberately narrow patterns from the data review (issue #11188): they
pick clear examples for reading and do not measure prevalence.
"""

import io
import json
import re
import zlib
from collections import defaultdict
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass

import click
import pyarrow as pa
import pyarrow.parquet as pq
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath

from infra.marina.apps.sft_glitch_gallery.categories import Example
from infra.marina.apps.sft_mix_viewer.snapshot import read_snapshot

EXAMPLES_FILE = "examples.parquet"
HEADER = re.compile(r"<\|start_header_id\|>(\w+)<\|end_header_id\|>\n?")
EOT = "<|eot_id|>"
BEGIN = "<|begin_of_text|>"
START_THINK = "<|start_think|>"
END_THINK = "<|end_think|>"
# Window kept around a match: the document head for context, then text on both sides of the match.
HEAD_CHARS = 1_500
BEFORE_CHARS = 5_000
AFTER_CHARS = 9_000
# zlib output over input size below which a long replay document is treated as machine output.
JUNK_COMPRESSION_RATIO = 0.12

GIVE_UP = re.compile(
    r"placeholder|dummy answer|not a correct|does NOT compute|cannot solve this|I give up|I(?:'ll| will) guess|"
    r"answer guess|[Gg]iven (?:the )?(?:lack of |limited )?time|I'm not confident|[Tt]he user may not mind"
)
TRANSLATOR = re.compile(
    r"</?translation>|It seems (?:like )?you want me to translate|haven't provided|(?:kept|copied) as-is|"
    r"it is kept as-is|The (?:provided|source) text|Deja un espacio en blanco"
)
EMPTY_BOX = re.compile(r"\\boxed\{\s*(?:\\displaystyle)?\s*\}")
HARNESS_WARNING = re.compile(
    r"Extra text detected before JSON object|had parsing errors|No valid JSON found|Missing required fields"
)
HIDDEN_TESTS = re.compile(
    r"hidden tests?|the grader|SWE-bench|upstream (?:fix|PR|implementation)|[Ll]et me recall the actual"
)
EXCUSE = re.compile(
    r"pre-existing|(?:un|not )related to (?:my|the|our) changes?|already failing|test patch will (?:update|fix)"
)
GRADER = re.compile(
    r"The verifier (?:extracts|only grades)|expected tool name for this request|Empty or missing files score 0"
)
CHEER = re.compile(r"(?:Perfect|Excellent|Great|Good)!")
CHEER_MIN_TURNS = 6
QA_TAIL = re.compile(r"\n(?:Question|Q):[^\n]*\n+(?:Answer|A):")
# (source substring, pattern inside the reasoning, text whose absence from the user turns makes it phantom)
PHANTOM_RULES = (
    ("competitive_coding_python", re.compile(r"[Nn]ever use sys\.stdin\.buffer"), "sys.stdin.buffer"),
    ("competitive_coding_cpp", re.compile(r"instructions?[^.\n]{0,60}C\+\+17"), "C++17"),
    ("/math_v", re.compile(r"Please reason step by step, and put your final answer within"), "reason step by step"),
    (
        "instruction_following_chat_v3/chat",
        re.compile(r"provided good response|the good response|desired output"),
        "good response",
    ),
)


@dataclass(frozen=True)
class Turn:
    role: str
    start: int  # first character of the turn's content
    visible: int  # first character after the reasoning span; equals start when the turn has none
    end: int


@dataclass(frozen=True)
class Doc:
    source: str
    text: str
    turns: tuple[Turn, ...]

    def content(self, turn: Turn) -> str:
        return self.text[turn.start : turn.end]

    def answer(self, turn: Turn) -> str:
        return self.text[turn.visible : turn.end]


@dataclass(frozen=True)
class Match:
    start: int
    length: int
    note: str


Detector = Callable[[Doc], Match | None]


def parse_turns(text: str) -> tuple[Turn, ...]:
    turns = []
    for header in HEADER.finditer(text):
        start = header.end()
        end = text.find(EOT, start)
        if end == -1:
            end = len(text)
        visible = start
        if text.startswith(START_THINK, start):
            close = text.find(END_THINK, start, end)
            visible = end if close == -1 else close + len(END_THINK)
        turns.append(Turn(header.group(1), start, visible, end))
    return tuple(turns)


def search_in(pattern: re.Pattern, doc: Doc, start: int, end: int, note: str, *, last: bool = False) -> Match | None:
    hits = list(pattern.finditer(doc.text, start, end))
    if not hits:
        return None
    hit = hits[-1] if last else hits[0]
    return Match(hit.start(), hit.end() - hit.start(), note)


def last_assistant(doc: Doc) -> Turn | None:
    return next((turn for turn in reversed(doc.turns) if turn.role == "assistant"), None)


def parses_as_json(text: str) -> bool:
    try:
        json.loads(text)
    except ValueError:
        return False
    return True


def give_up(doc: Doc) -> Match | None:
    if not any(key in doc.source for key in ("/math_v", "competitive_coding", "science_v2/vendor")):
        return None
    turn = last_assistant(doc)
    if turn is None:
        return None
    in_answer = search_in(GIVE_UP, doc, turn.visible, turn.end, "in the final answer")
    tail_start = max(turn.start, turn.visible - 3_000)
    return in_answer or search_in(GIVE_UP, doc, tail_start, turn.visible, "at the end of the reasoning", last=True)


def translator(doc: Doc) -> Match | None:
    if "multilingual_v1" not in doc.source:
        return None
    for turn in doc.turns:
        if turn.role == "user":
            found = search_in(TRANSLATOR, doc, turn.start, turn.end, "in the user prompt")
        elif turn.role == "assistant":
            found = search_in(TRANSLATOR, doc, turn.visible, turn.end, "in the answer")
        else:
            found = None
        if found:
            return found
    return None


def broken_answer(doc: Doc) -> Match | None:
    if "multilingual_v1/math" not in doc.source:
        return None
    turn = last_assistant(doc)
    if turn is None or turn.visible in (turn.start, turn.end):
        return None
    answer = doc.answer(turn)
    empty = EMPTY_BOX.search(answer)
    if empty:
        return Match(turn.visible + empty.start(), empty.end() - empty.start(), "the box is empty")
    if "\\boxed{" not in answer:
        return Match(turn.visible, min(len(answer), 400), "the answer has no boxed result")
    return None


def phantom_instructions(doc: Doc) -> Match | None:
    users = "".join(doc.content(turn) for turn in doc.turns if turn.role == "user")
    for source_key, pattern, absent in PHANTOM_RULES:
        if source_key not in doc.source or absent in users:
            continue
        for turn in doc.turns:
            if turn.role == "assistant":
                found = search_in(pattern, doc, turn.start, turn.visible, f"the user turn never says {absent!r}")
                if found:
                    return found
    return None


def harness_warning(doc: Doc) -> Match | None:
    if not doc.source.startswith(("agenttrove", "penfever-traces")):
        return None
    for previous, turn in zip(doc.turns, doc.turns[1:], strict=False):
        if turn.role != "user" or previous.role != "assistant":
            continue
        found = search_in(HARNESS_WARNING, doc, turn.start, turn.end, "the assistant turn above it is valid JSON")
        if found and parses_as_json(doc.answer(previous)):
            return found
    return None


def hidden_tests(doc: Doc) -> Match | None:
    if not doc.source.startswith("open_swe_traces"):
        return None
    for turn in doc.turns:
        if turn.role == "assistant":
            found = search_in(HIDDEN_TESTS, doc, turn.start, turn.end, "in assistant text")
            if found:
                return found
    return None


def excused_failures(doc: Doc) -> Match | None:
    if not (doc.source.startswith("open_swe_traces") or "/swe_v" in doc.source):
        return None
    cutoff = len(doc.text) - 8_000
    for turn in doc.turns:
        if turn.role == "assistant" and turn.end > cutoff:
            found = search_in(EXCUSE, doc, max(turn.visible, cutoff), turn.end, "in the closing turns")
            if found:
                return found
    return None


def unfinished(doc: Doc) -> Match | None:
    if not doc.turns or doc.turns[-1].role != "assistant":
        return None
    turn = doc.turns[-1]
    answer = doc.answer(turn)
    shown = Match(max(turn.visible, turn.end - 600), min(len(answer), 600), "")
    if "arc_agi" in doc.source and answer.rstrip().endswith("</tool_call>"):
        return Match(shown.start, shown.length, "the document ends on a tool call that is never answered")
    terminus = doc.source.startswith("agenttrove") or (
        doc.source.startswith("penfever-traces") and "opencode" not in doc.source
    )
    if terminus and '"keystrokes"' in answer and '"task_complete": true' not in answer:
        return Match(shown.start, shown.length, "the last turn still issues commands")
    return None


def tool_use_placeholder(doc: Doc) -> Match | None:
    turn = last_assistant(doc)
    if turn is None or doc.answer(turn).strip() != "(tool use)":
        return None
    return Match(turn.visible, turn.end - turn.visible, "the whole final answer")


def grader_leak(doc: Doc) -> Match | None:
    if not doc.source.startswith("penfever-traces"):
        return None
    for turn in doc.turns:
        if turn.role in ("user", "system"):
            found = search_in(GRADER, doc, turn.start, turn.end, f"in a {turn.role} turn")
            if found:
                return found
    return None


def cheerleading(doc: Doc) -> Match | None:
    assistants = [turn for turn in doc.turns if turn.role == "assistant"]
    openers = [turn for turn in assistants if CHEER.match(doc.text, turn.visible)]
    if len(openers) < CHEER_MIN_TURNS:
        return None
    opener = CHEER.match(doc.text, openers[0].visible)
    assert opener is not None
    note = f"{len(openers)} of the {len(assistants)} visible assistant turns open this way"
    return Match(opener.start(), opener.end() - opener.start(), note)


def empty_system(doc: Doc) -> Match | None:
    if len(doc.turns) < 2 or doc.turns[0].role != "system" or doc.turns[1].role != "system":
        return None
    second = doc.turns[1]
    if doc.content(second).strip():
        return None
    header_start = doc.text.rfind("<|start_header_id|>", 0, second.start)
    return Match(header_start, second.end + len(EOT) - header_start, "nothing between the header and the end of turn")


def replay_body(doc: Doc) -> str | None:
    if not doc.source.startswith("replay/"):
        return None
    return doc.text.removeprefix(BEGIN)


def replay_junk(doc: Doc) -> Match | None:
    body = replay_body(doc)
    if body is None or not doc.source.endswith("gt_64k"):
        return None
    encoded = body.encode()
    ratio = len(zlib.compress(encoded)) / len(encoded)
    if ratio >= JUNK_COMPRESSION_RATIO and body.count("\ufffd") < 50:
        return None
    return Match(len(BEGIN), min(len(body), 600), f"compresses to {ratio:.0%} of its size")


def replay_agent(doc: Doc) -> Match | None:
    body = replay_body(doc)
    if body is None:
        return None
    for opening, note in (
        ("This trajectory ", "an agent trajectory in another tool syntax"),
        ("# Repository Context", "an issue with its pull-request diff"),
    ):
        if body.lstrip().startswith(opening):
            return Match(doc.text.index(opening), 300, note)
    benchmark = doc.text.find("HumanEval_")
    if benchmark != -1:
        return Match(benchmark, 40, "a HumanEval results file")
    return None


def replay_synthetic(doc: Doc) -> Match | None:
    body = replay_body(doc)
    if body is None or not doc.source.endswith("lte_64k"):
        return None
    stripped = body.lstrip()
    for opening in ("You are tasked with", "Problem:", "input:"):
        if stripped.startswith(opening):
            return Match(doc.text.index(opening), 200, f"an instruction item opening {opening!r}")
    think = doc.text.find("<think>")
    if think != -1:
        return Match(think, len("<think>"), "a plain-text think tag from another template")
    return search_in(QA_TAIL, doc, max(0, len(doc.text) - 1_500), len(doc.text), "a generated question-and-answer tail")


DETECTORS: dict[str, Detector] = {
    "give_up": give_up,
    "translator": translator,
    "broken_answer": broken_answer,
    "phantom_instructions": phantom_instructions,
    "hidden_tests": hidden_tests,
    "excused_failures": excused_failures,
    "unfinished": unfinished,
    "tool_use_placeholder": tool_use_placeholder,
    "harness_warning": harness_warning,
    "grader_leak": grader_leak,
    "cheerleading": cheerleading,
    "empty_system": empty_system,
    "replay_junk": replay_junk,
    "replay_agent": replay_agent,
    "replay_synthetic": replay_synthetic,
}


def window(text: str, match: Match) -> tuple[str, int]:
    """Cut a long document to its head plus the text around the match; returns it with the match's new start."""
    low = max(0, match.start - BEFORE_CHARS)
    high = min(len(text), match.start + match.length + AFTER_CHARS)
    prefix = ""
    if low > HEAD_CHARS:
        prefix = f"{text[:HEAD_CHARS]}\n\n[… {low - HEAD_CHARS:,} characters omitted …]\n\n"
    else:
        low = 0
    suffix = f"\n\n[… {len(text) - high:,} characters omitted …]" if high < len(text) else ""
    return prefix + text[low:high] + suffix, len(prefix) + match.start - low


def spread_across_sources(examples: list[Example], limit: int) -> list[Example]:
    """Take examples source by source in turn, so one large source does not fill a category."""
    by_source: dict[str, list[Example]] = defaultdict(list)
    for example in examples:
        by_source[example.source].append(example)
    queues = [iter(group) for _, group in sorted(by_source.items())]
    chosen: list[Example] = []
    while queues and len(chosen) < limit:
        remaining = []
        for queue in queues:
            example = next(queue, None)
            if example is not None and len(chosen) < limit:
                chosen.append(example)
                remaining.append(queue)
        queues = remaining
    return chosen


def detect(documents: Iterator[tuple[str, int, int, str]], per_category: int) -> list[Example]:
    """Run every detector over ``(source, doc_index, tokens, text)`` documents."""
    found: dict[str, list[Example]] = defaultdict(list)
    for source, doc_index, tokens, text in documents:
        doc = Doc(source, text, parse_turns(text))
        for category, detector in DETECTORS.items():
            match = detector(doc)
            if match is None:
                continue
            clipped, start = window(text, match)
            found[category].append(
                Example(category, source, doc_index, tokens, match.note, clipped, start, match.length)
            )
    return [example for group in found.values() for example in spread_across_sources(group, per_category)]


def write_examples(root: StoragePath, examples: list[Example]) -> None:
    buffer = io.BytesIO()
    pq.write_table(pa.Table.from_pylist([asdict(example) for example in examples]), buffer, compression="zstd")
    (root / EXAMPLES_FILE).write_bytes(buffer.getvalue())


@click.command(help=__doc__)
@click.option("--snapshot", required=True, help="Snapshot prefix written by the SFT Mix Viewer sampler.")
@click.option("--output", required=True, help="Directory or object-storage prefix for examples.parquet.")
@click.option("--per-category", type=int, default=40, show_default=True)
def main(snapshot: str, output: str, per_category: int) -> None:
    configure_coreweave_s3()
    sampled = read_snapshot(StoragePath.parse(snapshot)).sources
    documents = (
        (entry.source.name, sample.doc_index, sample.tokens, sample.text)
        for entry in sampled
        for sample in entry.samples
    )
    examples = detect(documents, per_category)
    write_examples(StoragePath.parse(output), examples)
    counts: dict[str, int] = defaultdict(int)
    for example in examples:
        counts[example.category] += 1
    click.echo(json.dumps(counts))


if __name__ == "__main__":
    main()
