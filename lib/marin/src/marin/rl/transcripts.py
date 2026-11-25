# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Render SkyRL rollout records as human-readable transcript documents.

Two real record shapes exist in Marin RL artifacts, with semantics mirroring the
EvalDash sample adapter (`infra/marina/apps/evaldash` `skyrlTrajectory.ts`): the
eval-dump trial row (flat, `uid` + `output_response` + `messages` + `verification`)
and the SkyRL schema_v7 turn record (`response.messages`, `response.text`,
`verification_result`, `trajectory.instance_id`). `render_documents` accepts one
record, a list, or JSONL text; records sharing uid and repetition — plus the
training step for turn records — join into one cumulative transcript whose
repeated message prefixes collapse.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

DEFAULT_TRUNCATE_CHARS = 2_000
THINK_START = "<|start_think|>"
THINK_END = "<|end_think|>"
TURN_RECORD_KIND = "turn-v7"
EVAL_DUMP_KIND = "eval-dump"

# Document blocks are tagged tuples: ("heading", level, text), ("paragraph", text),
# ("fields", entries), ("code", text, language), ("details", summary, inner).
Block = tuple


class ToolCall(NamedTuple):
    """One normalized tool call; unparsable JSON arguments keep their raw text."""

    name: str
    arguments: dict[str, Any]


class GroupKey(NamedTuple):
    """One transcript's identity: records sharing it render together."""

    uid: str
    repetition: int | None
    step: int | None


def _clip(text: str, limit: int) -> str:
    """Truncate long text with an explicit elision marker."""
    if limit <= 0 or len(text) <= limit:
        return text
    return text[:limit] + f"\n[… {len(text) - limit} chars elided]"


def _message_text(content: Any) -> str:
    """Flatten message content: a string, a list of `{type, text}` parts, or null."""
    if isinstance(content, str):
        return content
    if isinstance(content, Sequence) and not isinstance(content, (str, bytes)):
        parts = [
            part.get("text", "") for part in content if isinstance(part, Mapping) and isinstance(part.get("text"), str)
        ]
        return "\n\n".join(part for part in parts if part)
    return ""


def _tool_calls(value: Any) -> list[ToolCall]:
    """Normalize OpenAI-style `tool_calls` entries to name plus parsed arguments."""
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    calls = []
    for call in value:
        if not isinstance(call, Mapping):
            continue
        function = call.get("function")
        function = function if isinstance(function, Mapping) else {}
        name = function.get("name")
        if not isinstance(name, str) or not name:
            continue
        raw = function.get("arguments") if "arguments" in function else call.get("arguments")
        if isinstance(raw, Mapping):
            arguments = dict(raw)
        elif isinstance(raw, str):
            try:
                arguments = json.loads(raw)
            except json.JSONDecodeError:
                arguments = {"raw": raw}
        else:
            arguments = {}
        calls.append(ToolCall(name=name, arguments=arguments))
    return calls


def _split_thinking(text: str) -> tuple[list[Block], list[Block]]:
    """Split assistant text into thinking paragraphs and answer paragraphs."""
    thinking: list[Block] = []
    answer: list[Block] = []
    rest = text
    while True:
        start = rest.find(THINK_START)
        if start < 0:
            break
        if rest[:start].strip():
            answer.append(("paragraph", rest[:start].strip()))
        rest = rest[start + len(THINK_START) :]
        end = rest.find(THINK_END)
        if end < 0:
            if rest.strip():
                thinking.append(("paragraph", rest.strip()))
            rest = ""
            break
        if rest[:end].strip():
            thinking.append(("paragraph", rest[:end].strip()))
        rest = rest[end + len(THINK_END) :]
    if rest.strip():
        answer.append(("paragraph", rest.strip()))
    return thinking, answer


def _drop_repeated_prefix(previous: Sequence[Any], nxt: Sequence[Any]) -> list[Any]:
    """Drop the head of `nxt` that repeats `previous` verbatim (cumulative transcripts)."""
    if not previous or len(nxt) < len(previous):
        return list(nxt)
    for index in range(len(previous)):
        if json.dumps(previous[index], sort_keys=True) != json.dumps(nxt[index], sort_keys=True):
            return list(nxt)
    return list(nxt[len(previous) :])


@dataclass(frozen=True)
class Turn:
    # One record normalized across both real shapes.
    uid: str
    repetition: int | None
    step: int | None
    kind: str
    session: str
    phase: str
    env_class: str
    source: str
    model: str
    prompt_text: str
    response_text: str
    messages: list[Mapping[str, Any]] | None
    verification: Mapping[str, Any] | None
    stop_reason: str | None
    exception: str | None
    metrics: Mapping[str, float]
    reward: Mapping[str, Any] | None
    score: Sequence[float] | None

    @property
    def group_key(self) -> GroupKey:
        """One transcript per uid, repetition and, for turn records, training step."""
        return GroupKey(self.uid, self.repetition, self.step if self.kind == TURN_RECORD_KIND else None)


def _numeric_metrics(value: Any) -> dict[str, float]:
    if not isinstance(value, Mapping):
        return {}
    return {name: metric for name, metric in value.items() if isinstance(metric, (int, float))}


def _is_record(value: Any) -> bool:
    return isinstance(value, Mapping)


def is_eval_dump_row(value: Any) -> bool:
    """The eval-dump trial row shape: `uid` plus a raw `output_response` string."""
    return _is_record(value) and isinstance(value.get("uid"), str) and isinstance(value.get("output_response"), str)


def is_turn_record(value: Any) -> bool:
    """The SkyRL schema_v7 turn-record shape: `schema_version` 7 with a nested `response`."""
    return _is_record(value) and value.get("schema_version") == 7 and _is_record(value.get("response"))


def _normalize_eval_dump(record: Mapping[str, Any]) -> Turn:
    step = record.get("ordinal") if isinstance(record.get("ordinal"), int) else record.get("step")
    step = step if isinstance(step, int) else None
    score = record.get("score")
    score = [value for value in score if isinstance(value, (int, float))] if isinstance(score, Sequence) else None
    return Turn(
        uid=record["uid"],
        repetition=None,
        step=step,
        kind=EVAL_DUMP_KIND,
        session=str(record.get("attempt_id") or ""),
        phase="eval",
        env_class=str(record.get("env_class") or ""),
        source=str(record.get("data_source") or record.get("task_id") or ""),
        model="",
        prompt_text=str(record.get("input_prompt") or ""),
        response_text=str(record.get("output_response") or ""),
        messages=record.get("messages") if isinstance(record.get("messages"), Sequence) else None,
        verification=record.get("verification") if _is_record(record.get("verification")) else None,
        stop_reason=record.get("stop_reason") if isinstance(record.get("stop_reason"), str) else None,
        exception=record.get("exception_type") if isinstance(record.get("exception_type"), str) else None,
        metrics=_numeric_metrics(record.get("env_metrics")),
        reward=None,
        score=score or None,
    )


def _normalize_turn_record(record: Mapping[str, Any]) -> Turn:
    response = record.get("response") or {}
    trajectory = record.get("trajectory") or {}
    extras = trajectory.get("environment_extras") or {}
    disposition = record.get("disposition") or {}
    provenance = record.get("provenance") or {}
    instance_id = trajectory.get("instance_id")
    repetition = trajectory.get("repetition_id")
    step = record.get("global_step")
    return Turn(
        uid=instance_id if isinstance(instance_id, str) and instance_id else str(record.get("record_id") or ""),
        repetition=repetition if isinstance(repetition, int) else None,
        step=step if isinstance(step, int) else None,
        kind=TURN_RECORD_KIND,
        session=str(record.get("run_id") or ""),
        phase=str(record.get("phase") or ""),
        env_class=str(trajectory.get("environment_class") or ""),
        source=str(extras.get("data_source") or ""),
        model=str(provenance.get("model_source_identity") or ""),
        prompt_text="",
        response_text=str(response.get("text") or ""),
        messages=response.get("messages") if isinstance(response.get("messages"), Sequence) else None,
        verification=record.get("verification_result") if _is_record(record.get("verification_result")) else None,
        stop_reason=response.get("stop_reason") if isinstance(response.get("stop_reason"), str) else None,
        exception=disposition.get("exception_type") if isinstance(disposition.get("exception_type"), str) else None,
        metrics=_numeric_metrics(record.get("metrics")),
        reward=record.get("reward") if _is_record(record.get("reward")) else None,
        score=None,
    )


def normalize(value: Any) -> Turn | None:
    """Normalize one record whatever real shape it arrived in; None when not a SkyRL record."""
    if is_eval_dump_row(value):
        return _normalize_eval_dump(value)
    if is_turn_record(value):
        return _normalize_turn_record(value)
    return None


def _verification_fields(turn: Turn) -> list[tuple[str, str]]:
    entries: list[tuple[str, str]] = []
    if turn.reward is not None:
        outcome, shaped = turn.reward.get("outcome"), turn.reward.get("shaped")
        if outcome is not None:
            entries.append(("reward outcome", f"{outcome:g}"))
        if shaped is not None and shaped != outcome:
            entries.append(("reward shaped", f"{shaped:g}"))
    if turn.score:
        entries.append(("score list mean", f"{sum(turn.score) / len(turn.score):g}"))
    verification = turn.verification
    if verification:
        for key, label in (("status", "verifier status"), ("score", "verifier score"), ("reason", "verifier reason")):
            value = verification.get(key)
            if value is None or value == "":
                continue
            entries.append((label, value if isinstance(value, str) else f"{value:g}"))
        if verification.get("passed") is not None:
            entries.append(("verifier passed", str(verification["passed"]).lower()))
    return entries


def _turn_blocks(messages: Sequence[Mapping[str, Any]], *, truncate: int) -> list[Block]:
    blocks: list[Any] = []
    for message in messages:
        if not isinstance(message, Mapping):
            continue
        role = message.get("role") if isinstance(message.get("role"), str) and message.get("role") else "user"
        text = _message_text(message.get("content"))
        calls = _tool_calls(message.get("tool_calls"))
        if not text and not calls:
            continue
        if role == "tool":
            name = message.get("name") if isinstance(message.get("name"), str) else ""
            call_id = message.get("tool_call_id") if isinstance(message.get("tool_call_id"), str) else ""
            blocks.append(("heading", 3, f"tool — {name or call_id or 'result'}"))
            blocks.append(("code", _clip(text, truncate), "text"))
            continue
        blocks.append(("heading", 3, role))
        reasoning = message.get("reasoning")
        if isinstance(reasoning, str) and reasoning.strip():
            blocks.append(("details", "Reasoning", (("paragraph", reasoning.strip()),)))
        if role == "assistant" and text:
            thinking, answer = _split_thinking(text)
            if thinking:
                blocks.append(("details", "Thinking", tuple(thinking)))
            blocks.extend(answer)
        elif text:
            blocks.append(("paragraph", text))
        for call in calls:
            blocks.append(("paragraph", f"**Tool call: `{call.name}`**"))
            blocks.append(("code", json.dumps(call.arguments, indent=2), "json"))
    return blocks


def _raw_response_blocks(turn: Turn) -> list[Block]:
    """Render raw response thinking that the message turns do not already carry."""
    if not turn.response_text:
        return []
    if turn.messages is None:
        thinking, answer = _split_thinking(turn.response_text)
        blocks: list[Any] = [("heading", 3, "assistant — raw response")]
        if thinking:
            blocks.append(("details", "Thinking", tuple(thinking)))
        blocks.extend(answer)
        return blocks
    has_thinking = any(
        isinstance(message.get("content"), str) and THINK_START in message["content"] for message in turn.messages
    )
    thinking, _ = _split_thinking(turn.response_text)
    if has_thinking or not thinking:
        return []
    return [("heading", 3, "assistant — raw response"), ("details", "Thinking", tuple(thinking))]


def _outcome_entries(turn: Turn) -> list[tuple[str, str]]:
    """The outcome facts shared by the document header and per-step outcome lines."""
    entries: list[tuple[str, str]] = []
    if turn.stop_reason:
        entries.append(("stop reason", turn.stop_reason))
    entries.extend(_verification_fields(turn))
    if turn.exception:
        entries.append(("exception", turn.exception))
    return entries


def _header_fields(turn: Turn) -> Block:
    entries = [("uid", turn.uid)]
    entries += [
        (name, value)
        for name, value in (
            ("source", turn.source),
            ("environment", turn.env_class),
            ("session", turn.session),
            ("phase", turn.phase),
            ("model", turn.model),
        )
        if value
    ]
    if turn.step is not None:
        entries.append(("step", str(turn.step)))
    entries.extend(_outcome_entries(turn))
    for label, metric in (
        ("generated tokens", "episode/generated_tokens"),
        ("assistant turns", "episode/assistant_turns"),
        ("tool calls", "episode/tool_calls"),
    ):
        value = turn.metrics.get(metric)
        if value is not None:
            entries.append((label, f"{value:g}"))
    if turn.response_text:
        entries.append(("response chars", str(len(turn.response_text))))
    return ("fields", tuple(entries))


def _turn_outcome_fields(turn: Turn) -> Block:
    entries = ([("step", str(turn.step))] if turn.step is not None else []) + _outcome_entries(turn)
    return ("fields", tuple(entries))


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-") or "record"


def _records_from_input(parsed: Any) -> list[Any]:
    """Accept one record, a list of them, or JSONL text (one record or list per line)."""
    if isinstance(parsed, list):
        return list(parsed)
    if not isinstance(parsed, str):
        return [parsed]
    records: list[Any] = []
    for line in parsed.splitlines():
        if line.strip():
            records.extend(value if isinstance(value := json.loads(line), list) else [value])
    return records


def render_documents(parsed: Any, *, truncate: int = DEFAULT_TRUNCATE_CHARS) -> Iterator[tuple[str, list[Block]]]:
    """Yield `(slug, blocks)` documents from one record, a list, or JSONL text."""
    records = _records_from_input(parsed)
    groups: dict[GroupKey, list[Turn]] = defaultdict(list)
    for record in records:
        turn = normalize(record)
        if turn is None:
            raise ValueError("record is not a SkyRL eval-dump row or schema_v7 turn record")
        groups[turn.group_key].append(turn)
    for key, group_turns in groups.items():
        uid, repetition = key.uid, key.repetition
        turns = sorted(group_turns, key=lambda turn: (turn.step is None, turn.step or 0))
        blocks: list[Any] = [
            ("heading", 1, f"Rollout {uid}" + (f" · repetition {repetition}" if repetition is not None else "")),
            _header_fields(turns[-1]),
        ]
        blocks.append(("heading", 2, "Conversation"))
        previous: list[Any] = []
        for index, turn in enumerate(turns):
            messages = turn.messages if turn.messages is not None else []
            fresh = _drop_repeated_prefix(previous, messages) if index else list(messages)
            previous = messages if turn.messages is not None else previous
            if not fresh and turn.prompt_text:
                fresh = [{"role": "user", "content": turn.prompt_text}]
            blocks.extend(_turn_blocks(fresh, truncate=truncate))
            blocks.extend(_raw_response_blocks(turn))
            if len(turns) > 1 and (outcome := _turn_outcome_fields(turn))[1]:
                blocks.append(outcome)
        blocks.append(("heading", 2, "Verifier verdict"))
        verdict = ("fields", tuple(_verification_fields(turns[-1])) or (("verifier status", "absent"),))
        blocks.append(verdict)
        yield _slug(uid) + (f"-r{repetition}" if repetition is not None else ""), blocks


def to_markdown(blocks: Sequence[Any]) -> str:
    """Serialize document blocks to Markdown."""
    lines: list[str] = []
    for block in blocks:
        match block:
            case ("heading", level, text):
                lines += [f"{'#' * level} {text}", ""]
            case ("paragraph", text):
                lines += [text, ""]
            case ("fields", entries):
                lines += [f"- **{name}:** {value}" for name, value in entries if value != ""]
                lines.append("")
            case ("code", text, language):
                lines += [f"```{language}", text, "```", ""]
            case ("details", summary, inner):
                lines += ["<details>", f"<summary>{summary}</summary>", ""]
                lines.append(to_markdown(inner))
                lines += ["</details>", ""]
    return "\n".join(lines).replace("\n\n\n", "\n\n").strip() + "\n"


def to_html(blocks: Sequence[Any]) -> str:
    """Serialize document blocks to a standalone HTML page."""

    def render(inner: Sequence[Any]) -> str:
        parts: list[str] = []
        for block in inner:
            match block:
                case ("heading", level, text):
                    parts.append(f"<h{level}>{html.escape(text)}</h{level}>")
                case ("paragraph", text):
                    parts.append(f"<p>{html.escape(text)}</p>")
                case ("fields", entries):
                    rows = "".join(
                        f"<tr><th>{html.escape(name)}</th><td>{html.escape(value)}</td></tr>" for name, value in entries
                    )
                    parts.append(f"<table>{rows}</table>")
                case ("code", text, _):
                    parts.append(f"<pre><code>{html.escape(text)}</code></pre>")
                case ("details", summary, nested):
                    parts.append(f"<details><summary>{html.escape(summary)}</summary>{render(nested)}</details>")
        return "\n".join(parts)

    return (
        '<!DOCTYPE html>\n<html><head><meta charset="utf-8">'
        "<title>SkyRL rollout transcript</title></head>\n<body>\n" + render(blocks) + "\n</body></html>\n"
    )


def main(argv: Sequence[str] | None = None) -> int:
    """CLI: render record JSON/JSONL files into Markdown (and optional HTML) documents."""
    parser = argparse.ArgumentParser(description="Render SkyRL rollout records as readable Markdown transcripts.")
    parser.add_argument("inputs", nargs="+", help="record .json (object or array) or .jsonl files")
    parser.add_argument("--out", required=True, help="directory to write rendered documents into")
    parser.add_argument("--html", action="store_true", help="also write a standalone HTML page per document")
    parser.add_argument(
        "--truncate", type=int, default=DEFAULT_TRUNCATE_CHARS, help="tool-result truncation limit in chars"
    )
    args = parser.parse_args(argv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for input_text in args.inputs:
        raw = Path(input_text).read_text()
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = raw  # JSONL: one record or record list per line
        for index, (slug, blocks) in enumerate(render_documents(parsed, truncate=args.truncate)):
            name = f"{index:04d}-{slug}"
            (out_dir / f"{name}.md").write_text(to_markdown(blocks))
            if args.html:
                (out_dir / f"{name}.html").write_text(to_html(blocks))
            written += 1
    print(f"wrote {written} documents to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
