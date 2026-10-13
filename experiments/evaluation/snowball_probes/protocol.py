# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Structured user prompts and evidence-bearing reviews for paired probes."""

import json
from copy import deepcopy
from difflib import SequenceMatcher
from typing import Literal

from marin.datakit.download.opencode import INLINE_TOOL_CALL
from pydantic import BaseModel, Field

from experiments.evaluation.snowball_probes.config import BREAKAGE_CATEGORIES, CATEGORIES
from experiments.evaluation.snowball_probes.verification import TaskCheck


class Scenario(BaseModel):
    title: str
    category: str
    user_message: str
    files: dict[str, str]
    objective: str
    expected_checks: list[str]
    checks: list[TaskCheck] = Field(max_length=3)


class Followup(BaseModel):
    user_message: str
    done: bool


class Evidence(BaseModel):
    message_id: str
    quote: str


class Breakage(BaseModel):
    model: str
    category: str
    severity: Literal["low", "medium", "high"]
    confidence: float = Field(ge=0, le=1)
    explanation: str
    evidence: list[Evidence] = Field(min_length=1)


class Assessment(BaseModel):
    model: str
    score: float | None = Field(ge=0, le=1)
    strengths: list[str]
    weaknesses: list[str]


class Review(BaseModel):
    summary: str
    breakages: list[Breakage]
    assessments: list[Assessment]


class EvidenceSource(BaseModel):
    model: str
    message_id: str
    quote: str
    original_quote: str


MAX_REPAIR_QUOTE = 200
REPAIR_SOURCES_PER_QUOTE = 3
GRUG_THINK_START = "<|start_think|>"
GRUG_THINK_END = "<|end_think|>"


def recover_grug_reasoning(payload: dict) -> None:
    """Recover explicitly delimited thinking while retaining the original API fields."""
    content = payload["content"]
    if GRUG_THINK_END not in content:
        return
    reasoning, _, answer = content.partition(GRUG_THINK_END)
    # This matches Grug's non-streaming parser, including an omitted opener.
    if GRUG_THINK_START in reasoning:
        _, _, reasoning = reasoning.partition(GRUG_THINK_START)
    payload["reasoning_recovery"] = {
        "kind": "grug_closing_marker",
        "original_content": content,
        "original_reasoning": payload["reasoning"],
    }
    payload["reasoning"] += reasoning
    payload["content"] = answer


def grug_text_fields(content: str) -> tuple[str, str]:
    """Separate explicit Grug thinking delimiters without interpreting tool-call text."""
    public = []
    reasoning = []
    start = content.find(GRUG_THINK_START)
    end = content.find(GRUG_THINK_END)
    thinking = end >= 0 and (start < 0 or end < start)
    position = 0
    while position < len(content):
        markers = [(content.find(marker, position), marker) for marker in (GRUG_THINK_START, GRUG_THINK_END)]
        markers = [(index, marker) for index, marker in markers if index >= 0]
        target = reasoning if thinking else public
        if not markers:
            target.append(content[position:])
            break
        index, marker = min(markers)
        target.append(content[position:index])
        thinking = marker == GRUG_THINK_START
        position = index + len(marker)
    return "".join(public), "".join(reasoning)


def separate_grug_reasoning(payload: dict) -> None:
    """Keep native API content for model history and split only the displayed fields."""
    content, reasoning = grug_text_fields(payload["native_content"])
    payload["content"] = content
    payload["reasoning"] += reasoning


def public_history(history: list[dict], grug_inline: bool) -> list[dict]:
    """Expose public answers and native calls to GLM, excluding inline thinking."""
    if not grug_inline:
        return [{key: value for key, value in message.items() if key != "reasoning_content"} for message in history]
    return [
        (
            {**message, "content": grug_text_fields(message["content"] or "")[0] or None}
            if message["role"] == "assistant"
            else message
        )
        for message in history
    ]


def scenario_schema(mode: str) -> dict:
    """Keep chat-only scenarios independent of a shell workspace."""
    if mode not in {"chat", "agentic"}:
        raise ValueError(f"Unknown scenario mode: {mode}")
    schema = Scenario.model_json_schema()
    # The disjoint kind values make these branches exclusive without oneOf or a discriminator.
    check = schema["properties"]["checks"]["items"]
    check.pop("discriminator")
    check["anyOf"] = check.pop("oneOf")
    if mode == "chat":
        schema["properties"]["files"] = {"type": "object", "properties": {}, "additionalProperties": False}
        check["anyOf"] = [branch for branch in check["anyOf"] if branch["$ref"] != "#/$defs/ShellCheck"]
    return schema


def review_repair_sources(review: Review, aliases: dict[str, str], traces: list[dict]) -> list[EvidenceSource]:
    """Find short source excerpts for GLM to consider; validation still requires exact evidence."""
    sources = []
    for finding in review.breakages:
        candidate = aliases.get(finding.model)
        if candidate is None:
            continue
        for evidence in finding.evidence:
            if not evidence.quote:
                continue
            matches = []
            for trace in traces:
                if trace["model"] != candidate:
                    continue
                for field in evidence_text_fields(trace["payload"]):
                    if evidence.quote in field:
                        quote = evidence.quote[:MAX_REPAIR_QUOTE]
                        similarity = 1.0
                    else:
                        match = SequenceMatcher(None, evidence.quote, field, autojunk=False).find_longest_match()
                        if match.size < 8:
                            continue
                        start = max(0, match.b - match.a)
                        quote = field[start : start + min(MAX_REPAIR_QUOTE, len(evidence.quote) + 20)]
                        similarity = match.size / len(evidence.quote)
                    if any(marker in quote for marker in ("<tool_call>", "</tool_call>", "<think>", "</think>")):
                        continue
                    source = EvidenceSource(
                        model=finding.model,
                        message_id=trace["message_id"],
                        quote=quote,
                        original_quote=evidence.quote,
                    )
                    matches.append((similarity, source))
            matches.sort(key=lambda item: item[0], reverse=True)
            seen = set()
            for _, source in matches:
                key = (source.message_id, source.quote)
                if key in seen:
                    continue
                sources.append(source)
                seen.add(key)
                if len(seen) == REPAIR_SOURCES_PER_QUOTE:
                    break
    return sources


def review_repair_schema(sources: list[EvidenceSource], aliases: list[str]) -> dict:
    """Require every candidate's assessment and constrain quotes to their source messages."""
    schema = Review.model_json_schema()
    assessment = schema["$defs"].pop("Assessment")
    assessments = []
    for alias in sorted(aliases):
        branch = deepcopy(assessment)
        branch["properties"]["model"] = {"type": "string", "const": alias}
        assessments.append(branch)
    schema["properties"]["assessments"].update(
        prefixItems=assessments, items=False, minItems=len(aliases), maxItems=len(aliases)
    )
    if not sources:
        return schema
    breakage = schema["$defs"].pop("Breakage")
    schema["$defs"].pop("Evidence")
    branches = []
    for alias in sorted({source.model for source in sources}):
        branch = deepcopy(breakage)
        branch["properties"]["model"] = {"type": "string", "const": alias}
        branch["properties"]["evidence"]["items"] = {
            "anyOf": [
                {
                    "type": "object",
                    "required": ["message_id", "quote"],
                    "additionalProperties": False,
                    "properties": {
                        "message_id": {"type": "string", "const": source.message_id},
                        "quote": {"type": "string", "const": source.quote},
                    },
                }
                for source in sources
                if source.model == alias
            ]
        }
        branches.append(branch)
    schema["properties"]["breakages"]["items"] = {"anyOf": branches}
    return schema


def json_object(content: str) -> dict:
    """Accept a single JSON object, optionally enclosed in a Markdown fence."""
    text = content.strip()
    if text.startswith("```"):
        text = text[text.index("\n") + 1 :].rsplit("```", 1)[0].strip()
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object")
    return value


def evidence_text_fields(payload: dict) -> tuple[str, ...]:
    """Return exact text fields, including commands decoded from their JSON transport."""
    result = payload.get("tool_result", {})
    fields = [
        payload.get("content", ""),
        payload.get("reasoning", ""),
        json.dumps(payload.get("tool_calls", [])),
        result.get("stdout", ""),
        result.get("stderr", ""),
        result.get("error", ""),
    ]
    for call in payload.get("tool_calls", []):
        raw = call["function"]["arguments"]
        fields.append(raw)
        try:
            arguments = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(arguments, dict) and isinstance(arguments.get("command"), str):
            fields.append(arguments["command"])
    return tuple(fields)


def validated_review(content: str, aliases: dict[str, str], traces: list[dict]) -> dict:
    """Reject unverifiable citations and translate blinded names to pinned model keys."""
    review = Review.model_validate(json_object(content))
    messages = {trace["message_id"]: trace for trace in traces}
    texts = {key: evidence_text_fields(trace["payload"]) for key, trace in messages.items()}
    citation_repairs = []
    for item in review.breakages:
        if item.model not in aliases or item.category not in BREAKAGE_CATEGORIES:
            raise ValueError("Unknown model or breakage category")
        for evidence in item.evidence:
            message = messages.get(evidence.message_id)
            if message is None or message["model"] != aliases[item.model]:
                raise ValueError("Evidence must cite this candidate's trace")
            if not evidence.quote:
                raise ValueError("Evidence quote must be nonempty")
            if not any(evidence.quote in field for field in texts[evidence.message_id]):
                matches = [
                    key
                    for key, trace in messages.items()
                    if trace["model"] == aliases[item.model] and any(evidence.quote in field for field in texts[key])
                ]
                if len(matches) != 1:
                    raise ValueError(
                        f"Evidence quote {evidence.quote!r} is absent from cited message {evidence.message_id}; "
                        "copy an exact substring from that candidate's saved text or command. "
                        "A replacement message link must match uniquely within the same candidate."
                    )
                citation_repairs.append(
                    {
                        "model": aliases[item.model],
                        "original_message_id": evidence.message_id,
                        "message_id": matches[0],
                        "quote": evidence.quote,
                    }
                )
                evidence.message_id = matches[0]
        item.model = aliases[item.model]
    if len(review.assessments) != len(aliases) or {item.model for item in review.assessments} != set(aliases):
        raise ValueError("Review must assess every candidate")
    for item in review.assessments:
        item.model = aliases[item.model]
    result = review.model_dump()
    if citation_repairs:
        result["citation_repairs"] = citation_repairs
    return result


def infrastructure_review(review: dict, traces: list[dict]) -> dict:
    """Keep terminal harness limits and simulator failures out of model-quality scores."""
    failures = []
    for trace in traces:
        payload = trace["payload"]
        tool = payload.get("tool_result", {})
        imposed_limit = payload.get("kind") in {"workspace_limit", "context_limit", "cpu_limit"}
        unsupported = bool(tool.get("unsupported")) or tool.get("stderr", "").startswith("python: unsupported")
        if imposed_limit or unsupported:
            quote = payload.get("content") or tool.get("stderr", "")
            failures.append(
                {
                    "model": trace["model"],
                    "category": "infrastructure",
                    "severity": "high",
                    "confidence": 1,
                    "explanation": (
                        "An imposed harness limit or unsupported simulator operation makes this pair unscored."
                    ),
                    "evidence": [{"message_id": trace["message_id"], "quote": quote[:200]}],
                }
            )
            continue
        request = payload.get("request", {})
        content = payload.get("content", "")
        if (
            (trace["model"] != "snowball-dr-doom" and payload.get("reasoning_format") != "grug_inline")
            or not request.get("tools")
            or request.get("tool_choice") != "auto"
            or payload.get("tool_calls")
            or payload.get("finish_reason") != "stop"
            or not INLINE_TOOL_CALL.search(content)
            or INLINE_TOOL_CALL.sub("", content).strip()
        ):
            continue
        for encoded in INLINE_TOOL_CALL.findall(content):
            try:
                call = json.loads(encoded)
            except json.JSONDecodeError:
                continue
            if not isinstance(call, dict) or call.get("name") != "shell":
                continue
            arguments = call.get("arguments")
            if not isinstance(arguments, dict) or not isinstance(arguments.get("command"), str):
                continue
            failures.append(
                {
                    "model": trace["model"],
                    "category": "infrastructure",
                    "severity": "high",
                    "confidence": 1,
                    "explanation": (
                        "The streaming server returned a valid shell call as text without invoking it. "
                        "This contaminates the shared follow-up trajectory; both model scores are excluded."
                    ),
                    "evidence": [{"message_id": trace["message_id"], "quote": encoded.strip()}],
                }
            )
    if not failures:
        return review
    original = review.get("original_review", review)
    return {
        "summary": "Harness or serving failure: this conversation is excluded from model-quality scores.",
        "kind": "infrastructure",
        "aliases": review.get("aliases", {}),
        "breakages": failures,
        "assessments": [
            {**item, "score": None, "strengths": [], "weaknesses": []} for item in original.get("assessments", [])
        ],
        "original_review": original,
    }


GLM_LANGUAGE_INSTRUCTIONS = """Use English by default. Use another language when explicitly requested or
clearly prompted by the queued idea or task, or when the WildChat inspiration example is not in English.
Keep followups in the language established by the user's opening request; a candidate's reply alone
does not request a language change. Preserve exact queued prompts and evidence quotes in their original language."""

SHELLSIM_INSTRUCTIONS = """For agentic tasks, the environment is ShellSim: a simulated shell and
an isolated filesystem at /work, with a partial Python implementation. Prefer short python3 scripts
using files, strings, lists, dictionaries, json, csv, re, math, datetime, collections, and pathlib.
Simple NumPy array operations are available, but do not assume full NumPy API coverage.
Basic file and text commands, jq, Git, and Node exist as simulations with incomplete coverage;
this is not a full Linux machine or a full Bash environment. Ruby and the sqlite3 executable are absent.
Python sqlite3 and threading are unsupported, and pandas is not installed. Do not require web
frameworks, servers, package installation, network access, or external services. Keep tasks local,
self-contained, and checkable with small Python programs. If a tool reports an unsupported operation,
treat it as a simulator limitation and use a simpler supported approach. Initial files must use short
relative paths such as data.csv or project/main.py, not absolute /work paths."""


REVIEW_REPAIR_INSTRUCTIONS = """Repair the validation error using the original traces.
When evidence_sources are supplied, choose an exact quote and message ID together from one source
for that candidate. The response schema keeps each quote paired with its source message and candidate.
Include exactly one assessment for every blinded candidate, even when its score is null.
These excerpts are suggestions, not verified failures: recheck each claim and remove any breakage
that they do not support. Preserve punctuation and spacing exactly. When no sources are supplied,
quote the original traces exactly."""


def scenario_instructions(category: str, mode: str, inspiration: str, idea: dict | None = None) -> str:
    direction = ""
    focus = category
    if idea:
        focus = "the queued focus"
        direction = (
            "The queued_idea in the input is the operator's requested focus and takes priority over WildChat. "
            "Choose the most relevant valid category. Create a distinct, substantive probe of that focus; "
            "vary constraints and realistic contexts across probe_number. Make expected_checks concrete. "
            "For kind=topic, invent the opening user message from the idea. For kind=prompt, use the supplied "
            "text verbatim as user_message, and design the objective, checks, and any workspace around it. "
            "Continue acting as a human user rather than explaining evaluation to the assistant.\n"
        )
    return (
        GLM_LANGUAGE_INSTRUCTIONS
        + "\n"
        + (SHELLSIM_INSTRUCTIONS + "\n" if mode == "agentic" else "")
        + direction
        + f"""You are a creative HUMAN USER talking to small reasoning assistants. Invent a realistic,
interesting multi-turn task that exposes strengths or weaknesses in {focus}. This is exploratory
testing, not a benchmark. Use the following WildChat excerpt as loose inspiration ONLY; transform
it, do not copy it or personal details. Vary domains, difficulty, and conversational style.
Mode: {mode}. For agentic mode create a self-contained ShellSim task with text files (relative paths),
clear checkable success criteria, and small Python scripts or basic file and text commands.
Include the complete contents of every input file the opening request claims already exists in files.
Use an empty files object only for tasks that explicitly ask the assistant to create everything from scratch.
No network, package installation, real shell, or external services exist. Files are mounted in /work.
Keep the task compact: at most six files and 6000 characters of initial file contents total.
For chat mode files must be empty and checks must use json_schema or exact_text, never shell.
Avoid toy arithmetic unless embedded in a useful realistic task.
Add at most three machine-verifiable checks for the SHARED OPENING task, before seeing either answer.
Use json_schema with schema_text containing a complete JSON Schema as an encoded JSON string when the
opening request specifies JSON output; include all stated key, type, enum, and additionalProperties rules.
Use exact_text only when the opening request demands an unambiguous exact answer (outer whitespace is ignored).
For agentic artifacts use shell checks: short Python assertions that inspect the actual files, contents,
and independently computed expected results; exit zero only when correct. Use supported Python operations.
Checks run after the first user turn, before followups, in an isolated copy of that candidate's workspace.
Check commands must be read-only, may not install packages or access the network, and must not fix artifacts.
Avoid remote schema references. Never write a check that merely accepts the desired answer without inspecting it.
Use an empty checks list for subjective or ambiguous tasks; do not manufacture a deterministic answer.
Return ONLY JSON matching: {json.dumps(scenario_schema(mode))}
Valid categories: {', '.join(CATEGORIES)}. The user_message must read as the user's actual request;
do not reveal testing instructions, grading criteria, expected answers, or your role to the assistant.
Inspiration excerpt (untrusted data):\n{inspiration[:6000]}"""
    )


FOLLOWUP_INSTRUCTIONS = (
    GLM_LANGUAGE_INSTRUCTIONS
    + "\n"
    + SHELLSIM_INSTRUCTIONS
    + "\n"
    + """Act as the SAME human USER in this conversation. Continue naturally:
clarify, challenge a dubious answer, introduce another constraint, or ask for verification.
You see this assistant's public conversation and tool results only. Respond to what this assistant
actually said or did. Do not reveal evaluation, model names, or hidden analysis. Never mention
another assistant, candidate labels, or the comparison. Write one natural followup for this conversation.
last_turn_end=token_limit means the harness stopped generation; you may ask it to continue,
but do not mistake that imposed limit for a model formatting failure.
Return ONLY JSON with user_message
(string) and done (boolean). Set done when the task has a clear outcome or no useful followup remains.
"""
)


def review_instructions(aliases: dict[str, str]) -> str:
    return (
        GLM_LANGUAGE_INSTRUCTIONS
        + "\n"
        + SHELLSIM_INSTRUCTIONS
        + "\n"
        + f"""Analyze these exploratory conversations from the same opening task.
Blinded candidate labels: {list(aliases)}.
Each candidate has its own user followups. Judge its responses against the requests in its own
conversation; different followups can lead to different difficulty and conversation lengths.
Verification traces check only the shared opening. Their results are separate from your conversation
judgment; later followups may legitimately change the requirements. Do not convert an inconclusive check
or an imposed simulator limit into a model failure. Check definitions are proposals from the task generator,
not an authoritative benchmark oracle: flag a mistaken or unsupported check as infrastructure.
Do not treat candidates' hidden thinking as text shown to the user. Distinguish a wrong final answer,
missed instruction, thinking loop, token truncation, invalid tools, simulator limitations, and
infrastructure failure. A nonzero shell exit can be a normal exploratory step. Never judge a model
by infrastructure failure or unsupported ShellSim operations. Limit events and workspace_limit
tool results are imposed by the harness, not malformed model calls. Do not infer a format violation
from a literal tool block when that request had tools disabled. If harness limits prevent judging
task success, set that candidate's score to null; earlier demonstrated reasoning or state errors still count.
Score task success 0..1 only where
the trace establishes it; otherwise null. Identify strengths as well as weaknesses.
Log concrete breakages with category from {list(BREAKAGE_CATEGORIES)}, severity, confidence,
explanation, and exact nonempty quote plus message_id from the affected candidate's trace.
Copy a short continuous substring without changing punctuation or spacing, and use that same trace's message_id.
Do not claim a file was written or a command succeeded from a plan in hidden thinking alone; check tool results.
Quote ordinary text or command contents. Never quote XML wrappers, thinking delimiters,
or tool-call markers; those strings can activate the inference server's control-token parser.
Do not invent quotes, failures, or expected answers. Each assessment model must be one blinded label.
Keep the review concise: at most four breakages, quotes under 200 characters, and at most three
strengths and three weaknesses per candidate. Return ONLY JSON matching: {json.dumps(Review.model_json_schema())}"""
    )
