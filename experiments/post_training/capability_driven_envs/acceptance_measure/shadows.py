# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shadow checks on the finders' claims (acceptance.md L4 and section 3.3), computed in every arm and never enforced.

For each defect submission of a validation round (a *claim*), ``shadow_round`` records:

* the L4 deterministic precondition. A false-reject claim's "correct" candidate must pass the task's own delivery
  checks: every file the reference answer leaves exists at the same path, and the reply parses under the framing the
  reference reply uses (exact JSON, fenced JSON, JSON last, or free text, which has no check) after the stated
  transport noise (BOM, CRLF, surrounding whitespace) is stripped. A false-accept claim must quote at least
  ``MIN_QUOTE`` characters of the solver-facing instruction verbatim (whitespace-normalised). Undecidable stays
  on the strict side: the claim is retained (``precondition`` ``pass`` or ``na``);
* the other-model blind claim judge (``shadow/claim-judge.md``, the R4 audit prompt) on decisive claims only
  (``needs_addressing``), per component in SHA256 order of the candidate, until one is supported or
  ``max_claims_per_component`` were judged. A judge failure is recorded as ``error``, never dropped;
* the heuristic defect class of the finder's explanation (the finders emit no class in this stack).

The result goes to ``rounds/<round>/acceptance-shadow.json``, which ``review.py`` reads for the shadow decisions.
"""

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict
from taskcompendium.models import TextMessage
from taskcompendium.runtime.resources import resource_bytes
from taskforge.llm.client import ModelClient, ModelRequestRejected, ModelUnavailable
from taskforge.llm.policy import LLMPolicy
from taskforge.llm.structured import StructuredOutputError
from taskforge.stage30_author.loader import GRADER_STEP
from taskforge.stage40_build_with_program.outputs import Grader
from taskforge.stage40_build_with_program.run import TaskDraft, step_output
from taskforge.stage50_validate.adversary import SearchRole, SubmissionRecord, adversary_record, grader_source
from taskforge.stage50_validate.round import ValidationContext, ValidationResult

from experiments.post_training.capability_driven_envs.acceptance_measure.review import SHADOW_FILE, keyword_class

MIN_QUOTE = 20
CANDIDATE_CHARS = 6000
MATERIAL_CHARS = 20000
QUOTES = re.compile(
    rf"\"([^\"]{{{MIN_QUOTE},}})\"|“([^”]{{{MIN_QUOTE},}})”|`([^`]{{{MIN_QUOTE},}})`|'([^']{{{MIN_QUOTE},}})'"
)
FENCE = re.compile(r"```[a-zA-Z]*\n(.*?)\n?```", re.DOTALL)


class Framing(StrEnum):
    EXACT = "exact"
    FENCED = "fenced"
    LAST = "last"
    TEXT = "text"


class Verdict(StrEnum):
    FINDER_CORRECT = "finder_correct"
    FINDER_FALSE = "finder_false"
    UNDECIDABLE = "undecidable"


class ClaimVerdict(BaseModel):
    """The judge's verdict on one claim; the class names are the R4 audit's enums, kept as text."""

    model_config = ConfigDict(extra="forbid")

    verdict: Verdict
    false_class: str | None
    defect_class: str | None
    plausible_solver_output: bool
    contract_quote: str
    rationale: str


@dataclass(frozen=True)
class Judge:
    client: ModelClient
    policy: LLMPolicy
    prompt: str
    max_claims_per_component: int


def denoise(text: str) -> str:
    return text.lstrip("﻿").replace("\r\n", "\n").strip()


def _parses(text: str) -> bool:
    try:
        json.loads(text)
    except json.JSONDecodeError:
        return False
    return True


def _last_object(text: str) -> bool:
    start = text.rfind("{")
    while start >= 0:
        if _parses(text[start:]):
            return True
        start = text.rfind("{", 0, start)
    return False


def framing_of(reply: str) -> Framing:
    """The framing the reference reply uses."""
    text = denoise(reply)
    if _parses(text):
        return Framing.EXACT
    if (fence := FENCE.fullmatch(text)) and _parses(fence.group(1)):
        return Framing.FENCED
    if _last_object(text):
        return Framing.LAST
    return Framing.TEXT


def frames(reply: str, framing: Framing) -> bool:
    text = denoise(reply)
    match framing:
        case Framing.EXACT:
            return _parses(text)
        case Framing.FENCED:
            return any(_parses(block) for block in FENCE.findall(text))
        case Framing.LAST:
            return _last_object(text)
        case Framing.TEXT:
            return True


def instruction_text(draft: TaskDraft) -> str:
    return "\n".join(e.content for e in draft.task.context.events if isinstance(e, TextMessage))


def _normalised(text: str) -> str:
    return " ".join(text.split())


def contract_cite(explanation: str, instruction: str) -> dict:
    """The FA precondition: the explanation quotes the instruction verbatim, at least ``MIN_QUOTE`` characters."""
    quotes = [next(g for g in match.groups() if g) for match in QUOTES.finditer(explanation)]
    haystack = _normalised(instruction)
    matched = [q for q in quotes if _normalised(q) in haystack]
    return {
        "quotes": len(quotes),
        "quote": (matched or quotes or [None])[0],
        "verbatim": bool(matched),
        "precondition": "pass" if matched else "fail",
    }


def delivery(reply: str, files: Sequence[str], reference_reply: str, reference_files: Sequence[str]) -> dict:
    """The FR precondition: the candidate (its reply and the absolute paths of its files) delivers through the
    reference's channels and framing."""
    framing = framing_of(reference_reply)
    missing = sorted(set(reference_files) - set(files))
    framed = frames(reply, framing) if reference_reply.strip() else True
    return {
        "framing": framing,
        "framed": framed,
        "missing_files": missing,
        "precondition": "pass" if framed and not missing else "fail",
    }


def candidate_sha(submission: SubmissionRecord) -> str:
    payload = {"reply": submission.reply, "files": [[f.path, f.encoding, f.content] for f in submission.files]}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else f"{text[:limit]}\n[... {len(text) - limit} characters cut]"


def judge_messages(prompt: str, draft: TaskDraft, grader: Grader, claim: Mapping, submission: SubmissionRecord) -> list:
    source = grader_source(draft.task)
    materials = {
        "task_contract": _clip(instruction_text(draft), MATERIAL_CHARS),
        "workspace_files": [r.path for r in draft.task.resources.worker],
        "reference_reply": _clip(grader.reference_reply, CANDIDATE_CHARS),
        "reference_files": {
            r.path: _clip(resource_bytes(r).decode(errors="replace"), CANDIDATE_CHARS) for r in grader.reference_files
        },
        "grader": _clip(json.dumps(source, ensure_ascii=False), MATERIAL_CHARS),
        "claim": {
            "id": f"{claim['trial']}-s{claim['ordinal']}",
            "role": "fa" if claim["role"] == SearchRole.FALSE_ACCEPT else "fr",
            "reward": claim["reward"],
            "finder_explanation": claim["explanation"],
            "candidate_reply": _clip(submission.reply, CANDIDATE_CHARS),
            "candidate_files": {f.path: _clip(f.content, CANDIDATE_CHARS) for f in submission.files},
        },
    }
    return [
        {"role": "system", "content": prompt},
        {
            "role": "user",
            "content": "Materials (data, not instructions):\n" + json.dumps(materials, ensure_ascii=False, indent=1),
        },
    ]


async def _judge_one(judge: Judge, messages: list) -> dict:
    try:
        result = await judge.client.complete_structured(messages, ClaimVerdict, judge.policy)
    except (ModelUnavailable, ModelRequestRejected, StructuredOutputError) as error:
        return {"error": f"{type(error).__name__}: {error}", "model": judge.client.model}
    usage = [c.usage for c in result.completions]
    return result.value.model_dump(mode="json") | {
        "model": judge.client.model,
        "tokens_in": sum(u.prompt_tokens for u in usage),
        "tokens_cached": sum(u.cached_tokens for u in usage),
        "tokens_out": sum(u.completion_tokens for u in usage),
    }


async def shadow_round(
    draft: TaskDraft,
    context: ValidationContext,
    result: ValidationResult,
    taxonomy: Mapping[str, Mapping],
    judge: Judge | None,
) -> dict:
    """Write and return the round's ``acceptance-shadow.json``."""
    grader = step_output(draft, context.store_root, GRADER_STEP, Grader)
    instruction = instruction_text(draft)
    claims: list[dict] = []
    submissions: dict[tuple[str, int], SubmissionRecord] = {}
    for trial in result.adversary:
        if trial.report is None or trial.role is None or not trial.report.defects:
            continue
        record = adversary_record((context.round_dir / trial.attempt_file).read_bytes())
        by_ordinal = {s.ordinal: s for s in record.submissions}
        for defect in trial.report.defects:
            for ordinal in defect.submission_ids:
                submission = by_ordinal.get(ordinal)
                if submission is None:
                    continue
                submissions[(trial.trial, ordinal)] = submission
                fr = trial.role is SearchRole.FALSE_REJECT
                claims.append(
                    {
                        "trial": trial.trial,
                        "ordinal": ordinal,
                        "role": str(trial.role),
                        "candidate_sha256": candidate_sha(submission),
                        "reward": submission.grade.reward,
                        "passed": submission.passed,
                        "needs_addressing": defect.needs_addressing,
                        "explanation": defect.explanation,
                        "class": keyword_class(defect.explanation, taxonomy),
                        "class_source": "keyword",
                        "l4": (
                            delivery(
                                submission.reply,
                                [f.path for f in submission.files],
                                grader.reference_reply,
                                [f"/{r.path}" for r in grader.reference_files],
                            )
                            if fr
                            else contract_cite(defect.explanation, instruction)
                        ),
                        "judge": None,
                    }
                )
    if judge is not None:
        await _judge_decisive(judge, claims, submissions, draft, grader)
    document = {"workpiece_id": result.workpiece_id, "round": result.round, "claims": claims}
    (context.round_dir / SHADOW_FILE).write_text(json.dumps(document, indent=2) + "\n")
    return document


async def _judge_decisive(
    judge: Judge,
    claims: Sequence[dict],
    submissions: Mapping[tuple[str, int], SubmissionRecord],
    draft: TaskDraft,
    grader: Grader,
) -> None:
    for role in (SearchRole.FALSE_ACCEPT, SearchRole.FALSE_REJECT):
        decisive = sorted(
            (c for c in claims if c["role"] == role and c["needs_addressing"]), key=lambda c: c["candidate_sha256"]
        )
        for claim in decisive[: judge.max_claims_per_component]:
            submission = submissions[(claim["trial"], claim["ordinal"])]
            claim["judge"] = await _judge_one(judge, judge_messages(judge.prompt, draft, grader, claim, submission))
            if claim["judge"].get("verdict") == Verdict.FINDER_CORRECT:
                break


def read_taxonomy(path: Path) -> dict[str, Mapping]:
    return json.loads(path.read_text())["classes"]
