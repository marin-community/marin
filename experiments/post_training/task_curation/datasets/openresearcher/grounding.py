# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Execute the sealed screen against a verified local index and an explicitly pinned judge."""

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, ClassVar, TypeVar, cast

import httpx
from pydantic import BaseModel, ValidationError
from rigging.filesystem.storage_path import StoragePath
from transformers import PreTrainedTokenizerBase
from verifyit.modes.grade_judge import JudgeConnection

from experiments.post_training.task_curation.datasets.browsecomp.scorers.judge_parser import parse_judge_response
from experiments.post_training.task_curation.datasets.browsecomp.scorers.prompts import GRADER_TEMPLATE
from experiments.post_training.task_curation.datasets.openresearcher.grounding_contract import (
    ANSWER,
    AUDIT,
    CONTRACT_SHA256,
    DECOMPOSE,
    GROUND,
    MAX_COMPLETION_TOKENS,
    MAX_DOCUMENT_TOKENS,
    MAX_INPUT_TOKENS,
    MAX_REQUIREMENTS,
    PARSE_ATTEMPTS,
    SYSTEM,
    TEMPERATURE,
    TOP_K,
    TOP_P,
    CompleteAudit,
    Decomposition,
    EvidenceAnswer,
    GroundedRequirement,
    audit_passes,
    verified_quotes,
)

Model = TypeVar("Model", bound=BaseModel)
Retrieve = Callable[[tuple[str, str]], Awaitable[dict[str, str]]]
logger = logging.getLogger(__name__)


class GroundingBudgetExceeded(ValueError):
    """A complete evidence packet cannot fit the sealed screen's context budget."""

    reason: ClassVar[str] = "complete_evidence_exceeds_sealed_context_budget"


class GroundingRequirementsBudgetExceeded(GroundingBudgetExceeded):
    """The complete decomposition exceeds the sealed requirement budget."""

    reason: ClassVar[str] = "complete_requirements_exceed_sealed_budget"


class GroundingCompletionIncomplete(ValueError):
    """The model exhausted its output budget before completing a response."""


class GroundingIncompleteExhausted(ValueError):
    """A structured request exhausted its attempts with an incomplete response."""

    reason: ClassVar[str] = "grounding_incomplete_exhausted"

    def __init__(self, request_name: str):
        self.request_name = request_name
        super().__init__(f"{request_name}: incomplete response after {PARSE_ATTEMPTS} attempts")


def validate_technical_exclusions(decisions: list[dict]) -> None:
    """Require explicit drop decisions and cap incomplete exhaustion at one percent."""
    exclusions = []
    for decision in decisions:
        technical = decision["status"] == "technical_exclusion"
        incomplete = decision["reason"] == GroundingIncompleteExhausted.reason
        if technical != incomplete or (technical and decision["keep"] is not False):
            raise ValueError("Incomplete grounding must be an explicit technical exclusion")
        if technical:
            exclusions.append(decision)
    if len(exclusions) * 100 > len(decisions):
        raise ValueError(f"Incomplete grounding exclusions exceed one percent: {len(exclusions)}/{len(decisions)}")


@dataclass(frozen=True)
class GroundingJudge:
    connection: JudgeConnection
    client: httpx.AsyncClient
    tokenizer: PreTrainedTokenizerBase
    observations: StoragePath

    async def complete(self, name: str, prompt: str, *, structured: bool) -> str:
        messages = [{"role": "user", "content": prompt}]
        if structured:
            messages.insert(0, {"role": "system", "content": SYSTEM})
        tokens = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_dict=False
        )
        if len(tokens) > MAX_INPUT_TOKENS:
            raise GroundingBudgetExceeded(f"{name}: complete prompt exceeds the sealed input budget")
        request: dict[str, Any] = {
            "model": self.connection.default_model,
            "messages": messages,
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "top_k": TOP_K,
            "max_tokens": MAX_COMPLETION_TOKENS,
        }
        if structured:
            request["response_format"] = {"type": "json_object"}
        response = await self.client.post(
            self.connection.base_url.rstrip("/") + "/chat/completions",
            headers={"Authorization": "Bearer " + self.connection.api_key},
            json=request,
        )
        if not response.is_success:
            (self.observations / (name + "-http-error.json")).write_text(
                json.dumps({"request": request, "status_code": response.status_code, "response_body": response.text})
            )
        response.raise_for_status()
        result = response.json()
        (self.observations / (name + ".json")).write_text(
            json.dumps({"request": request, "response": result}, ensure_ascii=False)
        )
        if result["model"] != self.connection.default_model:
            raise ValueError("Grounding response model differs from the explicitly pinned backend")
        choice = result["choices"][0]
        if choice["finish_reason"] == "length":
            raise GroundingCompletionIncomplete("Grounding completion did not finish: length")
        if choice["finish_reason"] != "stop":
            raise ValueError(f"Grounding completion did not finish: {choice['finish_reason']}")
        return choice["message"]["content"]

    async def structured(self, name: str, prompt: str, schema: type[Model]) -> Model:
        for attempt in range(PARSE_ATTEMPTS):
            try:
                text = await self.complete(f"{name}-{attempt}", prompt, structured=True)
                if schema is Decomposition:
                    decoded = json.loads(text)
                    requirements = decoded.get("requirements") if isinstance(decoded, dict) else None
                    if isinstance(requirements, list) and len(requirements) > MAX_REQUIREMENTS:
                        raise GroundingRequirementsBudgetExceeded(
                            f"Complete decomposition has {len(requirements)} requirements; budget is {MAX_REQUIREMENTS}"
                        )
                return schema.model_validate_json(text)
            except (ValidationError, json.JSONDecodeError, GroundingCompletionIncomplete) as error:
                (self.observations / f"{name}-{attempt}-error.txt").write_text(str(error))
                if attempt + 1 == PARSE_ATTEMPTS:
                    if isinstance(error, GroundingCompletionIncomplete):
                        raise GroundingIncompleteExhausted(name) from error
                    raise
        raise AssertionError("A positive parse-attempt budget must produce a result or an exception")


def evidence_documents(documents: dict[str, str], tokenizer: PreTrainedTokenizerBase) -> dict[str, str]:
    """Bound context conservatively; a lost supporting passage can only reduce retention."""
    bounded = {}
    for docid, text in documents.items():
        tokens = tokenizer.encode(text, add_special_tokens=False)
        bounded[docid] = (
            text
            if len(tokens) <= MAX_DOCUMENT_TOKENS
            else cast(str, tokenizer.decode(tokens[:MAX_DOCUMENT_TOKENS], skip_special_tokens=True))
        )
    return bounded


async def screen_question(row: dict[str, str], judge: GroundingJudge, retrieve: Retrieve) -> dict:
    packet = {"question": row["question"], "proposed_answer": row["answer"]}
    decomposition = await judge.structured("decompose", DECOMPOSE + json.dumps(packet), Decomposition)
    result = {
        "id": row["id"],
        "contract_sha256": CONTRACT_SHA256,
        "keep": False,
        "status": "screened",
        "reason": "requirements_not_decomposable",
        "decomposition": decomposition.model_dump(mode="json"),
        "grounded_requirements": [],
    }
    requirements = decomposition.requirements
    if not decomposition.decomposable or not requirements:
        return result
    if len(requirements) > MAX_REQUIREMENTS:
        return {**result, "reason": "complete_requirements_exceed_sealed_budget"}
    if [requirement.id for requirement in requirements] != [f"h{index + 1}" for index in range(len(requirements))]:
        raise ValueError("Decomposition does not identify every requirement exactly once")
    quotations = []
    for requirement in requirements:
        raw_documents = await retrieve(requirement.queries)
        documents = evidence_documents(raw_documents, judge.tokenizer)
        prompt = GROUND + json.dumps({**packet, "requirement": requirement.model_dump(), "documents": documents})
        grounded = await judge.structured("ground-" + requirement.id, prompt, GroundedRequirement)
        result["grounded_requirements"].append({"id": requirement.id, **grounded.model_dump(mode="json")})
        if not verified_quotes(grounded, documents):
            return {**result, "reason": "required_hop_not_grounded_or_citation_not_literal"}
        # Check the original corpus body too, since tokenizer decode can normalize text.
        if not verified_quotes(grounded, raw_documents):
            return {**result, "reason": "citation_differs_from_original_corpus_bytes"}
        quotations.extend(evidence.model_dump() for evidence in grounded.evidence)
    # The independent solve never receives the reference or the gold-derived claim wording.
    answer = await judge.structured(
        "answer", ANSWER + json.dumps({"question": row["question"], "quotations": quotations}), EvidenceAnswer
    )
    result["evidence_answer"] = answer.model_dump(mode="json")
    if not answer.answerable:
        return {**result, "reason": "independent_evidence_only_answer_not_possible"}
    match = parse_judge_response(
        await judge.complete(
            "answer-match",
            GRADER_TEMPLATE.format(question=row["question"], response=answer.answer, correct_answer=row["answer"]),
            structured=False,
        )
    )
    result["answer_match"] = match
    if match["parse_error"]:
        raise ValueError("Original answer-equivalence parser cannot determine the grounding answer match")
    if not match["correct"]:
        return {**result, "reason": "independent_evidence_answer_differs_from_gold"}
    audit = await judge.structured(
        "audit",
        AUDIT + json.dumps({**packet, "requirements": decomposition.model_dump(), "quotations": quotations}),
        CompleteAudit,
    )
    result["audit"] = audit.model_dump(mode="json")
    return {
        **result,
        "keep": audit_passes(audit),
        "reason": "pass" if audit_passes(audit) else "independent_audit_failed",
    }


async def screen_population(
    rows: list[dict[str, str]],
    connection: JudgeConnection,
    tokenizer: PreTrainedTokenizerBase,
    retrieve: Retrieve,
    output: StoragePath,
    concurrency: int,
) -> list[dict]:
    semaphore = asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient(timeout=300) as client:

        async def one(row: dict[str, str]) -> dict:
            async with semaphore:
                target = output / "decisions" / (row["id"] + ".json")
                judge = GroundingJudge(connection, client, tokenizer, output / "observations" / row["id"])
                try:
                    result = await screen_question(row, judge, retrieve)
                except GroundingIncompleteExhausted as error:
                    result = {
                        "id": row["id"],
                        "contract_sha256": CONTRACT_SHA256,
                        "keep": False,
                        "status": "technical_exclusion",
                        "reason": error.reason,
                        "detail": str(error),
                        "observations": [
                            str(judge.observations / f"{error.request_name}-{attempt}.json")
                            for attempt in range(PARSE_ATTEMPTS)
                        ],
                    }
                except GroundingBudgetExceeded as error:
                    result = {
                        "id": row["id"],
                        "contract_sha256": CONTRACT_SHA256,
                        "keep": False,
                        "status": "screened",
                        "reason": error.reason,
                        "detail": str(error),
                    }
                except Exception as error:
                    # Preserve failed execution as failed execution; never count it as a content rejection.
                    target.write_text(json.dumps({"id": row["id"], "status": "execution_failure", "error": repr(error)}))
                    raise
                target.write_text(json.dumps(result, ensure_ascii=False))
                logger.info("Grounding decision %s: keep=%s reason=%s", row["id"], result["keep"], result["reason"])
                return result

        async with asyncio.TaskGroup() as group:
            tasks = [group.create_task(one(row)) for row in rows]
        decisions = [task.result() for task in tasks]
        validate_technical_exclusions(decisions)
        return decisions
