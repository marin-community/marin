# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Explicit judge evidence, model policies, and infrastructure failure outcomes."""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import openai
from tasktrove_verify.modes.grade_ifeval import resolve_checks
from tasktrove_verify.modes.grade_judge import CHECKLIST_PROMPT, REFERENCE_PROMPT, normalize
from tasktrove_verify.spec import RUBRIC_CHECKLIST, RUBRIC_REFERENCE, JudgeSpec, Spec

from taskcompendium.grading_paths import submission_relative
from taskcompendium.models import GradingResult, JudgeModelPolicy, Outcome, TaskSpecification, TaskTroveVerifier
from taskcompendium.resources import contained_path

_SCORE = re.compile(r"(?:^|\n)SCORE: (0|0\.5|1)\s*\Z")


@dataclass(frozen=True)
class JudgeReply:
    text: str
    model: str
    revision: str | None = None


class JudgeClient(Protocol):
    def complete(self, prompt: str, policy: JudgeModelPolicy, timeout: float) -> JudgeReply: ...


class OpenAIJudgeClient:
    """Call an explicitly configured endpoint; credentials stay outside task data."""

    def __init__(self, api_key: str):
        self.api_key = api_key

    def complete(self, prompt: str, policy: JudgeModelPolicy, timeout: float) -> JudgeReply:
        with openai.OpenAI(base_url=policy.base_url, api_key=self.api_key) as client:
            response = client.chat.completions.create(
                model=policy.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=policy.temperature,
                timeout=timeout,
            )
        return JudgeReply(response.choices[0].message.content or "", response.model, response.system_fingerprint)


def grade_judge_attempt(
    specification: TaskSpecification,
    contract: Spec,
    candidate: str,
    workspace: Path,
    tests: Path,
    transcript: tuple[dict[str, Any], ...],
    client: JudgeClient | None,
    step_index: int = 0,
) -> GradingResult:
    verifier = specification.steps[step_index].verifier
    if not isinstance(verifier, TaskTroveVerifier):
        raise ValueError("Model judging requires a TaskTrove judge verifier")
    config = verifier.judge
    assert config is not None and isinstance(contract, JudgeSpec)
    references = tuple(reference for reference in contract.references if reference.strip())
    criteria = tuple(criterion for criterion in contract.criteria if criterion.strip())
    if contract.rubric not in {RUBRIC_REFERENCE, RUBRIC_CHECKLIST}:
        return GradingResult(Outcome.INVALID_TASK, None, {"error": "Unknown judge rubric"})
    if (contract.rubric == RUBRIC_REFERENCE and not references) or (
        contract.rubric == RUBRIC_CHECKLIST and not criteria
    ):
        return GradingResult(Outcome.INVALID_TASK, None, {"error": "Empty judge rubric"})
    try:
        failed = [
            constraint.name
            for constraint, check in (resolve_checks(contract.constraints) if contract.constraints else [])
            if not check(candidate, constraint.params)[0]
        ]
    except Exception as error:
        return GradingResult(Outcome.INVALID_TASK, None, {"error": f"Invalid judge constraint: {error}"})
    if failed:
        return GradingResult(Outcome.GRADED, 0.0, {"gate": "constraints", "failed": failed})
    if (
        contract.rubric == RUBRIC_REFERENCE
        and contract.exact_gate
        and normalize(candidate) in {normalize(r) for r in references}
    ):
        return GradingResult(Outcome.GRADED, 1.0, {"gate": "exact"})
    if client is None:
        return GradingResult(Outcome.INFRA_ERROR, None, {"error": "No judge client configured"})
    evidence: dict[str, Any] = {"answer": candidate}
    if config.view.transcript:
        evidence["transcript"] = transcript
    evidence["files"] = {
        path: contained_path(workspace, submission_relative(path)).read_text() for path in config.view.files
    }
    contexts = {path: contained_path(tests, path).read_text() for path in config.view.reference_context}
    if contract.context and contract.context not in contexts:
        return GradingResult(Outcome.INVALID_TASK, None, {"error": "Source judge context is absent from JudgeView"})
    candidate_view = candidate if not config.view.transcript and not config.view.files else json.dumps(evidence)
    context_text = json.dumps(contexts) if contexts else ""
    question = f"\nQuestion:\n{contract.question}\n" if contract.question else ""
    prompts = [
        REFERENCE_PROMPT.format(
            question=question, references="\n".join(f"- {r}" for r in references), candidate=candidate_view
        )
    ]
    if contract.rubric == RUBRIC_CHECKLIST:
        prompts = [
            CHECKLIST_PROMPT.format(context=context_text, question=question, candidate=candidate_view, criterion=c)
            for c in criteria
        ]
    elif context_text:
        prompts = [f"Reference context:\n{context_text}\n\n{prompts[0]}"]
    replies: list[dict[str, Any]] = []
    try:
        for sample in range(config.policy.samples):
            for criterion, prompt in enumerate(prompts):
                reply = client.complete(prompt, config.policy, contract.request_timeout)
                match = _SCORE.search(reply.text)
                if match is None or (contract.rubric == RUBRIC_CHECKLIST and match[1] == "0.5"):
                    return GradingResult(
                        Outcome.INFRA_ERROR,
                        None,
                        {"error": "Malformed judge verdict", "reply": reply.text, "model": reply.model},
                    )
                replies.append(
                    {
                        "sample": sample,
                        "criterion": criterion,
                        "score": float(match[1]),
                        "reason": reply.text[: match.start()].strip(),
                        "model": reply.model,
                        "revision": reply.revision,
                    }
                )
    except Exception as error:
        return GradingResult(Outcome.INFRA_ERROR, None, {"error": f"Judge request failed: {error}"})
    reward = sum(reply["score"] for reply in replies) / len(replies)
    return GradingResult(
        Outcome.GRADED,
        reward,
        {
            "provider": config.policy.provider,
            "requested_model": config.policy.model,
            "temperature": config.policy.temperature,
            "judgments": replies,
        },
    )
