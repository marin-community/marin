"""Teacher-oracle adapter for fixed vLLM inference engines."""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Sequence
from typing import cast

import ray
import torch
from transformers import PreTrainedTokenizerBase

from marinskyrl.distillation import TeacherEvidenceKind
from skyrl_train.distillation import (
    ChosenTokenTeacherEvidence,
    INVALID_TOPK_INDEX,
    StudentSelectedTeacherEvidence,
    TeacherEvidenceBatch,
    TeacherScoreRequest,
    TopKTeacherEvidence,
)
from skyrl_train.inference_engines.base import InferenceEngineInterface
from skyrl_train.teacher_oracle import TeacherCapabilities, TeacherEndpointUnavailable


PromptLogprobs = Sequence[Sequence[dict[int, float] | None]]


def tokenizer_vocabulary_fingerprint(tokenizer: PreTrainedTokenizerBase) -> str:
    """Hash token-ID semantics without coupling identity to a model or chat template."""
    vocabulary = tokenizer.get_vocab()
    ordered_tokens: list[str | None] = [None] * (max(vocabulary.values(), default=-1) + 1)
    for token, token_id in vocabulary.items():
        if token_id < 0 or token_id >= len(ordered_tokens) or ordered_tokens[token_id] is not None:
            raise ValueError("tokenizer vocabulary must map each non-negative token ID to exactly one token")
        ordered_tokens[token_id] = token
    if any(token is None for token in ordered_tokens):
        raise ValueError("tokenizer vocabulary token IDs must be contiguous")
    encoded = json.dumps(ordered_tokens, ensure_ascii=False, separators=(",", ":")).encode()
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


async def close_owned_inference_engine(engine: InferenceEngineInterface) -> None:
    """Tear down an inference engine and force-kill its Ray actor if one exists."""
    try:
        await engine.teardown()
    finally:
        actor = getattr(engine, "inference_engine_actor", None)
        if actor is not None:
            await asyncio.to_thread(ray.kill, actor, no_restart=True)


def _position_scores(prompt_logprobs: PromptLogprobs, row: int, position: int) -> dict[int, float]:
    if position >= len(prompt_logprobs[row]) or prompt_logprobs[row][position] is None:
        raise ValueError(f"teacher returned no prompt logprobs at row {row}, token position {position}")
    return cast(dict[int, float], prompt_logprobs[row][position])


def teacher_evidence_from_prompt_logprobs(
    request: TeacherScoreRequest,
    *,
    teacher_revision: str,
    prompt_lengths: Sequence[int],
    prompt_logprobs: PromptLogprobs,
) -> TeacherEvidenceBatch:
    """Normalize prompt scores from local or remote vLLM into shared evidence."""
    if request.evidence is TeacherEvidenceKind.STUDENT_SELECTED_TOPK:
        assert request.student_topk_indices is not None
        assert request.student_selected_mask is not None
        selected_scores = torch.full(request.student_topk_indices.shape, torch.nan, dtype=torch.float32)
        for row, prompt_length in enumerate(prompt_lengths):
            response_length = int(request.response_mask[row].sum().item())
            selected_mask = request.student_selected_mask[row, :response_length].tolist()
            candidate_ids_by_offset = request.student_topk_indices[row, :response_length].tolist()
            selected_offsets = []
            row_scores = []
            for offset, (is_selected, token_ids) in enumerate(zip(selected_mask, candidate_ids_by_offset, strict=True)):
                if not is_selected:
                    continue
                scores = _position_scores(prompt_logprobs, row, prompt_length + offset)
                try:
                    row_scores.append([scores[token_id] for token_id in token_ids])
                except KeyError as error:
                    raise ValueError(
                        f"teacher omitted student-selected token {error.args[0]} at row {row}, response offset {offset}"
                    ) from error
                selected_offsets.append(offset)
            if selected_offsets:
                selected_scores[row, selected_offsets] = torch.tensor(row_scores, dtype=torch.float32)
        return StudentSelectedTeacherEvidence(
            trajectory_ids=request.trajectory_ids,
            route_ids=request.route_ids,
            teacher_id=request.teacher_id,
            teacher_revision=teacher_revision,
            plan_version=request.plan_version,
            valid_mask=request.student_selected_mask.clone(),
            student_topk_indices=request.student_topk_indices.clone(),
            teacher_on_student_logprobs=selected_scores,
        )
    if request.evidence is TeacherEvidenceKind.CHOSEN_TOKEN:
        chosen = torch.full(request.response_mask.shape, torch.nan, dtype=torch.float32)
        for row, prompt_length in enumerate(prompt_lengths):
            for offset in range(int(request.response_mask[row].sum().item())):
                token_id = int(request.response_token_ids[row, offset].item())
                scores = _position_scores(prompt_logprobs, row, prompt_length + offset)
                if token_id not in scores:
                    raise ValueError(f"teacher omitted chosen token {token_id} at row {row}, response offset {offset}")
                chosen[row, offset] = scores[token_id]
        return ChosenTokenTeacherEvidence(
            trajectory_ids=request.trajectory_ids,
            route_ids=request.route_ids,
            teacher_id=request.teacher_id,
            teacher_revision=teacher_revision,
            plan_version=request.plan_version,
            valid_mask=request.response_mask.clone(),
            chosen_logprobs=chosen,
        )

    top_k = request.top_k
    assert top_k is not None
    shape = (*request.response_mask.shape, top_k)
    indices = torch.full(shape, INVALID_TOPK_INDEX, dtype=torch.long)
    logprobs = torch.full(shape, torch.nan, dtype=torch.float32)
    retained_mass = torch.full(request.response_mask.shape, torch.nan, dtype=torch.float32)
    for row, prompt_length in enumerate(prompt_lengths):
        for offset in range(int(request.response_mask[row].sum().item())):
            scores = _position_scores(prompt_logprobs, row, prompt_length + offset)
            selected = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:top_k]
            if len(selected) != top_k:
                raise ValueError(
                    f"teacher returned {len(selected)} scores at row {row}, response offset {offset}; expected {top_k}"
                )
            token_ids, token_logprobs = zip(*selected, strict=True)
            indices[row, offset] = torch.tensor(token_ids)
            logprobs[row, offset] = torch.tensor(token_logprobs)
            retained_mass[row, offset] = logprobs[row, offset].exp().sum()
    return TopKTeacherEvidence(
        trajectory_ids=request.trajectory_ids,
        route_ids=request.route_ids,
        teacher_id=request.teacher_id,
        teacher_revision=teacher_revision,
        plan_version=request.plan_version,
        valid_mask=request.response_mask.clone(),
        topk_indices=indices,
        topk_logprobs=logprobs,
        retained_mass=retained_mass,
    )


class VLLMTeacherOracle:
    """Score exact student token sequences through one fixed vLLM engine."""

    def __init__(
        self,
        engine: InferenceEngineInterface,
        *,
        teacher_id: str,
        teacher_revision: str,
        tokenizer: PreTrainedTokenizerBase,
        evidence_kind: TeacherEvidenceKind,
        max_concurrency: int = 1,
    ) -> None:
        max_model_len = engine.get_model_max_len()
        if max_model_len is None or max_model_len <= 1:
            raise ValueError("vLLM teacher must expose a resolved maximum model length")
        self.capabilities = TeacherCapabilities(
            teacher_id=teacher_id,
            teacher_revision=teacher_revision,
            tokenizer_fingerprint=tokenizer_vocabulary_fingerprint(tokenizer),
            evidence_kinds=frozenset({evidence_kind}),
            max_sequence_length=max_model_len - 1,
            supports_prompt_token_scoring=True,
            max_concurrency=max_concurrency,
        )
        self._engine = engine
        self._closed = False

    async def score(self, request: TeacherScoreRequest) -> TeacherEvidenceBatch:
        if self._closed:
            raise RuntimeError("vLLM teacher oracle is closed")
        full_sequences = [
            request.prompt_token_ids[row][request.prompt_mask[row]].tolist()
            + request.response_token_ids[row][request.response_mask[row]].tolist()
            for row in range(len(request.trajectory_ids))
        ]
        prompt_lengths = [int(mask.sum().item()) for mask in request.prompt_mask]
        requested_top_k = request.top_k if request.evidence is not TeacherEvidenceKind.CHOSEN_TOKEN else 1
        assert requested_top_k is not None
        per_prompt_sampling = None
        if request.evidence is TeacherEvidenceKind.STUDENT_SELECTED_TOPK:
            assert request.student_topk_indices is not None
            assert request.student_selected_mask is not None
            per_prompt_sampling = []
            for row, sequence in enumerate(full_sequences):
                candidate_rows = [[token_id] * requested_top_k for token_id in sequence]
                for offset in range(int(request.response_mask[row].sum().item())):
                    if request.student_selected_mask[row, offset]:
                        candidate_rows[prompt_lengths[row] + offset] = request.student_topk_indices[
                            row, offset
                        ].tolist()
                per_prompt_sampling.append({"prompt_logprob_token_ids": candidate_rows})
        engine_input = {
            "prompts": None,
            "prompt_token_ids": full_sequences,
            "sampling_params": {
                "max_tokens": 1,
                "prompt_logprobs": requested_top_k,
                "temperature": 1.0,
            },
            "session_ids": None,
        }
        if per_prompt_sampling is not None:
            engine_input["sampling_params_per_prompt"] = per_prompt_sampling
        try:
            output = await self._engine.generate(engine_input)
        except Exception as error:
            raise TeacherEndpointUnavailable(f"vLLM teacher {request.teacher_id!r} scoring failed") from error
        prompt_logprobs = output.get("prompt_logprobs")
        if prompt_logprobs is None or len(prompt_logprobs) != len(full_sequences):
            raise ValueError("vLLM teacher did not return one prompt-logprob sequence per request row")
        return teacher_evidence_from_prompt_logprobs(
            request,
            teacher_revision=self.capabilities.teacher_revision,
            prompt_lengths=prompt_lengths,
            prompt_logprobs=prompt_logprobs,
        )

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        await close_owned_inference_engine(self._engine)
