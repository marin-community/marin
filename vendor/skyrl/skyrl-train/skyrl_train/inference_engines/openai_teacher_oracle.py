"""Teacher scoring through the OpenAI completions protocol and vLLM extensions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import aiohttp

from marinskyrl.distillation import OpenAICompatibleTeacherSpec, TeacherEndpointSpec, TeacherEvidenceKind
from skyrl_train.distillation import TeacherEvidenceBatch, TeacherScoreRequest
from skyrl_train.inference_engines.vllm_teacher_oracle import teacher_evidence_from_prompt_logprobs
from skyrl_train.teacher_oracle import TeacherCapabilities, TeacherEndpointUnavailable

_TOKEN_ID_PREFIX = "token_id:"
_MAX_ERROR_BODY_LENGTH = 2048


def _token_id(value: object, *, row: int, position: int) -> int:
    if not isinstance(value, str) or not value.startswith(_TOKEN_ID_PREFIX):
        raise ValueError(
            "remote teacher must return token IDs via return_tokens_as_token_ids; "
            f"row {row}, position {position} returned {value!r}"
        )
    try:
        return int(value.removeprefix(_TOKEN_ID_PREFIX))
    except ValueError as error:
        raise ValueError(f"remote teacher returned an invalid token ID at row {row}, position {position}") from error


def _float_score(value: object, *, row: int, position: int) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"remote teacher returned a non-numeric logprob at row {row}, position {position}")
    return float(value)


def _choice_prompt_logprobs(
    choice: Mapping[str, object], expected_tokens: Sequence[int], row: int
) -> list[dict[int, float] | None]:
    raw_logprobs = choice.get("logprobs")
    if not isinstance(raw_logprobs, Mapping):
        raise ValueError(f"remote teacher returned no completion logprobs for row {row}")
    raw_tokens = raw_logprobs.get("tokens")
    raw_chosen_scores = raw_logprobs.get("token_logprobs")
    raw_top_scores = raw_logprobs.get("top_logprobs")
    if (
        not isinstance(raw_tokens, list)
        or not isinstance(raw_chosen_scores, list)
        or not isinstance(raw_top_scores, list)
    ):
        raise ValueError(f"remote teacher returned an incomplete completion-logprob payload for row {row}")
    if not (len(raw_tokens) == len(raw_chosen_scores) == len(raw_top_scores) == len(expected_tokens)):
        raise ValueError(f"remote teacher returned misaligned completion logprobs for row {row}")

    prompt_logprobs: list[dict[int, float] | None] = []
    for position, (raw_token, raw_chosen, raw_top, expected_token) in enumerate(
        zip(raw_tokens, raw_chosen_scores, raw_top_scores, expected_tokens, strict=True)
    ):
        token_id = _token_id(raw_token, row=row, position=position)
        if token_id != expected_token:
            raise ValueError(
                f"remote teacher changed token identity at row {row}, position {position}: "
                f"expected {expected_token}, got {token_id}"
            )
        if raw_chosen is None:
            if position != 0:
                raise ValueError(f"remote teacher omitted the chosen-token logprob at row {row}, position {position}")
            prompt_logprobs.append(None)
            continue
        if raw_top is None:
            scores: dict[int, float] = {}
        elif isinstance(raw_top, Mapping):
            scores = {
                _token_id(candidate, row=row, position=position): _float_score(score, row=row, position=position)
                for candidate, score in raw_top.items()
            }
        else:
            raise ValueError(f"remote teacher returned invalid top logprobs at row {row}, position {position}")
        scores[token_id] = _float_score(raw_chosen, row=row, position=position)
        prompt_logprobs.append(scores)
    return prompt_logprobs


def _response_prompt_logprobs(
    payload: object, full_sequences: Sequence[Sequence[int]]
) -> list[list[dict[int, float] | None]]:
    if not isinstance(payload, Mapping) or not isinstance(payload.get("choices"), list):
        raise TeacherEndpointUnavailable("remote teacher returned an invalid completions response")
    raw_choices = payload["choices"]
    choices_by_index: dict[int, Mapping[str, object]] = {}
    for raw_choice in raw_choices:
        if not isinstance(raw_choice, Mapping):
            raise TeacherEndpointUnavailable("remote teacher returned an invalid completion choice")
        index = raw_choice.get("index")
        if isinstance(index, bool) or not isinstance(index, int) or index in choices_by_index:
            raise TeacherEndpointUnavailable("remote teacher returned invalid completion-choice indices")
        choices_by_index[index] = raw_choice
    if set(choices_by_index) != set(range(len(full_sequences))):
        raise TeacherEndpointUnavailable("remote teacher did not return exactly one choice per request row")
    return [
        _choice_prompt_logprobs(choices_by_index[row], sequence, row) for row, sequence in enumerate(full_sequences)
    ]


class OpenAICompatibleTeacherOracle:
    """Score exact token sequences through a vLLM-compatible completions endpoint.

    The endpoint must accept token-ID prompts and the vLLM ``echo``,
    ``return_tokens_as_token_ids``, and prompt-logprob behavior. Generic hosted
    chat APIs do not satisfy this contract unless they expose those extensions.
    """

    def __init__(
        self,
        *,
        teacher: OpenAICompatibleTeacherSpec,
        endpoint: TeacherEndpointSpec,
        api_key: str | None,
    ) -> None:
        self.capabilities = TeacherCapabilities(
            teacher_id=teacher.id,
            teacher_revision=teacher.model.revision,
            tokenizer_fingerprint=teacher.tokenizer_fingerprint,
            evidence_kinds=frozenset({teacher.evidence}),
            max_sequence_length=teacher.max_sequence_length,
            supports_prompt_token_scoring=True,
            max_concurrency=endpoint.max_concurrency,
        )
        self._url = f"{endpoint.url}/completions"
        self._model = teacher.model.path
        self._timeout = aiohttp.ClientTimeout(total=teacher.request_timeout_seconds)
        self._headers = {"Content-Type": "application/json"}
        if api_key is not None:
            self._headers["Authorization"] = f"Bearer {api_key}"
        self._session: aiohttp.ClientSession | None = None
        self._closed = False

    async def score(self, request: TeacherScoreRequest) -> TeacherEvidenceBatch:
        if self._closed:
            raise RuntimeError("remote teacher oracle is closed")
        if request.evidence is TeacherEvidenceKind.STUDENT_SELECTED_TOPK:
            raise ValueError("OpenAI-compatible prompt top-K cannot score arbitrary student-selected token IDs")
        full_sequences = [
            request.prompt_token_ids[row][request.prompt_mask[row]].tolist()
            + request.response_token_ids[row][request.response_mask[row]].tolist()
            for row in range(len(request.trajectory_ids))
        ]
        prompt_lengths = [int(mask.sum().item()) for mask in request.prompt_mask]
        requested_top_k = request.top_k if request.top_k is not None else 1
        body = {
            "model": self._model,
            "prompt": full_sequences,
            "max_tokens": 0,
            "echo": True,
            "logprobs": requested_top_k,
            "temperature": 0,
            "return_tokens_as_token_ids": True,
        }
        if self._session is None:
            self._session = aiohttp.ClientSession(timeout=self._timeout)
        try:
            async with self._session.post(self._url, json=body, headers=self._headers) as response:
                response_text = await response.text()
                if response.status >= 400:
                    detail = response_text[:_MAX_ERROR_BODY_LENGTH]
                    raise TeacherEndpointUnavailable(
                        f"remote teacher {request.teacher_id!r} returned HTTP {response.status}: {detail}"
                    )
                try:
                    payload = await response.json(content_type=None)
                except ValueError as error:
                    raise TeacherEndpointUnavailable(
                        f"remote teacher {request.teacher_id!r} returned a non-JSON response"
                    ) from error
        except (TimeoutError, aiohttp.ClientError) as error:
            raise TeacherEndpointUnavailable(f"remote teacher {request.teacher_id!r} request failed") from error

        prompt_logprobs = _response_prompt_logprobs(payload, full_sequences)
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
        if self._session is not None:
            await self._session.close()
