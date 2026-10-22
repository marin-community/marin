"""Model transports used by trajectory runners."""

import asyncio
from copy import deepcopy
from dataclasses import dataclass
from collections.abc import Awaitable, Callable
from typing import Any, NotRequired, Protocol
from uuid import uuid4

import numpy as np
from rolloutengine.models import RolloutContractError

from skyrl_gym.envs.nemotron_ultra.answer_extraction import REASONING_DELIMITERS, final_answer_text

from skyrl_train.inference_engines.base import (
    CHOICE_POLICY_STEPS_KEY,
    SESSION_ID_HEADER,
    ChatContinuation,
    InferenceEngineInput,
    InferenceEngineOutput,
)
from skyrl_train.inference_engines.chat_continuation import (
    CHAT_TOKENIZE_FIELDS,
    EXACT_PROMPT_TOKEN_IDS_KEY,
    render_exact_chat_continuation,
)
from skyrl_train.inference_engines.chat_template import (
    SINGLE_TOOL_CALL_TEMPLATE_ERROR,
    sequentialize_multi_tool_call_turns,
    template_error_from_exception,
)
from skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient
from skyrl_train.inference_engines.chat_template import assistant_prefix_token_ids
from skyrl_train.inference_engines.response_topk import select_chat_response_topk
from skyrl_train.trajectory_runners.types import TokenProvenance
from skyrl_train.trajectory_runners.routed_experts import normalize_routed_experts
from skyrl_train.rollout_observability import rollout_phase


# The per-turn output limit, as the trainer config (`max_generate_length`) and vLLM (`max_tokens`) spell it. Chat
# requests carry it only as `max_completion_tokens`, which vLLM prefers over `max_tokens`.
_OUTPUT_LIMIT_KEYS = ("max_generate_length", "max_tokens")
_CHAT_SAMPLING_EXCLUSIONS = frozenset({*_OUTPUT_LIMIT_KEYS, "logprobs", "stop"})


class ModelClientOutput(InferenceEngineOutput):
    """Normalized model output with explicit token provenance."""

    token_provenance: TokenProvenance
    generation_token_budgets: NotRequired[list[int | None]]


class ModelClient(Protocol):
    """Transport-neutral model request boundary for trajectory runners."""

    async def generate(self, request: InferenceEngineInput) -> ModelClientOutput: ...


class ModelServerError(RuntimeError):
    """A model-serving failure with safe diagnostics for retained trajectories."""

    def __init__(self, category: str, request_id: str | None, status_code: int | None):
        self.category = category
        self.request_id = request_id
        self.status_code = status_code
        super().__init__(f"Model server error: {category}; request_id={request_id}")


class ContextLengthExceededError(ModelServerError):
    """A serving rejection caused by an overlong model context."""


class GenerationBudgetExceededError(ContextLengthExceededError):
    """A local token limit stopped the request before inference."""

    def __init__(self, prompt_token_ids: list[int]):
        self.prompt_token_ids = tuple(prompt_token_ids)
        super().__init__(category="context_overflow", request_id=None, status_code=None)


def _generation_token_budget(
    request: InferenceEngineInput, prompt_ids: list[int], requested_tokens: list[int]
) -> int | None:
    max_prompt_length = request.get("max_prompt_length")
    if max_prompt_length is not None and len(prompt_ids) > max_prompt_length:
        raise GenerationBudgetExceededError(prompt_ids)
    sampling = request.get("sampling_params") or {}
    configured = [int(sampling[key]) for key in _OUTPUT_LIMIT_KEYS if sampling.get(key) is not None]
    configured.extend(requested_tokens)
    max_context_length = request.get("max_context_length")
    if max_context_length is not None:
        remaining = max_context_length - len(prompt_ids)
        if remaining <= 0:
            raise GenerationBudgetExceededError(prompt_ids)
        configured.append(remaining)
    return min(configured) if configured else None


@dataclass(frozen=True)
class _ChatResult:
    prompt_ids: list[int]
    response_ids: list[int]
    response_logprobs: list[float] | None
    student_topk_indices: list[list[int]] | None
    behavior_topk_logprobs: list[list[float]] | None
    text: str
    stop_reason: str
    assistant_message: dict[str, Any]
    generation_token_budget: int | None
    routed_experts: np.ndarray | None = None
    policy_steps: list[int] | None = None


def _choice_routed_experts(choice: dict[str, Any], prompt_ids: list[int], response_ids: list[int]) -> np.ndarray | None:
    provider_fields = choice.get("provider_specific_fields") or {}
    routes = choice.get("routed_experts", provider_fields.get("routed_experts"))
    if routes is None:
        return None
    return normalize_routed_experts(routes, prompt_ids, response_ids)


def _assemble_chat_results(results: list[_ChatResult]) -> ModelClientOutput:
    logprobs = [result.response_logprobs for result in results]
    selected_indices = [result.student_topk_indices for result in results]
    selected_scores = [result.behavior_topk_logprobs for result in results]
    output = ModelClientOutput(
        prompt_ids=[result.prompt_ids for result in results],
        response_ids=[result.response_ids for result in results],
        response_logprobs=logprobs if all(value is not None for value in logprobs) else None,
        responses=[result.text for result in results],
        stop_reasons=[result.stop_reason for result in results],
        prompt_logprobs=None,
        assistant_messages=[result.assistant_message for result in results],
        token_provenance=TokenProvenance.ENGINE,
        generation_token_budgets=[result.generation_token_budget for result in results],
    )
    if all(rows is not None for rows in selected_indices):
        output["student_topk_indices"] = selected_indices
        output["behavior_topk_logprobs"] = selected_scores
    if any(result.routed_experts is not None for result in results):
        output["routed_experts"] = [result.routed_experts for result in results]
    if all(result.policy_steps is not None for result in results):
        output["response_policy_steps"] = [result.policy_steps for result in results]
    return output


async def _render_chat_prompt(
    tokenize: Callable[[dict[str, Any]], Awaitable[dict[str, Any]]],
    render_request: dict[str, Any],
    continuation: ChatContinuation | None,
) -> tuple[list[dict[str, Any]], list[int]]:
    async def render(request: dict[str, Any], current_continuation: ChatContinuation | None) -> list[int] | None:
        if current_continuation is None:
            return (await tokenize(request)).get("tokens")
        return await render_exact_chat_continuation(
            tokenize,
            request,
            assistant_message_index=current_continuation["assistant_message_index"],
            served_prefix_token_ids=current_continuation["served_prefix_token_ids"],
        )

    messages = render_request["json"]["messages"]
    try:
        prompt_ids = await render(render_request, continuation)
    except Exception as error:
        template_error = template_error_from_exception(error)
        if template_error is None:
            raise
        if SINGLE_TOOL_CALL_TEMPLATE_ERROR not in str(template_error):
            raise template_error from error
        assistant_index = continuation["assistant_message_index"] if continuation is not None else None
        messages, assistant_index = sequentialize_multi_tool_call_turns(messages, assistant_index)
        if messages == render_request["json"]["messages"]:
            raise template_error from error
        render_request = deepcopy(render_request)
        render_request["json"]["messages"] = messages
        if continuation is not None:
            assert assistant_index is not None
            continuation = ChatContinuation(
                served_prefix_token_ids=continuation["served_prefix_token_ids"],
                assistant_message_index=assistant_index,
            )
        try:
            prompt_ids = await render(render_request, continuation)
        except Exception as retry_error:
            retry_template_error = template_error_from_exception(retry_error)
            if retry_template_error is not None:
                raise retry_template_error from retry_error
            raise
    if prompt_ids is None:
        raise RuntimeError("Cannot preserve the exact served token prefix across this chat turn")
    if not isinstance(prompt_ids, list) or not all(isinstance(token, int) for token in prompt_ids):
        raise RuntimeError("vLLM tokenization did not return prompt token IDs")
    return messages, prompt_ids


class DirectModelClient:
    """Delegate model requests to the colocated inference-engine client."""

    def __init__(self, inference_engine_client: InferenceEngineClient):
        self._client = inference_engine_client

    async def generate(self, request: InferenceEngineInput) -> ModelClientOutput:
        if request.get("chat_completion_params") is not None:
            return await self._generate_chat(request)
        output = await self._client.generate(request)
        return ModelClientOutput(**output, token_provenance=TokenProvenance.ENGINE)

    async def _generate_assistant_prefix(
        self, request: InferenceEngineInput, messages: list[dict[str, Any]], options: dict[str, Any], session_id: str
    ) -> _ChatResult:
        prompt_ids = assistant_prefix_token_ids(self._client.tokenizer, messages, options)
        sampling = dict(request.get("sampling_params") or {})
        requested_tokens = [
            int(options[key])
            for key in ("max_completion_tokens", "max_output_tokens")
            if options.get(key) is not None
        ]
        token_budget = _generation_token_budget(request, prompt_ids, requested_tokens)
        if token_budget is not None:
            sampling["max_tokens"] = token_budget
        sampling.pop("max_generate_length", None)
        sampling.update(skip_special_tokens=False, include_stop_str_in_output=True)
        # Raw completions retain delimiters and bypass chat tool/reasoning parsers.
        output = await self._client.generate(
            {
                "prompts": None,
                "prompt_token_ids": [prompt_ids],
                "session_ids": [session_id],
                "sampling_params": sampling,
            }
        )
        if output.get("prompt_ids") != [prompt_ids]:
            raise RolloutContractError("Assistant prefix inference did not report the rendered prompt token IDs")
        text = output["responses"][0]
        message = (output.get("assistant_messages") or [{"role": "assistant", "content": text}])[0]
        return _ChatResult(
            prompt_ids,
            output["response_ids"][0],
            None if output.get("response_logprobs") is None else output["response_logprobs"][0],
            None if output.get("student_topk_indices") is None else output["student_topk_indices"][0],
            None if output.get("behavior_topk_logprobs") is None else output["behavior_topk_logprobs"][0],
            text,
            output["stop_reasons"][0],
            message,
            token_budget,
            (output.get("routed_experts") or [None])[0],
            (output.get("response_policy_steps") or [None])[0],
        )

    @staticmethod
    def _chat_options(options: dict[str, Any], sampling_params: dict[str, Any]) -> dict[str, Any]:
        result = dict(options)
        result.pop("input", None)
        if not result.get("tools"):
            result.pop("tools", None)
        if "max_output_tokens" in result:
            result["max_completion_tokens"] = result.pop("max_output_tokens")
        for key in _OUTPUT_LIMIT_KEYS:
            if sampling_params.get(key) is None:
                continue
            configured_max = int(sampling_params[key])
            requested_max = result.get("max_completion_tokens")
            result["max_completion_tokens"] = (
                configured_max if requested_max is None else min(configured_max, int(requested_max))
            )
        return result

    async def _generate_chat(self, request: InferenceEngineInput) -> ModelClientOutput:
        prompts = request.get("prompts")
        options = request.get("chat_completion_params")
        if prompts is None or options is None or len(prompts) != len(options):
            raise ValueError("Chat generation requires aligned prompts and chat_completion_params")
        session_ids = request.get("session_ids") or [None] * len(prompts)
        if len(session_ids) != len(prompts):
            raise ValueError("session_ids and prompts must have the same batch size")
        continuations = request.get("chat_continuations") or [None] * len(prompts)
        if len(continuations) != len(prompts):
            raise ValueError("chat_continuations and prompts must have the same batch size")
        sampling_params = dict(request.get("sampling_params") or {})
        requested_top_k = sampling_params.get("logprobs")
        requested_top_k = requested_top_k if isinstance(requested_top_k, int) and requested_top_k > 0 else None

        async def generate_one(messages, row_options, session_id, continuation):
            if row_options.get("extra_body", {}).get("continue_final_message") is True:
                if continuation is not None:
                    raise ValueError("An initial assistant prefix cannot reuse a sampled chat continuation")
                return await self._generate_assistant_prefix(request, messages, row_options, session_id)
            chat_options = self._chat_options(row_options, sampling_params)
            tokenize_options = {key: value for key, value in chat_options.items() if key in CHAT_TOKENIZE_FIELDS}
            render_request = {
                "json": {
                    "model": self._client.model_name,
                    "messages": messages,
                    **tokenize_options,
                    "add_generation_prompt": True,
                },
                "headers": {},
            }
            with rollout_phase("tokenize"):
                messages, prompt_ids = await _render_chat_prompt(self._client.tokenize, render_request, continuation)
            requested_tokens = (
                [] if chat_options.get("max_completion_tokens") is None else [int(chat_options["max_completion_tokens"])]
            )
            token_budget = _generation_token_budget(request, prompt_ids, requested_tokens)
            if token_budget is not None:
                # Preserve chat's explicit-null handling when context supplies a budget.
                chat_options["max_completion_tokens"] = min(
                    int(chat_options.get("max_completion_tokens", token_budget)), token_budget
                )

            body = {
                "model": self._client.model_name,
                "messages": messages,
                "session_id": session_id,
                **{key: value for key, value in sampling_params.items() if key not in _CHAT_SAMPLING_EXCLUSIONS},
                **chat_options,
                "include_stop_str_in_output": True,
            }
            body[EXACT_PROMPT_TOKEN_IDS_KEY] = prompt_ids
            if sampling_params.get("stop") is not None:
                body["stop"] = sampling_params["stop"]
            if sampling_params.get("logprobs") is not None:
                body["logprobs"] = True
            if requested_top_k is not None:
                # vLLM may include the sampled token outside the natural top K.
                body["top_logprobs"] = requested_top_k + 1
                body["return_tokens_as_token_ids"] = True
            request_id = uuid4().hex
            headers = {"x-request-id": request_id}
            if session_id is not None:
                headers[SESSION_ID_HEADER] = str(session_id)
            response = await self._client.chat_completion({"json": body, "headers": headers})
            if "choices" not in response:
                error = response.get("error") or {}
                error_type = (
                    ContextLengthExceededError
                    if response.get("error_category") == "context_overflow"
                    else ModelServerError
                )
                raise error_type(
                    category=response.get("error_category", "server_error"),
                    request_id=response.get("request_id", request_id),
                    status_code=error.get("code") if isinstance(error, dict) else None,
                )
            choice = response["choices"][0]
            response_ids = choice.get("token_ids")
            if not isinstance(response_ids, list) or not all(isinstance(token, int) for token in response_ids):
                raise RuntimeError("vLLM chat completion did not return exact token IDs")
            message = choice["message"]
            text = message.get("content") or ""
            # Some serving configurations omit reasoning parsers and remove
            # special delimiters from content. Exact tokens retain the boundary.
            tokenizer = self._client.tokenizer
            semantic_ids = (
                response_ids[:-1] if response_ids and response_ids[-1] == tokenizer.eos_token_id else response_ids
            )
            if len(semantic_ids) != len(response_ids):
                eos_text = tokenizer.decode(response_ids[-1:], skip_special_tokens=False)
                text = text.removesuffix(eos_text)
                if message.get("content") is not None:
                    message = {**message, "content": text}
            decoded = tokenizer.decode(semantic_ids, skip_special_tokens=False)
            if any(marker in decoded for pair in REASONING_DELIMITERS for marker in pair):
                text = final_answer_text(decoded)
                if not message.get("tool_calls"):
                    message = {**message, "content": text}
            logprob_items = (choice.get("logprobs") or {}).get("content")
            response_logprobs = (
                [float(item["logprob"]) for item in logprob_items] if logprob_items is not None else None
            )
            selected = None
            if requested_top_k is not None and logprob_items is not None:
                if len(logprob_items) != len(response_ids):
                    raise ValueError("chat response top-K rows must align with exact token IDs")
                selected = [
                    select_chat_response_topk(item.get("top_logprobs") or [], requested_top_k) for item in logprob_items
                ]
            stop_reason = choice["finish_reason"]
            token_budget = chat_options.get("max_completion_tokens")
            if (
                stop_reason == "tool_calls"
                and token_budget is not None
                and len(response_ids) >= token_budget
                and choice.get("stop_reason") is None
                and response_ids[-1] != tokenizer.eos_token_id
            ):
                # A tool parser can replace the backend's output-budget stop reason.
                stop_reason = "length"
            return _ChatResult(
                prompt_ids,
                response_ids,
                response_logprobs,
                None if selected is None else [ids for ids, _ in selected],
                None if selected is None else [scores for _, scores in selected],
                text,
                stop_reason,
                message,
                token_budget,
                _choice_routed_experts(choice, prompt_ids, response_ids),
                choice.get(CHOICE_POLICY_STEPS_KEY),
            )

        results = await asyncio.gather(
            *(
                generate_one(messages, row_options, session_id, continuation)
                for messages, row_options, session_id, continuation in zip(prompts, options, session_ids, continuations)
            )
        )
        return _assemble_chat_results(results)
