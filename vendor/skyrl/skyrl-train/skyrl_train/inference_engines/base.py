from abc import ABC, abstractmethod
from typing import List, Dict, TypedDict, Any, Optional, Hashable, NotRequired

import numpy as np

MessageType = Dict[str, str]
ConversationType = List[MessageType]
OnlineEagleResult = Dict[str, Any] | List[Dict[str, Any]]
SESSION_ID_HEADER = "x-skyrl-session-id"
UNSTAMPED_POLICY_STEP = -1
# Chat-completion choice field with one policy step per token in ``token_ids``.
CHOICE_POLICY_STEPS_KEY = "policy_steps"
# The data-parallel rank of a remote engine.
UNKNOWN_ENGINE_DP_RANK = -1


class PromptSamplingOverride(TypedDict, total=False):
    prompt_logprob_token_ids: List[List[int]]
    logprob_token_ids: List[int]


class ChatContinuation(TypedDict):
    served_prefix_token_ids: List[int]
    assistant_message_index: int


class InferenceEngineInput(TypedDict):
    # Either prompts or prompt_token_ids must be provided, but not both.
    prompts: Optional[List[ConversationType]]
    prompt_token_ids: Optional[List[List[int]]]
    sampling_params: Optional[Dict[str, Any]]
    # Per-prompt token scoring overrides. Rollout sampling remains shared.
    sampling_params_per_prompt: NotRequired[List[PromptSamplingOverride]]
    session_ids: Optional[List[Hashable]]
    # Per-sample Responses-API options (tools, parallel_tool_calls, etc.) that
    # require the serving backend's resolved chat renderer.
    chat_completion_params: NotRequired[List[Dict[str, Any]]]
    # Preserve sampled tokens when the backend re-renders a structured assistant turn.
    chat_continuations: NotRequired[List[ChatContinuation | None]]
    # The request window, including the backend-rendered prompt and new tokens.
    max_context_length: NotRequired[int]
    max_prompt_length: NotRequired[int | None]


class InferenceEngineOutput(TypedDict):
    # We always return both tokens and text outputs. The tokens are the outputs
    # of inference engine, and the text is the decoded text output. Therefore,
    # it is guaranteed that tokenizer.decode(response_token_ids, skip_special_tokens=True) == responses,
    # but the reverse is not guaranteed, since there are multiple ways to
    # represent the same text with tokens. Therefore, for multi-turn generation,
    # please use token-in-token-out to ensure correctness.
    # `skip_special_tokens=True` is needed because string responses do not include EOS tokens like `<|im_end|>`
    responses: List[str]
    response_ids: List[List[int]]
    stop_reasons: List[str]
    response_logprobs: Optional[List[List[float]]]
    # Exact student-policy candidates at each generated token, when generation
    # requested a positive number of response logprobs. The selected token may
    # also be returned by vLLM, but is not forced into these top-K rows.
    student_topk_indices: NotRequired[List[List[List[int]]]]
    behavior_topk_logprobs: NotRequired[List[List[List[float]]]]
    # prompt_logprobs: per-prompt-token top-K logprobs from vLLM (for teacher scoring).
    # Format: List[List[Optional[Dict[int, float]]]] — outer list is batch,
    # inner list is prompt positions, dict maps token_id → logprob.
    # Only populated when SamplingParams(prompt_logprobs=K) is used.
    prompt_logprobs: Optional[List[List[Optional[Dict[int, float]]]]]
    prompt_ids: NotRequired[List[List[int]]]
    assistant_messages: NotRequired[List[Dict[str, Any]]]
    # One [response token, MoE layer, topk] array per sample. Row i describes
    # the pass that predicted response_ids[i], starting at the final prompt token.
    routed_experts: NotRequired[List[np.ndarray | None]]
    # The policy step whose weights sampled each response token, or UNSTAMPED_POLICY_STEP.
    response_policy_steps: NotRequired[List[List[int]]]


class NamedWeightsUpdateRequest(TypedDict):
    names: List[str]
    dtypes: List[str]
    shapes: List[List[int]]
    sizes: NotRequired[List[int]]
    extras: Optional[List[Dict[str, Any]]]
    packed: NotRequired[bool]


LORA_DISK_LOAD_NAME = "lora_disk_load"
LORA_DISK_PATH_KEY = "lora_disk_path"


def lora_disk_load_request(path: str) -> NamedWeightsUpdateRequest:
    """Return the named-weight request used to load a vLLM LoRA directory."""
    return {
        "names": [LORA_DISK_LOAD_NAME],
        "dtypes": [],
        "shapes": [],
        "extras": [{LORA_DISK_PATH_KEY: path}],
    }


class InferenceEngineInterface(ABC):
    weight_sync_relative_rank_offset: int | None = None
    max_model_len: int | None = None

    def get_model_max_len(self) -> int | None:
        """Return the context limit resolved by the live serving backend."""
        return self.max_model_len

    @abstractmethod
    async def generate(self, input_batch: InferenceEngineInput) -> InferenceEngineOutput:
        raise NotImplementedError()

    @abstractmethod
    async def chat_completion(self, request_payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handles OpenAI-compatible HTTP endpoint.

        Accepts a JSON payload: {"json": <request-body>, "headers": <headers-dict>}.
        The request body will be used to construct a ChatCompletionRequest.
        Returns a plain dict, either a ChatCompletionResponse or an ErrorResponse.
        The specific fields of the response/request depend on the engine's backend.
        """
        raise NotImplementedError()

    @abstractmethod
    async def completion(self, request_payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handles OpenAI-compatible HTTP endpoint.

        Accepts a JSON payload: {"json": <request-body>, "headers": <headers-dict>}.
        The request body will be used to construct a CompletionRequest.
        Returns a plain dict, either a CompletionResponse or an ErrorResponse.
        The specific fields of the response/request depend on the engine's backend.
        """
        raise NotImplementedError()

    async def tokenize(self, request_payload: Dict[str, Any]) -> Dict[str, Any]:
        """Tokenize through the serving backend when supported.

        ``request_payload`` contains the JSON request body under ``json`` and
        serialized HTTP headers under ``headers``. Implementations return the
        backend's native tokenization or error response as a plain dictionary.
        """
        raise NotImplementedError()

    @abstractmethod
    async def wake_up(self, *args: Any, **kwargs: Any):
        raise NotImplementedError()

    @abstractmethod
    async def sleep(self, *args: Any, **kwargs: Any):
        raise NotImplementedError()

    @abstractmethod
    async def init_weight_update_communicator(
        self, master_addr, master_port, rank_offset, world_size, group_name, backend, override_existing: bool = False
    ):
        raise NotImplementedError()

    @abstractmethod
    async def update_named_weights(self, request: NamedWeightsUpdateRequest):
        raise NotImplementedError()

    @abstractmethod
    async def teardown(self):
        raise NotImplementedError()

    @abstractmethod
    async def reset_prefix_cache(self):
        raise NotImplementedError()

    @abstractmethod
    def tp_size(self) -> int:
        """Return the tensor parallel size of this inference engine."""
        raise NotImplementedError()

    @abstractmethod
    def pp_size(self) -> int:
        """Return the pipeline parallel size of this inference engine."""
        raise NotImplementedError()

    @abstractmethod
    def dp_size(self) -> int:
        """Return the data parallel size of this inference engine."""
        raise NotImplementedError()

    @abstractmethod
    async def pause_generation(self) -> None:
        """
        Pause the scheduler for a weight update after aborting all running and waiting
        requests. Running requests return their generated tokens with stop_reason "abort";
        waiting requests return zero completion tokens.
        """
        raise NotImplementedError()

    @abstractmethod
    async def resume_generation(self, policy_step: int) -> None:
        """Resume the scheduler after a weight update; ``policy_step`` stamps the tokens the new weights sample."""
        raise NotImplementedError()

    async def begin_online_eagle_capture(self, config: Dict[str, Any]) -> OnlineEagleResult:
        """Begin a bounded online-EAGLE capture interval when supported."""
        raise NotImplementedError()

    async def seal_online_eagle_capture(self, destination: str) -> OnlineEagleResult:
        """Publish the active capture to cloud storage."""
        raise NotImplementedError()

    async def update_draft_weights(self, weights_path: str) -> OnlineEagleResult:
        """Best-effort update the resident draft from one completed object."""
        raise NotImplementedError()
