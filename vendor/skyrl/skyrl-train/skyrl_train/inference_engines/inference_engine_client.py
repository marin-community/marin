from skyrl_train.inference_engines.base import (
    CHOICE_POLICY_STEPS_KEY,
    SESSION_ID_HEADER,
    UNKNOWN_ENGINE_DP_RANK,
    InferenceEngineInterface,
    InferenceEngineInput,
    InferenceEngineOutput,
    NamedWeightsUpdateRequest,
    OnlineEagleResult,
    PromptSamplingOverride,
)
from skyrl_train.inference_engines.vllm.stats import (
    HTTPBridgeStatsAccumulator,
    IntervalReadMode,
    InferenceStatsSnapshot,
)
from skyrl_train.inference_engines.inference_engine_client_http_endpoint import (
    ErrorResponse,
    ErrorInfo,
    is_engine_error_response,
)
from skyrl_train.inference_engines.chat_template import template_error_from_exception
from transformers import PreTrainedTokenizerBase
import asyncio
from copy import deepcopy
from typing import List, Any, Optional, Dict, Union, Hashable
from skyrl_train.inference_engines.utils import (
    route_prompts_to_engines,
    hash_with_sha256,
    is_single_completion_prompt,
    postprocess_completion_request,
    aggregate_completion_usage_info,
)
from omegaconf import DictConfig
import threading
from collections import OrderedDict
from loguru import logger
import random
import ray.exceptions
from dataclasses import dataclass, field
from http import HTTPStatus
from skyrl_train.inference_engines.chat_continuation import EXACT_PROMPT_TOKEN_IDS_KEY
from skyrl_train.trajectory_runners.routed_experts import decode_routed_experts
import base64
import io
import numpy as np

ABORT_FINISH_REASON = "abort"


def _route_rows(rows: Any, response_length: int) -> np.ndarray:
    """Keep prediction-position routes aligned with the response token they produced."""
    try:
        result = np.asarray(rows)
    except (TypeError, ValueError) as error:
        raise ValueError("routed_experts must be uniform integer rows") from error
    if response_length == 0 and result.size == 0:
        return np.empty((0, 0, 0), dtype=np.uint8)
    if (
        result.ndim != 3
        or result.shape[0] != response_length
        or result.shape[1] == 0
        or result.shape[2] == 0
        or not np.issubdtype(result.dtype, np.integer)
        or np.any(result < 0)
        or np.any(result > np.iinfo(np.uint32).max)
    ):
        raise ValueError("routed_experts must be nonnegative integer rows aligned with response tokens")
    maximum = int(result.max())
    dtype = np.uint8 if maximum <= 255 else np.uint16 if maximum <= 65535 else np.uint32
    return result.astype(dtype, copy=False)


# Cap on the session -> engine memo so it cannot grow unbounded across a long run.
# Sessions are evicted LRU once the cap is exceeded (a re-appearing evicted session
# is simply re-assigned by load, which is harmless).
SESSION_ENGINE_MEMO_MAX_SIZE = 200_000


class InferenceEngineClient(InferenceEngineInterface):
    """
    Client to talk to a set of InferenceEngines.

    Note that InferenceEngineClient sub-classes InferenceEngineInterface so it can be used as if talking to a single engine.
    """

    def __init__(
        self, engines: List[InferenceEngineInterface], tokenizer: PreTrainedTokenizerBase, full_config: DictConfig
    ):
        """
        Args:
            engines: List[InferenceEngineInterface] - The inference engines, remote or local.
            tokenizer: PreTrainedTokenizerBase - The tokenizer to use.
            full_config: DictConfig - See ppo_base_config.yaml
        """
        self.engines = engines
        # create_ray_wrapped_inference_engines lists one local engine per data-parallel rank, in rank order within
        # each engine group; a remote server's rank is unknown.
        data_parallel_size = int(full_config.generator.inference_engine_data_parallel_size)
        self._engine_dp_ranks = [
            index % data_parallel_size if full_config.generator.run_engines_locally else UNKNOWN_ENGINE_DP_RANK
            for index in range(len(engines))
        ]
        self.tokenizer = tokenizer
        # Use served_model_name if configured (for Harbor/LiteLLM compatibility),
        # otherwise fall back to the full model path.
        # See https://github.com/NovaSky-AI/SkyRL/pull/238#discussion_r2326561295
        served_model_name = None
        if hasattr(full_config.generator, "engine_init_kwargs"):
            served_model_name = getattr(full_config.generator.engine_init_kwargs, "served_model_name", None)
        self.model_name = served_model_name if served_model_name else full_config.trainer.policy.model.path
        self.max_model_len = next((getattr(engine, "max_model_len", None) for engine in engines), None)
        self.backend = full_config.generator.backend
        self.enable_http_endpoint = full_config.generator.enable_http_endpoint
        self.http_endpoint_host = full_config.generator.http_endpoint_host
        self.http_endpoint_port = full_config.generator.http_endpoint_port
        self.generation_paused_event = threading.Event()
        # One wake-up event per event loop that has passed the pause barrier since the last
        # release: the trainer's loop and, with the HTTP endpoint, the server thread's loop.
        self._resume_events: dict[asyncio.AbstractEventLoop, asyncio.Event] = {}
        self.weight_sync_pause_timeout = float(full_config.generator.weight_sync_pause_timeout_seconds)
        if self.weight_sync_pause_timeout <= 0:
            raise ValueError("generator.weight_sync_pause_timeout_seconds must be positive")
        self._dead_engines: set[int] = set()

        # ---- Load-aware session routing state ----
        # Per-engine count of requests this client has dispatched but not yet gotten a
        # response for. Because this client is the SOLE router in front of the engines,
        # this outstanding count equals (running + waiting) as seen by each engine — a
        # real-time load signal with no reset-on-read side effect (unlike get_stats(),
        # which resets each vLLM engine's per-step accumulators that InferenceStatsCallback
        # depends on). Used for power-of-two-choices balancing of new sessions.
        self._engine_inflight: List[int] = [0] * len(engines)
        # session_id (str) -> engine_idx. Populated on a session's FIRST request (load
        # balanced) and reused for every later turn (sticky, to preserve prefix-cache
        # reuse). LRU-capped so it cannot grow unbounded. OrderedDict is used as an LRU.
        self._session_engine_memo: "OrderedDict[str, int]" = OrderedDict()
        # Guards _engine_inflight and _session_engine_memo. The critical sections hold
        # no awaits, so this cheap lock keeps the two structures consistent even if the
        # HTTP endpoint thread and other callers ever touch them concurrently.
        self._routing_lock = threading.Lock()
        self._http_bridge_stats = HTTPBridgeStatsAccumulator()

        if self.enable_http_endpoint:
            self._spin_up_http_endpoint()

        logger.info(f"InferenceEngineClient initialized with {len(engines)} engines.")

    def _mark_engine_dead(self, engine_idx: int, error: Exception) -> None:
        """Mark an engine as dead and log a warning."""
        if engine_idx not in self._dead_engines:
            self._dead_engines.add(engine_idx)
            remaining = len(self.engines) - len(self._dead_engines)
            logger.warning(
                f"Inference engine {engine_idx} died ({type(error).__name__}). "
                f"{remaining}/{len(self.engines)} engines remaining."
            )
            if remaining == 0:
                logger.error("All inference engines have died!")

    def _pick_fallback_engine(self, exclude_idx: int) -> int | None:
        """Pick a random live engine, excluding the given index."""
        live = [i for i in range(len(self.engines)) if i not in self._dead_engines and i != exclude_idx]
        return random.choice(live) if live else None

    def _resolve_engine_idx(self, engine_idx: int) -> int:
        """If the chosen engine is dead, pick a fallback. Raises if all are dead."""
        if engine_idx not in self._dead_engines:
            return engine_idx
        fallback = self._pick_fallback_engine(engine_idx)
        if fallback is None:
            raise RuntimeError("All inference engines have died")
        return fallback

    async def _call_engine_with_fallback(self, engine_idx: int, method_name: str, *args: Any) -> tuple[Any, int]:
        """Call an actor, returning its response and index, with one actor-death retry.

        A fallback waits for generation to resume before entering another engine.
        """
        try:
            return await getattr(self.engines[engine_idx], method_name)(*args), engine_idx
        except (ray.exceptions.ActorDiedError, ray.exceptions.RayActorError) as e:
            self._mark_engine_dead(engine_idx, e)
            fallback = self._pick_fallback_engine(engine_idx)
            if fallback is None:
                raise RuntimeError("All inference engines have died") from e
            await self._wait_for_generation_to_resume()
            return await getattr(self.engines[fallback], method_name)(*args), fallback

    # ----------------------------
    # Load-aware session routing
    # ----------------------------
    def _get_engine_loads(self) -> Optional[List[int]]:
        """Return a snapshot of per-engine outstanding-request counts, or None if
        unavailable. None triggers graceful fallback to pure hash routing."""
        try:
            return list(self._engine_inflight)
        except Exception:
            return None

    def _pick_engine_for_new_session(self, session_key: str) -> int:
        """Choose an engine for a not-yet-seen session using power-of-two-choices:
        hash the session to two candidate engines and pick the less loaded of the two.

        Two independent hashes are used so the pair of candidates is well spread; the
        winner (lower outstanding load) is chosen, with ties broken toward the first
        candidate so the choice stays deterministic for a given load snapshot. If load
        info is unavailable, fall back to the original single-hash routing.
        """
        n = len(self.engines)
        if n == 1:
            return 0
        c1 = hash_with_sha256(session_key) % n
        c2 = hash_with_sha256("skyrl-lb:" + session_key) % n

        loads = self._get_engine_loads()
        if loads is None:
            # Graceful fallback: behave exactly like the legacy hash routing.
            return c1
        # Prefer live engines: if a candidate is dead, treat it as maximally loaded so
        # the other candidate wins; if both are dead, _resolve_engine_idx handles it.
        big = float("inf")
        l1 = big if c1 in self._dead_engines else loads[c1]
        l2 = big if c2 in self._dead_engines else loads[c2]
        return c2 if (c2 != c1 and l2 < l1) else c1

    def _route_session(self, session_id: Union[str, int]) -> int:
        """Return the engine index for `session_id`, load-balancing the FIRST request of
        a session and returning the same engine (sticky) for every later turn."""
        key = str(session_id)
        with self._routing_lock:
            memo = self._session_engine_memo
            cached = memo.get(key)
            if cached is not None:
                memo.move_to_end(key)  # LRU touch
                # Keep stickiness, but reroute if the assigned engine has since died.
                return self._resolve_engine_idx(cached)
            engine_idx = self._pick_engine_for_new_session(key)
            engine_idx = self._resolve_engine_idx(engine_idx)
            memo[key] = engine_idx
            memo.move_to_end(key)
            while len(memo) > SESSION_ENGINE_MEMO_MAX_SIZE:
                memo.popitem(last=False)  # evict least-recently-used
            return engine_idx

    def _inc_inflight(self, engine_idx: int) -> None:
        with self._routing_lock:
            if 0 <= engine_idx < len(self._engine_inflight):
                self._engine_inflight[engine_idx] += 1

    def _dec_inflight(self, engine_idx: int) -> None:
        with self._routing_lock:
            if 0 <= engine_idx < len(self._engine_inflight) and self._engine_inflight[engine_idx] > 0:
                self._engine_inflight[engine_idx] -= 1

    async def _run_on_all_engines(self, method_name: str, *args, **kwargs):
        """
        Call a method on all live engines concurrently and gather the results.
        """
        if not self.engines:
            # A zero-engine client (static preference-pair training) has nothing to fan out to.
            return []
        live_engines = [engine for i, engine in enumerate(self.engines) if i not in self._dead_engines]
        if not live_engines:
            raise RuntimeError("All inference engines have died")

        awaitables = [getattr(engine, method_name)(*args, **kwargs) for engine in live_engines]
        return await asyncio.gather(*awaitables)

    async def _run_on_all_engines_before_deadline(self, method_name: str, *args):
        """Fan out a weight-sync pause or resume, failing loudly if an engine never answers."""
        try:
            async with asyncio.timeout(self.weight_sync_pause_timeout):
                return await self._run_on_all_engines(method_name, *args)
        except TimeoutError:
            raise TimeoutError(
                f"{method_name} did not complete on every engine within "
                f"generator.weight_sync_pause_timeout_seconds={self.weight_sync_pause_timeout}; "
                f"requests in flight per engine: {self._engine_inflight}"
            ) from None

    async def expert_block_rpc(self, method: str, *args) -> list:
        """Call one expert-block sync method on every engine. Returns the replies in engine order."""
        return await self._run_on_all_engines("expert_block_rpc", method, *args)

    async def generate(self, input_batch: InferenceEngineInput) -> InferenceEngineOutput:
        # 0. Extract input
        prompts = input_batch.get("prompts")
        prompt_token_ids = input_batch.get("prompt_token_ids")
        session_ids = input_batch.get("session_ids")
        sampling_params = input_batch.get("sampling_params")
        per_prompt_sampling_params = input_batch.get("sampling_params_per_prompt")

        if (prompts is None and prompt_token_ids is None) or (prompts is not None and prompt_token_ids is not None):
            raise ValueError("Either `prompts` or `prompt_token_ids` must be provided, but not both.")
        if prompt_token_ids is None:
            prompt_token_ids = self.tokenizer.apply_chat_template(
                prompts,
                add_generation_prompt=True,
                add_special_tokens=False,
                return_dict=True,
                tokenize=True,
            )["input_ids"]

        num_prompts = len(prompt_token_ids)
        if per_prompt_sampling_params is not None and len(per_prompt_sampling_params) != num_prompts:
            raise ValueError("per-prompt sampling parameters must align with prompt token rows")
        num_inference_engines = len(self.engines)
        if session_ids is not None and len(session_ids) != num_prompts:
            raise ValueError("session_ids must align one-for-one with prompts")

        # 1. Route prompts to engines
        engine_idx_to_prompt_ids: dict[int, list[int]] = route_prompts_to_engines(
            num_prompts=num_prompts,
            num_inference_engines=num_inference_engines,
            session_ids=session_ids,
        )

        # We do a shortcut for non-batched requests, which can support pause/continue generation for
        # in-flight weight updates.
        if num_prompts == 1:
            # Route to a single engine for this single prompt and use retry flow.
            assert len(engine_idx_to_prompt_ids) == 1
            ((engine_idx, prompt_ids_list),) = engine_idx_to_prompt_ids.items()
            assert prompt_ids_list == [0], "Single prompt should map to index [0]"
            engine_idx = self._resolve_engine_idx(engine_idx)
            original_prompt_ids = prompt_token_ids[0]
            return await self._generate_single_with_retry(
                engine_idx=engine_idx,
                original_prompt_ids=original_prompt_ids,
                sampling_params=sampling_params,
                session_id=session_ids[0] if session_ids is not None else None,
                per_prompt_sampling_params=(
                    per_prompt_sampling_params[0] if per_prompt_sampling_params is not None else None
                ),
            )

        # For batched generate(), pause/continue cannot be supported.
        if self.generation_paused_event.is_set():
            raise RuntimeError("pause_generation is unsupported for batched InferenceEngineClient.generate().")

        # 2. Generate responses concurrently (with failover for dead engines)
        tasks: list[asyncio.Task] = []
        indices_list: list[list[int]] = []
        task_engine_idxs: list[int] = []

        # Reroute prompts away from known-dead engines
        rerouted_mapping: dict[int, list[int]] = {}
        for engine_idx, prompt_ids in engine_idx_to_prompt_ids.items():
            resolved = self._resolve_engine_idx(engine_idx)
            rerouted_mapping.setdefault(resolved, []).extend(prompt_ids)

        for engine_idx, prompt_ids in rerouted_mapping.items():
            cur_prompt_token_ids = [prompt_token_ids[i] for i in prompt_ids]
            engine_input = InferenceEngineInput(
                prompt_token_ids=cur_prompt_token_ids,
                sampling_params=sampling_params,
                session_ids=[session_ids[i] for i in prompt_ids] if session_ids is not None else None,
            )
            if per_prompt_sampling_params is not None:
                engine_input["sampling_params_per_prompt"] = [per_prompt_sampling_params[i] for i in prompt_ids]
            tasks.append(asyncio.create_task(self.engines[engine_idx].generate(engine_input)))
            indices_list.append(prompt_ids)
            task_engine_idxs.append(engine_idx)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Retry tasks whose engine died on an available fallback.
        for i, result in enumerate(results):
            if isinstance(result, (ray.exceptions.ActorDiedError, ray.exceptions.RayActorError)):
                self._mark_engine_dead(task_engine_idxs[i], result)
                fallback = self._pick_fallback_engine(task_engine_idxs[i])
                if fallback is None:
                    raise RuntimeError("All inference engines have died") from result
                cur_prompt_token_ids = [prompt_token_ids[j] for j in indices_list[i]]
                engine_input = InferenceEngineInput(
                    prompt_token_ids=cur_prompt_token_ids,
                    sampling_params=sampling_params,
                    session_ids=[session_ids[j] for j in indices_list[i]] if session_ids is not None else None,
                )
                if per_prompt_sampling_params is not None:
                    engine_input["sampling_params_per_prompt"] = [
                        per_prompt_sampling_params[j] for j in indices_list[i]
                    ]
                results[i] = await self.engines[fallback].generate(engine_input)
            elif isinstance(result, BaseException):
                raise result

        # 3. Reconstruct output in original order
        n = len(prompt_token_ids)
        responses: list[str] = [""] * n
        stop_reasons: list[str] = [""] * n
        response_logprobs: List[Optional[List[float]]] = [None for _ in range(n)]
        response_ids: List[List[int]] = [[] for _ in range(n)]
        routed_experts: List[Optional[np.ndarray]] = [None for _ in range(n)]
        prompt_logprobs: List[Optional[Any]] = [None for _ in range(n)]
        student_topk_indices: List[Optional[List[List[int]]]] = [None for _ in range(n)]
        behavior_topk_logprobs: List[Optional[List[List[float]]]] = [None for _ in range(n)]
        response_policy_steps: List[Optional[List[int]]] = [None for _ in range(n)]
        # a bit hacky for now
        add_resp_logprobs = False
        add_prompt_logprobs = False
        add_student_topk = False
        add_routed_experts = False

        for indices, result in zip(indices_list, results):
            policy_steps = result.get("response_policy_steps")
            selected_ids = result.get("student_topk_indices")
            selected_scores = result.get("behavior_topk_logprobs")
            if (selected_ids is None) != (selected_scores is None):
                raise ValueError("Inference engine must return student top-K IDs and behavior scores together")
            if selected_ids is not None:
                if len(selected_ids) != len(indices) or len(selected_scores) != len(indices):
                    raise ValueError("Inference engine student top-K rows must align with responses")
                add_student_topk = True
            captured_routes = result.get("routed_experts")
            if captured_routes is not None:
                if len(captured_routes) != len(indices):
                    raise ValueError("Inference engine routed-expert rows must align with responses")
                add_routed_experts = True
            for local_idx, original_idx in enumerate(indices):
                responses[original_idx] = result["responses"][local_idx]
                stop_reasons[original_idx] = result["stop_reasons"][local_idx]
                response_ids[original_idx] = result["response_ids"][local_idx]
                if captured_routes is not None:
                    routed_experts[original_idx] = _route_rows(
                        captured_routes[local_idx], len(response_ids[original_idx])
                    )
                if policy_steps is not None:
                    response_policy_steps[original_idx] = policy_steps[local_idx]
                if result.get("response_logprobs", None):
                    add_resp_logprobs = True
                    response_logprobs[original_idx] = result["response_logprobs"][local_idx]
                if result.get("prompt_logprobs") is not None:
                    add_prompt_logprobs = True
                    prompt_logprobs[original_idx] = result["prompt_logprobs"][local_idx]
                if selected_ids is not None:
                    if len(selected_ids[local_idx]) != len(response_ids[original_idx]) or len(
                        selected_scores[local_idx]
                    ) != len(response_ids[original_idx]):
                        raise ValueError("Inference engine student top-K tokens must align with response tokens")
                    student_topk_indices[original_idx] = selected_ids[local_idx]
                    behavior_topk_logprobs[original_idx] = selected_scores[local_idx]

        output = InferenceEngineOutput(
            responses=responses,
            stop_reasons=stop_reasons,
            response_ids=response_ids,
            response_logprobs=response_logprobs if add_resp_logprobs else None,
            prompt_logprobs=prompt_logprobs if add_prompt_logprobs else None,
        )
        if add_student_topk:
            if any(row is None for row in student_topk_indices) or any(row is None for row in behavior_topk_logprobs):
                raise ValueError("Inference engine omitted student top-K evidence for part of the batch")
            output["student_topk_indices"] = student_topk_indices
            output["behavior_topk_logprobs"] = behavior_topk_logprobs
        if add_routed_experts:
            if any(routes is None for routes in routed_experts):
                raise ValueError("Inference engine omitted routed experts for part of a batch")
            output["routed_experts"] = [routes for routes in routed_experts if routes is not None]
        if all(row is not None for row in response_policy_steps):
            output["response_policy_steps"] = response_policy_steps
        return output

    async def begin_online_eagle_capture(self, config: Dict[str, Any]) -> List[OnlineEagleResult]:
        """Begin the same capture interval on every live inference engine."""
        return await self._run_on_all_engines("begin_online_eagle_capture", config)

    async def seal_online_eagle_capture(self, destination: str) -> List[OnlineEagleResult]:
        """Publish every engine's capture before target-weight synchronization."""
        return await self._run_on_all_engines("seal_online_eagle_capture", destination)

    async def update_draft_weights(self, weights_path: str) -> List[OnlineEagleResult]:
        """Ask every live engine to best-effort update its resident draft."""
        awaitables = [
            engine.update_draft_weights(weights_path)
            for index, engine in enumerate(self.engines)
            if index not in self._dead_engines
        ]
        results = await asyncio.gather(*awaitables, return_exceptions=True)
        return [
            {"active": False, "error": f"{type(result).__name__}: {result}"}
            if isinstance(result, BaseException)
            else result
            for result in results
        ]

    async def _generate_single_with_retry(
        self,
        engine_idx: int,
        original_prompt_ids: List[int],
        sampling_params: Optional[Dict[str, Any]],
        session_id: Hashable | None = None,
        per_prompt_sampling_params: Optional[PromptSamplingOverride] = None,
    ) -> InferenceEngineOutput:
        """
        Generate a single response with retry mechanism.

        This method is equivalent to `_chat_completion_with_retry()` but for the `generate()` codepath.
        We keep sending `generate` requests (with previous responses accumulated) until the finish_reason
        is not "abort". Per-prompt teacher scoring instead fails on an abort because replaying selected-ID
        overrides against an extended prompt would misalign the scored positions.

        This method is equivalent to a single `generate()` call if we do not use `pause_generation()`.

        Since we operate purely in the token space, it is token-in-token-out, unlike `_chat_completion_with_retry()`
        which re-encodes in each new request.

        For subsequent retry requests (`InferenceEngineInput`), we:
        - Update the `InferenceEngineInput.prompt_token_ids` with the accumulated output tokens.
        - Skip accumulating `InferenceEngineOutput.responses` since we decode the final output.
        - Adjust remaining max tokens if `max_tokens` or `max_completion_tokens` is present.

        For the final response, we return `InferenceEngineOutput` with:
        - `responses`: decoded at the end from `response_ids` if generation is completed in > 1 turns, otherwise the text response of the first turn.
        - `response_ids`: the accumulated output tokens
        - `stop_reasons`: the stop reason of the final response
        - `response_logprobs`: the accumulated logprobs
        """
        if sampling_params is None:
            sampling_params = {}

        # 1. First determine original max tokens key and value (if any)
        max_key = None
        if "max_tokens" in sampling_params:
            max_key = "max_tokens"
        elif "max_completion_tokens" in sampling_params:
            max_key = "max_completion_tokens"
        original_max_tokens: Optional[int] = sampling_params.get(max_key) if max_key else None

        # 2. Initialize fields we want to accumulate or update in each loop iteration
        accum_response_ids: List[int] = []
        accum_response_logprobs: List[float] = []
        accum_student_topk_indices: List[List[int]] = []
        accum_behavior_topk_logprobs: List[List[float]] = []
        accum_routed_experts: List[np.ndarray] = []
        accum_policy_steps: List[int] = []
        saw_student_topk: Optional[bool] = None
        saw_routed_experts: Optional[bool] = None
        stop_reason: str = ABORT_FINISH_REASON

        # We only use it if generation is completed in one turn to maintain original behavior with no retry.
        text_response: Optional[str] = None
        num_turns = 0

        # 3. Loop until geneartion is completed.
        while stop_reason == ABORT_FINISH_REASON:
            await self._wait_for_generation_to_resume()

            # 3.1. Prepare the request payload.
            cur_sampling_params = sampling_params.copy()
            if original_max_tokens is not None:
                new_max_tokens = original_max_tokens - len(accum_response_ids)
                assert new_max_tokens >= 0, f"Expect new_max_tokens to be non-negative, but got {new_max_tokens}"
                cur_sampling_params[max_key] = new_max_tokens
            new_prompt_ids = original_prompt_ids + accum_response_ids
            engine_input = InferenceEngineInput(
                prompt_token_ids=[new_prompt_ids],
                sampling_params=cur_sampling_params,
                session_ids=[session_id] if session_id is not None else None,
            )
            if per_prompt_sampling_params is not None:
                engine_input["sampling_params_per_prompt"] = [per_prompt_sampling_params]

            # 3.2. Send the request.
            logger.debug("generate() request sent (including potential retries): {}", engine_input)
            try:
                partial_response: InferenceEngineOutput = await self.engines[engine_idx].generate(engine_input)
            except (ray.exceptions.ActorDiedError, ray.exceptions.RayActorError) as e:
                self._mark_engine_dead(engine_idx, e)
                fallback = self._pick_fallback_engine(engine_idx)
                if fallback is None:
                    raise RuntimeError("All inference engines have died") from e
                engine_idx = fallback
                # Reset accumulation — new engine has no prior context
                accum_response_ids = []
                accum_response_logprobs = []
                accum_student_topk_indices = []
                accum_behavior_topk_logprobs = []
                accum_routed_experts = []
                accum_policy_steps = []
                saw_student_topk = None
                saw_routed_experts = None
                num_turns = 0
                stop_reason = ABORT_FINISH_REASON
                continue

            # 3.3. Parse the partial response.
            assert len(partial_response["response_ids"]) == 1, "Expected exactly one response."
            new_response_ids: List[int] = partial_response["response_ids"][0]
            text_response = partial_response["responses"][0]
            stop_reason = partial_response["stop_reasons"][0]
            if per_prompt_sampling_params is not None and stop_reason == ABORT_FINISH_REASON:
                raise RuntimeError("per-prompt teacher scoring cannot continue after an aborted generation")
            new_response_logprobs: Optional[List[float]] = None
            new_response_logprobs_list: Optional[List[List[float]]] = partial_response.get("response_logprobs", None)
            if new_response_logprobs_list is not None and len(new_response_logprobs_list) > 0:
                new_response_logprobs = new_response_logprobs_list[0]

            # 3.4 Aborted without generating tokens, so partial_response is useless.
            if stop_reason == ABORT_FINISH_REASON and len(new_response_ids) == 0:
                continue

            selected_ids = partial_response.get("student_topk_indices")
            selected_scores = partial_response.get("behavior_topk_logprobs")
            if (selected_ids is None) != (selected_scores is None):
                raise ValueError("Inference engine must return student top-K IDs and behavior scores together")
            has_student_topk = selected_ids is not None
            if saw_student_topk is not None and saw_student_topk != has_student_topk:
                raise ValueError("Inference engine omitted student top-K evidence for part of a response")
            saw_student_topk = has_student_topk
            if has_student_topk:
                if (
                    len(selected_ids) != 1
                    or len(selected_scores) != 1
                    or len(selected_ids[0]) != len(new_response_ids)
                    or len(selected_scores[0]) != len(new_response_ids)
                ):
                    raise ValueError("Inference engine student top-K tokens must align with response tokens")
                accum_student_topk_indices.extend(selected_ids[0])
                accum_behavior_topk_logprobs.extend(selected_scores[0])

            captured_routes = partial_response.get("routed_experts")
            has_routes = captured_routes is not None
            if saw_routed_experts is not None and saw_routed_experts != has_routes:
                raise ValueError("Inference engine omitted routed experts for part of a response")
            saw_routed_experts = has_routes
            if has_routes:
                if len(captured_routes) != 1:
                    raise ValueError("Inference engine routed-expert rows must align with responses")
                rows = _route_rows(captured_routes[0], len(new_response_ids))
                if rows.size:
                    if accum_routed_experts and rows.shape[1:] != accum_routed_experts[0].shape[1:]:
                        raise ValueError("Inference engine routed-expert geometry changed during retry")
                    accum_routed_experts.append(rows)

            # 3.5 Accumulate outputs
            accum_response_ids.extend(new_response_ids)
            if new_response_logprobs is not None:
                accum_response_logprobs.extend(new_response_logprobs)
            if partial_response.get("response_policy_steps") is not None:
                accum_policy_steps.extend(partial_response["response_policy_steps"][0])
            num_turns += 1

        # 4. Build the final response and return.
        if num_turns == 1:
            final_text_response = text_response
        else:
            final_text_response = self.tokenizer.decode(accum_response_ids, skip_special_tokens=True)

        # Propagate prompt_logprobs from the last partial response (only meaningful
        # for teacher scoring where max_tokens=1 and num_turns=1).
        final_prompt_logprobs = partial_response.get("prompt_logprobs") if partial_response else None

        output = InferenceEngineOutput(
            responses=[final_text_response],
            stop_reasons=[stop_reason],
            response_ids=[accum_response_ids],
            response_logprobs=[accum_response_logprobs] if len(accum_response_logprobs) > 0 else None,
            prompt_logprobs=final_prompt_logprobs,
        )
        if saw_student_topk:
            output["student_topk_indices"] = [accum_student_topk_indices]
            output["behavior_topk_logprobs"] = [accum_behavior_topk_logprobs]
        if saw_routed_experts:
            output["routed_experts"] = [
                np.concatenate(accum_routed_experts, axis=0)
                if accum_routed_experts
                else np.empty((0, 0, 0), dtype=np.uint8)
            ]
        if accum_policy_steps:
            output["response_policy_steps"] = [accum_policy_steps]
        return output

    async def _chat_completion_with_retry(
        self, engine_idx: int, original_request_payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Continue an aborted chat response after generation resumes.

        The final response retains the first nonempty response's fields.
        Token IDs, log probabilities, content, and routed experts accumulate
        across requests. Finish and stop reasons come from the last response.
        """
        original_request_json: Dict[str, Any] = original_request_payload.get("json", {}).copy()
        headers: Dict[str, str] = original_request_payload.get("headers", {}).copy()

        assert not original_request_json.get("continue_final_message", False), (
            "continue_final_message must be False for /chat/completions requests"
        )

        # Accumulated fields for building subsequent requests and final response. It is inplace-updated
        # in `_parse_partial_response_and_inplace_update_accum()`.
        accum = AccumulatedResponse()

        # First non-empty response (i.e. the response that prefilled the prompt) to copy meta from.
        base_response: Optional[Dict[str, Any]] = None

        # Determine original max tokens key and value (if any)
        max_key = None
        if "max_tokens" in original_request_json:
            max_key = "max_tokens"
        elif "max_completion_tokens" in original_request_json:
            max_key = "max_completion_tokens"
        orig_max_tokens: Optional[int] = original_request_json.get(max_key) if max_key else None

        # Fields to be updated in each loop iteration
        finish_reason: str = ABORT_FINISH_REASON
        stop_reason: Optional[str] = None
        response_role: Optional[str] = None

        # 1. Loop until the generation is completed.
        while finish_reason == ABORT_FINISH_REASON:
            await self._wait_for_generation_to_resume()

            # 1.1. Prepare the request payload.
            cur_request_json = _prepare_retry_request(
                original_request_json=original_request_json,
                accum=accum,
                response_role=response_role,
                orig_max_tokens=orig_max_tokens,
                max_key=max_key,
            )

            # 1.2. Send the request.
            logger.debug("/chat/completions request sent (including potential retries): {}", cur_request_json)
            try:
                partial_response = await self.engines[engine_idx].chat_completion(
                    {"json": cur_request_json, "headers": headers}
                )
            except (ray.exceptions.ActorDiedError, ray.exceptions.RayActorError) as e:
                self._mark_engine_dead(engine_idx, e)
                fallback = self._pick_fallback_engine(engine_idx)
                if fallback is None:
                    raise RuntimeError("All inference engines have died") from e
                engine_idx = fallback
                # Reset accumulator — new engine has no prior context
                accum = AccumulatedResponse()
                base_response = None
                response_role = None
                finish_reason = ABORT_FINISH_REASON
                continue

            # 1.2.1. Check for error response from vLLM/sglang.
            # Error responses have "error" key (vLLM) or "object"="error" (sglang), not "choices".
            if is_engine_error_response(partial_response):
                error_info = partial_response.get("error", partial_response)
                error_msg = (
                    error_info.get("message", str(error_info)) if isinstance(error_info, dict) else str(error_info)
                )

                # Handle continue_final_message errors by falling back to fresh request.
                # This can happen when chat templates (e.g., Qwen3 thinking) modify assistant
                # content in ways that make vLLM unable to find the continuation point.
                if "continue_final_message" in error_msg and accum.completion_tokens > 0:
                    logger.warning(
                        f"continue_final_message failed after {accum.completion_tokens} tokens, retrying fresh"
                    )
                    # Reset accumulator and retry with original request
                    accum = AccumulatedResponse()
                    response_role = None
                    continue

                # Return the error response dict instead of raising, so that
                # the HTTP endpoint can forward it with the correct status code
                # (e.g., 400 for context length errors). Raising RuntimeError here
                # would cause the endpoint to wrap it as a generic HTTP 500, which
                # breaks LiteLLM's error classification in downstream consumers
                # like Harbor.
                logger.warning(
                    "Inference engine error: category={}, request_id={}",
                    partial_response.get("error_category", "server_error"),
                    partial_response.get("request_id"),
                )
                return partial_response

            # 1.3. Parse partial response and in-place update accumulators.
            finish_reason, stop_reason, response_role, aborted_without_generating = (
                _parse_partial_response_and_inplace_update_accum(
                    partial_response=partial_response,
                    accum=accum,
                    response_role=response_role,
                )
            )

            # 1.4. Aborted without generating tokens, so partial_response is useless.
            if aborted_without_generating:
                continue

            # At this point, either some tokens were generated and/or request completed with a non-"abort" finish_reason

            # 1.5. Update base response if it is the first non-empty response
            if base_response is None:
                if finish_reason != ABORT_FINISH_REASON:
                    # If we only made one request and it is not aborted, return the partial result directly.
                    # This is the codepath that will hit when we do not use `pause_generation()` or `resume_generation()`.
                    return partial_response
                # NOTE(Charlie): not doing deepcopy here to avoid copying large logprobs, so be careful when modifying this.
                base_response = partial_response.copy()

        # 2. Build final response by combining fields
        assert base_response is not None, "Expected at least one non-empty response to build final response"
        return _build_final_response(
            base_response=base_response,
            accum=accum,
            finish_reason=finish_reason,
            stop_reason=stop_reason,
        )

    async def generate_on_live_engines(
        self, prompt_token_ids: List[int], sampling_params: Dict[str, Any]
    ) -> List[tuple[int, InferenceEngineOutput]]:
        """One response to ``prompt_token_ids`` from every live engine, with that engine's data-parallel rank."""
        ranks = [rank for index, rank in enumerate(self._engine_dp_ranks) if index not in self._dead_engines]
        engine_input = InferenceEngineInput(prompt_token_ids=[prompt_token_ids], sampling_params=sampling_params)
        outputs = await self._run_on_all_engines("generate", engine_input)
        for output in outputs:
            routes = output.get("routed_experts")
            if routes is not None:
                output["routed_experts"] = [_route_rows(routes[0], len(output["response_ids"][0]))]
        return list(zip(ranks, outputs, strict=True))

    def _chat_request(self, request_payload: Dict[str, Any]) -> tuple[int, Dict[str, Any]]:
        """Prepare the common direct, HTTP, and streaming chat request boundary."""
        request_payload = deepcopy(request_payload)
        # Route capture is a server capability, independent of the caller's transport.
        # Retry assembly needs exact IDs even for ordinary OpenAI clients that never
        # request training metadata. Do not reconstruct these IDs from decoded text.
        request_payload["json"]["return_token_ids"] = True
        session_id = request_payload["json"].pop("session_id", None)
        if session_id is None:
            session_id = request_payload.get("headers", {}).get(SESSION_ID_HEADER)
        if session_id is None:
            engine_idx = self._resolve_engine_idx(random.randint(0, len(self.engines) - 1))
        else:
            assert isinstance(session_id, (str, int)), "Session ID must be an integer or string for `/chat/completions`"
            # Load-balance the FIRST request of this session (power-of-two-choices over
            # per-engine outstanding load); every later turn of the same session is
            # sticky to that engine to preserve prefix-cache reuse.
            engine_idx = self._route_session(session_id)

        return engine_idx, request_payload

    async def chat_completion(self, request_payload: Dict[str, Any]) -> Dict[str, Any]:
        engine_idx, request_payload = self._chat_request(request_payload)

        # Track outstanding load so concurrent sessions balance across engines instead of
        # hash-pinning onto one. Attributed to the initially chosen engine; the retry loop
        # may fail over to another engine on death, a rare case we don't re-attribute.
        self._inc_inflight(engine_idx)
        try:
            # Always use the retry loop which also issues the first request inside
            return await self._chat_completion_with_retry(engine_idx, request_payload)
        finally:
            self._dec_inflight(engine_idx)

    async def tokenize(self, request_payload: Dict[str, Any]) -> Dict[str, Any]:
        """Delegate terminal-bench tokenization to a live vLLM serving actor."""
        if self.backend != "vllm":
            return ErrorResponse(
                error=ErrorInfo(
                    message=f"The /tokenize endpoint requires the vLLM backend, got {self.backend}.",
                    type=HTTPStatus.BAD_REQUEST.phrase,
                    code=HTTPStatus.BAD_REQUEST.value,
                )
            ).model_dump()

        engine_idx = self._resolve_engine_idx(random.randint(0, len(self.engines) - 1))
        try:
            response, _ = await self._call_engine_with_fallback(engine_idx, "tokenize", request_payload)
        except Exception as error:
            template_error = template_error_from_exception(error)
            if template_error is not None:
                raise template_error from error
            raise
        return response

    async def chat_completion_stream(self, request_payload: Dict[str, Any]):
        """Streaming chat completion — yields SSE-formatted strings.

        Uses the same session-based routing as ``chat_completion``. An in-flight
        stream cannot be re-issued after abort because there is no per-turn
        boundary. With the vLLM keep policy, the engine holds the stream in place
        until weights are updated and generation resumes.

        A new stream waits at the same pause barrier as non-streaming requests.
        """
        # Wait before routing or counting the request so it cannot enter the engine
        # during the reload or hold an engine slot while blocked.
        await self._wait_for_generation_to_resume()

        engine_idx, request_payload = self._chat_request(request_payload)

        self._inc_inflight(engine_idx)
        try:
            async for chunk in self.engines[engine_idx].chat_completion_stream(request_payload):
                yield chunk
        except (ray.exceptions.ActorDiedError, ray.exceptions.RayActorError) as e:
            self._mark_engine_dead(engine_idx, e)
            raise
        finally:
            self._dec_inflight(engine_idx)

    async def _single_completion_with_pause_retry(
        self, engine_idx: int, request_json: Dict[str, Any], headers: Dict[str, str]
    ) -> Dict[str, Any]:
        """Issue a single-prompt completion, retrying one pause-aborted response."""
        payload = {"json": request_json, "headers": headers}
        for attempt in (0, 1):
            await self._wait_for_generation_to_resume()
            result, engine_idx = await self._call_engine_with_fallback(engine_idx, "completion", payload)

            if not _completion_response_was_aborted(result):
                return result
            if attempt == 0:
                logger.info("/completions generation was aborted by a weight-sync pause; re-issuing once after resume")

        logger.warning("/completions generation was aborted again after the pause retry; returning it as-is")
        return result

    async def completion(self, request_payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handles an OpenAI /completions request.

        Since `request["prompt"]` can be `Union[list[int], list[list[int]], str, list[str]]`,
        (i.e. {batched, single} x {string, token IDs}), we need to route the request to engines
        differently, based on whether it's a single or batched request, and whether `request["session_id"]`
        is provided. This is similar to `generate()` method.

        For single, we do the same routing logic as `chat_completion()`. For batched, we route by
        `request["session_id"]` if present, and if not we split evenly across engines.

        Regardless, the order will be maintained, i.e. `output["choices"][i]` corresponds to `request["prompt"][i]`.

        Single-prompt requests wait out a weight-sync pause and retry once if a pause aborts
        generation mid-flight. Batched requests still reject a pause because their fan-out cannot
        be resumed atomically.
        """
        body = request_payload.get("json", {})

        # NOTE(Charlie): do not reuse headers here as the single request may become various new requests
        headers = {"Content-Type": "application/json"}

        # 1. Postprocess prompt, session_id, and validate request. `is_single_prompt` has to be
        # read off the RAW prompt: postprocess normalises a single prompt into a singleton list,
        # after which it is indistinguishable from a length-1 batch.
        prompt = body.get("prompt")
        is_single_prompt = is_single_completion_prompt(prompt)
        session_id_value = body.pop("session_id", None)
        ret = postprocess_completion_request(prompt, session_id_value)
        session_id_list: Optional[Union[List[int], List[str], ErrorResponse]] = ret[0]
        prompt: Union[List[List[int]], List[str]] = ret[1]
        if isinstance(session_id_list, ErrorResponse):
            return session_id_list.model_dump()

        num_prompts = len(prompt)
        num_inference_engines = len(self.engines)
        assert num_prompts > 0, "Number of prompts must be greater than 0"

        # 1. Route prompts to engines
        engine_idx_to_prompt_ids: dict[int, list[int]] = route_prompts_to_engines(
            num_prompts=num_prompts,
            num_inference_engines=num_inference_engines,
            session_ids=session_id_list,
        )

        # Reroute prompts away from known-dead engines before dispatching
        rerouted_mapping: dict[int, list[int]] = {}
        for engine_idx, prompt_ids in engine_idx_to_prompt_ids.items():
            resolved = self._resolve_engine_idx(engine_idx)
            rerouted_mapping.setdefault(resolved, []).extend(prompt_ids)

        # 1.1. Single prompt: one engine, one sub-request, and a weight-sync pause is survivable
        # (barrier + one re-issue on abort). Handled on its own path below.
        if is_single_prompt:
            ((single_engine_idx, _),) = rerouted_mapping.items()
            single_json = dict(body)
            single_json["prompt"] = list(prompt)
            result = await self._single_completion_with_pause_retry(single_engine_idx, single_json, headers)
            error_response = _completion_engine_error_response([result])
            return error_response if error_response is not None else result

        return await self._batched_completion(body, prompt, rerouted_mapping, headers)

    async def _batched_completion(
        self,
        body: Dict[str, Any],
        prompt: Union[List[List[int]], List[str]],
        engine_idx_to_prompt_ids: dict[int, list[int]],
        headers: Dict[str, str],
    ) -> Dict[str, Any]:
        """Fan out a batched completion request and restore prompt order."""

        # Batched: no pause handling exists for a fan-out request, so a pause is still an error.
        if self.generation_paused_event.is_set():
            raise RuntimeError("pause_generation is unsupported for batched /completions requests.")

        tasks: list[asyncio.Task] = []
        indices_list: list[list[int]] = []
        task_engine_idxs: list[int] = []
        for engine_idx, prompt_ids in engine_idx_to_prompt_ids.items():
            cur_prompt = [prompt[i] for i in prompt_ids]
            cur_json = dict(body)
            cur_json["prompt"] = cur_prompt
            coro = self.engines[engine_idx].completion({"json": cur_json, "headers": headers})
            tasks.append(asyncio.create_task(coro))
            indices_list.append(prompt_ids)
            task_engine_idxs.append(engine_idx)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Retry tasks whose engine died on an available fallback.
        for i, result in enumerate(results):
            if isinstance(result, (ray.exceptions.ActorDiedError, ray.exceptions.RayActorError)):
                self._mark_engine_dead(task_engine_idxs[i], result)
                fallback = self._pick_fallback_engine(task_engine_idxs[i])
                if fallback is None:
                    raise RuntimeError("All inference engines have died") from result
                cur_prompt = [prompt[j] for j in indices_list[i]]
                cur_json = dict(body)
                cur_json["prompt"] = cur_prompt
                results[i] = await self.engines[fallback].completion({"json": cur_json, "headers": headers})
            elif isinstance(result, BaseException):
                raise result

        error_response = _completion_engine_error_response(results)
        if error_response is not None:
            return error_response

        # Combine choices and preserve original order.
        # If there is only one result, we return it directly.
        if len(results) == 1:
            return results[0]

        # Use the first result as base response. There are some fields that cannot be shared
        # across sub-requests. For now it is just the usage field.
        final_response = dict(results[0])
        final_response["usage"] = aggregate_completion_usage_info(results, self.backend)

        # Aggregate choices. TODO(Charlie): improve logic when we need to support n > 1
        # vLLM sets index positions per sub-batch, so we reset indices to be 0..n-1 for the combined response.
        combined_choices: list[Dict[str, Any]] = [None] * len(prompt)
        for indices, result in zip(indices_list, results):
            # indices are the original prompt indices that the task's response corresponds to
            for local_idx, original_idx in enumerate(indices):
                choice = result["choices"][local_idx]
                choice["index"] = original_idx  # overwrite index with the global position
                combined_choices[original_idx] = choice

        # sanity check that the index is correct
        for new_idx in range(len(combined_choices)):
            assert combined_choices[new_idx]["index"] == new_idx

        final_response["choices"] = combined_choices
        return final_response

    async def wake_up(self, *args: Any, **kwargs: Any):
        return await self._run_on_all_engines("wake_up", *args, **kwargs)

    async def sleep(self, *args: Any, **kwargs: Any):
        return await self._run_on_all_engines("sleep", *args, **kwargs)

    def _live_engine_communicator_offsets(self, rank_offset: int) -> list[tuple[InferenceEngineInterface, int]]:
        offsets = []
        next_rank_offset = rank_offset
        for index, engine in enumerate(self.engines):
            if index in self._dead_engines:
                continue
            relative_rank_offset = engine.weight_sync_relative_rank_offset
            engine_rank_offset = (
                rank_offset + relative_rank_offset if relative_rank_offset is not None else next_rank_offset
            )
            offsets.append((engine, engine_rank_offset))
            if relative_rank_offset is None:
                next_rank_offset += engine.tp_size() * engine.pp_size()
        return offsets

    async def init_weight_update_communicator(
        self,
        master_addr,
        master_port,
        rank_offset,
        world_size,
        group_name,
        backend,
        override_existing: bool = False,
    ):
        tasks = []
        for engine, engine_rank_offset in self._live_engine_communicator_offsets(rank_offset):
            tasks.append(
                engine.init_weight_update_communicator(
                    master_addr=master_addr,
                    master_port=master_port,
                    rank_offset=engine_rank_offset,
                    world_size=world_size,
                    group_name=group_name,
                    backend=backend,
                    override_existing=override_existing,
                )
            )
        await asyncio.gather(*tasks)

    async def update_named_weights(self, request: NamedWeightsUpdateRequest):
        return await self._run_on_all_engines("update_named_weights", request=request)

    async def begin_weight_reload(self):
        """#1685 fix: open the layerwise-reload bracket on all engines so a multi-chunk
        RL weight sync defers per-layer processing; finish_weight_reload() then re-runs
        process_weights_after_loading (re-applying the FlashInfer-CUTLASS w13 swap) once."""
        return await self._run_on_all_engines("begin_weight_reload")

    async def finish_weight_reload(self):
        """#1685 fix: close the layerwise-reload bracket on all engines."""
        return await self._run_on_all_engines("finish_weight_reload")

    async def reset_prefix_cache(self):
        return await self._run_on_all_engines("reset_prefix_cache")

    async def teardown(self):
        return await self._run_on_all_engines("teardown")

    async def get_stats(self, read_mode: IntervalReadMode = IntervalReadMode.RESET) -> InferenceStatsSnapshot:
        """Read every live engine without aggregating away labels or histogram buckets."""
        engine_snapshots = await self._run_on_all_engines("get_stats", read_mode=read_mode)
        return InferenceStatsSnapshot(
            engines=tuple(engine_snapshots),
            http_bridge=self._http_bridge_stats.snapshot(read_mode),
        )

    def tp_size(self) -> int:
        raise NotImplementedError("InferenceEngineClient does not implement tp_size()")

    def pp_size(self) -> int:
        raise NotImplementedError("InferenceEngineClient does not implement pp_size()")

    def dp_size(self) -> int:
        raise NotImplementedError("InferenceEngineClient does not implement dp_size()")

    # ----------------------------
    # Generation pause and resume
    # ----------------------------
    async def _wait_for_generation_to_resume(self) -> None:
        """Wait, on this loop, until a weight-sync pause is released."""
        while True:
            # Register before checking the flag; the release clears the flag and then takes
            # the events. A waiter that sees the flag set is therefore always among the events
            # the release wakes. Both sides are single dict operations, so no lock is needed.
            event = self._resume_events.setdefault(asyncio.get_running_loop(), asyncio.Event())
            if not self.generation_paused_event.is_set():
                return
            await event.wait()

    def _release_generation_waiters(self) -> None:
        """Clear the pause and wake every parked request on every loop."""
        self.generation_paused_event.clear()
        events, self._resume_events = self._resume_events, {}
        for loop, event in events.items():
            try:
                loop.call_soon_threadsafe(event.set)
            except RuntimeError:
                # The HTTP server's loop has closed; nothing waits on it any more.
                pass

    async def pause_generation(self) -> None:
        """Pause every engine scheduler for an in-flight weight update.

        Returns once every live engine has acknowledged the pause, with no request running and
        any later arrival held until resume. Chat, single-prompt ``generate()`` and single-prompt
        completion requests wait for the resume; batched ``generate()`` and batched completions
        are rejected while paused.
        """
        if self.generation_paused_event.is_set():
            raise RuntimeError("Generation is already paused, cannot pause again.")
        self.generation_paused_event.set()
        await self._run_on_all_engines_before_deadline("pause_generation")

    async def resume_generation(self, policy_step: int) -> None:
        """
        Resumes generation for all engines, intended for in-flight weight updates and partial rollouts.

        Resume all in-flight requests with the previously-generated tokens, and unblock incoming requests
        that were blocked by `pause_generation()`. ``policy_step`` stamps the tokens the engines sample next.
        """
        if not self.generation_paused_event.is_set():
            raise RuntimeError("Generation is not paused, cannot resume.")
        await self._run_on_all_engines_before_deadline("resume_generation", policy_step)
        self._release_generation_waiters()

    # ----------------------------
    # HTTP endpoint related methods
    # ----------------------------

    def __del__(self):
        """
        Destructor to shut down the HTTP endpoint if it was started.
        """
        # TODO(Charlie): __del__ is not guaranteed to be called in general. Add to `teardown` method
        # when the `_handle_termination` flow is implemented. See `skyrl_train/workers/worker.py`
        # comments on `_handle_termination` for more details.
        if (
            self.enable_http_endpoint
            and hasattr(
                self, "_server_thread"
            )  # don't want to shut down the server when it is pickled as a ray method argument.
            and self._server_thread is not None
        ):
            try:
                from skyrl_train.inference_engines.inference_engine_client_http_endpoint import shutdown_server

                shutdown_server(
                    host=self.http_endpoint_host,
                    port=self.http_endpoint_port,
                    max_wait_seconds=10,
                )
                if hasattr(self, "_server_thread") and self._server_thread.is_alive():
                    self._server_thread.join(timeout=10)
            except Exception as e:
                logger.error(f"Error shutting down HTTP endpoint: {e}")

    def __getstate__(self):
        """
        Override to avoid pickling the server thread and the threading.Event object, which are not picklable.
        Needed when passing InferenceEngineClient as an argument to async_run_ray_method(), mainly for
        invoking `init_weight_sync_state()` and `broadcast_to_inference_engines()`, which do
        not need these attributes.
        """
        state = self.__dict__.copy()
        state["_server_thread"] = None
        state["generation_paused_event"] = None
        state["_resume_events"] = None
        # threading.Lock is not picklable; the pickled copy is only used for weight-sync
        # RPC args and never routes, so dropping the routing lock is safe.
        state["_routing_lock"] = None
        state["_http_bridge_stats"] = None
        return state

    def _spin_up_http_endpoint(self):
        from skyrl_train.inference_engines.inference_engine_client_http_endpoint import (
            serve,
            wait_for_server_ready,
        )

        # Bind the uvicorn server to 0.0.0.0 so that clients addressing this node
        # by its IP (e.g. Harbor rollout workers and their agents) can reach the
        # endpoint over the internal compute network. The configured
        # `http_endpoint_host` (default 127.0.0.1) is the CLIENT-side host used
        # for the local readiness probe below; binding the SERVER to 0.0.0.0 is
        # a superset of binding to 127.0.0.1, so the loopback readiness probe and
        # the fan-out-OFF path (client uses 127.0.0.1 on the same node) are both
        # unaffected. This is safe: the endpoint only lives on the internal
        # 10.128.x.x compute net, never the public internet.
        self._server_thread = threading.Thread(
            target=serve,
            args=(self,),
            kwargs={
                "host": "0.0.0.0",
                "port": self.http_endpoint_port,
                "log_level": "warning",
                "bridge_stats": self._http_bridge_stats,
            },
            daemon=True,
        )
        self._server_thread.start()
        wait_for_server_ready(
            host=self.http_endpoint_host,
            port=self.http_endpoint_port,
            max_wait_seconds=30,
        )
        logger.info(
            f"InferenceEngineClient HTTP endpoint started on {self.http_endpoint_host}:{self.http_endpoint_port}"
        )

    def shutdown_http_endpoint(self) -> None:
        """Shut down the HTTP endpoint server.

        Must be called during teardown BEFORE killing inference engines.
        Otherwise in-flight requests get forwarded to dead engines, receive
        ActorDiedError, and the retry logic keeps the process alive
        indefinitely.
        """
        if not self.enable_http_endpoint:
            return

        from skyrl_train.inference_engines.inference_engine_client_http_endpoint import (
            shutdown_server,
        )

        try:
            shutdown_server(
                host=self.http_endpoint_host,
                port=self.http_endpoint_port,
                max_wait_seconds=10,
            )
        except Exception as e:
            logger.warning(f"HTTP endpoint shutdown error (non-fatal): {e}")

        if self._server_thread is not None and self._server_thread.is_alive():
            self._server_thread.join(timeout=5)
            if self._server_thread.is_alive():
                logger.warning("HTTP endpoint thread did not exit within 5s")


# ----------------------------------------------
# Helper methods for _chat_completion_with_retry
# ----------------------------------------------


@dataclass
class AccumulatedResponse:
    content: str = ""
    logprobs_content: List[Any] = field(default_factory=list)
    token_ids: List[int] = field(default_factory=list)
    policy_steps: List[int] = field(default_factory=list)
    completion_tokens: int = 0
    routed_experts: np.ndarray | None = None
    route_prompt_ids: List[int] | None = None


def _prepare_retry_request(
    original_request_json: Dict[str, Any],
    accum: AccumulatedResponse,
    response_role: Optional[str],
    orig_max_tokens: Optional[int],
    max_key: Optional[str],
) -> Dict[str, Any]:
    """Build the per-iteration request payload.

    If no tokens have been generated yet, resend the original request unchanged.
    Otherwise, build a continuation request that appends the accumulated content
    and adjusts remaining max tokens if present.
    """
    if accum.completion_tokens == 0:
        return original_request_json.copy()

    assert accum.content != "", "accum.content must be non-empty for a continuation request"
    assert response_role is not None, "response_role must be set for a continuation request"

    cur_request_json = original_request_json.copy()
    cur_request_json["messages"] = original_request_json["messages"] + [
        {"role": response_role, "content": accum.content}
    ]
    cur_request_json["continue_final_message"] = True
    cur_request_json["add_generation_prompt"] = False
    if accum.route_prompt_ids is not None:
        cur_request_json[EXACT_PROMPT_TOKEN_IDS_KEY] = accum.route_prompt_ids + accum.token_ids
    elif EXACT_PROMPT_TOKEN_IDS_KEY in original_request_json:
        if len(accum.token_ids) != accum.completion_tokens:
            raise ValueError("Exact chat continuation requires token IDs for every sampled token")
        cur_request_json[EXACT_PROMPT_TOKEN_IDS_KEY] = (
            original_request_json[EXACT_PROMPT_TOKEN_IDS_KEY] + accum.token_ids
        )
    if orig_max_tokens is not None:
        assert orig_max_tokens - accum.completion_tokens >= 0, (
            "orig_max_tokens - accum.completion_tokens must be non-negative"
        )
        assert max_key is not None
        cur_request_json[max_key] = orig_max_tokens - accum.completion_tokens

    return cur_request_json


def _completion_response_was_aborted(response: Dict[str, Any]) -> bool:
    """Return whether the first completion choice has an abort finish reason."""
    if is_engine_error_response(response):
        return False
    choices = response.get("choices") or []
    return bool(choices) and choices[0].get("finish_reason") == ABORT_FINISH_REASON


def _completion_engine_error_response(results: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Wrap the first engine-side `/completions` error as a client-level error, else None.

    There is no partial success: if any sub-request errored, the whole request errors, following
    vLLM's and SGLang's behavior.
    """
    for result in results:
        if is_engine_error_response(result):
            error_details = result.get("error", result)
            return ErrorResponse(
                error=ErrorInfo(
                    message=f"In one of the engines that SkyRL manages, an error occurred: {error_details['message']}",
                    type=error_details["type"],
                    code=error_details["code"],
                ),
            ).model_dump()
    return None


def _parse_partial_response_and_inplace_update_accum(
    partial_response: Dict[str, Any],
    accum: AccumulatedResponse,
    response_role: Optional[str],
) -> tuple[str, Optional[str], Optional[str], bool]:
    """Parse the partial response and in-place update accumulators.

    Returns (finish_reason, stop_reason, response_role, aborted_without_generating).
    """
    choice = partial_response["choices"][0]
    finish_reason: str = choice["finish_reason"]
    stop_reason: Optional[str] = choice.get("stop_reason", None)
    new_content: Optional[str] = choice["message"]["content"]

    assert partial_response["usage"] is not None and partial_response["usage"]["completion_tokens"] is not None, (
        "partial_response['usage']['completion_tokens'] must be present"
    )
    new_completion_tokens: int = partial_response["usage"]["completion_tokens"]

    if response_role is None:
        response_role = choice["message"]["role"]
    else:
        assert response_role == choice["message"]["role"], "response_role must be the same across retries"

    # If aborted without generating tokens, ignore this partial response.
    aborted_without_generating = finish_reason == ABORT_FINISH_REASON and new_completion_tokens == 0
    if not aborted_without_generating:
        provider_fields = choice.get("provider_specific_fields") or {}
        routes = choice.get("routed_experts") or provider_fields.get("routed_experts")
        if routes is not None or accum.routed_experts is not None:
            if not isinstance(routes, str):
                raise ValueError("routed_experts capture is incomplete across chat retries")
            prompt_ids = partial_response.get("prompt_token_ids")
            response_ids = choice.get("token_ids")
            if not isinstance(prompt_ids, list) or not isinstance(response_ids, list):
                raise ValueError("routed_experts requires exact prompt and response token IDs on every retry")
            rows = decode_routed_experts(routes, len(prompt_ids) + len(response_ids) - 1)
            if accum.routed_experts is None:
                if accum.completion_tokens:
                    raise ValueError("routed_experts capture is incomplete across chat retries")
                accum.routed_experts = rows
                accum.route_prompt_ids = prompt_ids
            else:
                if prompt_ids != accum.route_prompt_ids + accum.token_ids:
                    raise ValueError("routed_experts retry prompt does not match accumulated token IDs")
                if rows.shape[1:] != accum.routed_experts.shape[1:]:
                    raise ValueError("routed_experts shape changed across chat retries")
                accum.routed_experts = np.concatenate((accum.routed_experts, rows[len(prompt_ids) - 1 :]))
        if new_content is not None:
            accum.content += new_content
        logprobs = choice.get("logprobs")
        if logprobs is not None and logprobs.get("content") is not None:
            accum.logprobs_content.extend(logprobs["content"])
        if choice.get("token_ids") is not None:
            accum.token_ids.extend(choice["token_ids"])
        if choice.get(CHOICE_POLICY_STEPS_KEY) is not None:
            accum.policy_steps.extend(choice[CHOICE_POLICY_STEPS_KEY])
        accum.completion_tokens += new_completion_tokens

    return finish_reason, stop_reason, response_role, aborted_without_generating


def _build_final_response(
    base_response: Dict[str, Any],
    accum: AccumulatedResponse,
    finish_reason: str,
    stop_reason: Optional[str],
) -> Dict[str, Any]:
    """Construct the final aggregated response from the base and accumulators."""
    # NOTE(Charlie): not doing deepcopy for performance. Be careful when re-using this method
    # as it mutates base_response.
    final_response = base_response

    # Combine usage: prompt_tokens from base, completion_tokens summed, total_tokens accordingly
    base_usage = final_response["usage"]
    prompt_tokens = base_usage["prompt_tokens"]
    final_usage = base_usage.copy()
    final_usage["completion_tokens"] = accum.completion_tokens
    final_usage["total_tokens"] = prompt_tokens + accum.completion_tokens
    final_response["usage"] = final_usage

    # Set accumulated content, logprobs, token_ids.
    final_choice = final_response["choices"][0]
    final_choice["message"]["content"] = accum.content
    if final_choice.get("logprobs", None) is not None:
        final_choice["logprobs"]["content"] = accum.logprobs_content
    if final_choice.get("token_ids", None) is not None:
        final_choice["token_ids"] = accum.token_ids
    if final_choice.get(CHOICE_POLICY_STEPS_KEY) is not None:
        final_choice[CHOICE_POLICY_STEPS_KEY] = accum.policy_steps
    if accum.routed_experts is not None:
        route_buffer = io.BytesIO()
        np.save(route_buffer, accum.routed_experts)
        encoded_routes = base64.b64encode(route_buffer.getvalue()).decode("ascii")
        if final_choice.get("routed_experts") is not None:
            final_choice["routed_experts"] = encoded_routes
        else:
            final_choice["provider_specific_fields"]["routed_experts"] = encoded_routes

    # Set last response's finish_reason and stop_reason.
    final_choice["finish_reason"] = finish_reason
    if stop_reason is not None:
        final_choice["stop_reason"] = stop_reason

    return final_response
