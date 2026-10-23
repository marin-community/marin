from collections.abc import Collection
from dataclasses import dataclass
from typing import Dict, Any, Optional, Union
import random
import hashlib
import socket
from omegaconf import DictConfig, ListConfig
from skyrl_train.inference_engines.inference_engine_client_http_endpoint import ErrorResponse, ErrorInfo
from typing import List
from http import HTTPStatus
from typing import Tuple
import ray
from ray.actor import ActorHandle
from ray.util.placement_group import PlacementGroupSchedulingStrategy

VLLM_DATA_PARALLEL_MASTER_PORT_COUNT = 5


def get_vllm_sampling_params(sampling_params: DictConfig) -> Dict[str, Any]:
    stop_val = sampling_params.get("stop", None)
    vllm_sampling_params = {
        "min_tokens": sampling_params.get("min_tokens", 1),
        "skip_special_tokens": True,
        "include_stop_str_in_output": True,
        "max_tokens": sampling_params.max_generate_length,
        "temperature": sampling_params.temperature,
        "top_p": sampling_params.top_p,
        "top_k": sampling_params.top_k,
        "min_p": sampling_params.min_p,
        "logprobs": sampling_params.logprobs,
        "stop": list(stop_val) if stop_val is not None else None,
    }
    exclude_keys = ["max_generate_length"]
    for key, value in sampling_params.items():
        if key not in vllm_sampling_params and key not in exclude_keys:
            # Convert OmegaConf ListConfig to regular list if needed
            if isinstance(value, ListConfig):
                value = list(value)
            vllm_sampling_params[key] = value
    return vllm_sampling_params


def get_sglang_sampling_params(sampling_params: DictConfig) -> Dict[str, Any]:
    # `min_tokens` in vllm is equivalent to `min_new_tokens` in sglang. However `min_new_tokens` and
    # `stop` are not supported when `skip_tokenizer_init` is True, which we need for token-in-token-out.
    # See this issue for more: https://github.com/sgl-project/sglang/issues/9039#issuecomment-3218331087
    sglang_sampling_params = {
        "skip_special_tokens": True,
        "no_stop_trim": True,  # equivalent to include_stop_str_in_output=True
        "max_new_tokens": sampling_params.max_generate_length,
        "temperature": sampling_params.temperature,
        "top_p": sampling_params.top_p,
        "top_k": sampling_params.top_k,
        "min_p": sampling_params.min_p,
    }
    # logprobs not supported with sglang for now
    exclude_keys = ["max_generate_length", "logprobs"]
    for key, value in sampling_params.items():
        if key not in sglang_sampling_params and key not in exclude_keys:
            # Convert OmegaConf ListConfig to regular list if needed
            if isinstance(value, ListConfig):
                value = list(value)
            sglang_sampling_params[key] = value
    return sglang_sampling_params


def get_sampling_params_for_backend(backend: str, sampling_params: DictConfig) -> Dict[str, Any]:
    if backend == "vllm":
        return get_vllm_sampling_params(sampling_params)
    elif backend == "sglang":
        return get_sglang_sampling_params(sampling_params)
    else:
        raise ValueError(f"Unsupported generation backend: {backend}")


def hash_with_sha256(x: Union[int, str]) -> int:
    return int.from_bytes(hashlib.sha256(str(x).encode()).digest(), "big")


def route_prompts_to_engines(
    num_prompts: int, num_inference_engines: int, session_ids: Optional[Union[list[int], list[str]]]
) -> dict[int, list[int]]:
    """
    Given the number of prompts, number of inference engines, and the session_id, return a mapping
    from engine index to the list of prompt IDs the engine will process.

    Args:
    - num_prompts: int - The number of prompts.
    - num_inference_engines: int - The number of inference engines.
    - session_ids: Optional[Union[list[int], list[str]]] - The session IDs.

    Required:
    - num_prompts > 0
    - num_inference_engines > 0
    - session_ids is a list of integers or strings if provided
    - len(session_ids) == num_prompts if provided

    Returns:
    - dict[int, list[int]] - A mapping from engine index to the list of prompt IDs the engine will process.
    """
    # 0. Validation
    assert num_prompts > 0, "Number of prompts must be greater than 0"
    assert num_inference_engines > 0, "Number of inference engines must be greater than 0"
    if session_ids is not None:
        assert isinstance(session_ids, list) and all(isinstance(sid, (int, str)) for sid in session_ids), (
            "Session ID must be a list of integers or strings"
        )
        assert len(session_ids) == num_prompts, "Session ID must have the same length as the number of prompts"

    # 1. session_id not provided, with a single prompt: route to a random engine for a naive load balancing.
    if session_ids is None and num_prompts == 1:
        engine_idx = random.randint(0, num_inference_engines - 1)
        return {engine_idx: [0]}

    # 2. session_id not provided, with a batched prompt: split evenly across engines.
    engine_idx_to_prompt_ids: dict[int, list[int]] = {}
    if session_ids is None:
        dp_item_size = (num_prompts + num_inference_engines - 1) // num_inference_engines
        for dp_rank in range(num_inference_engines):
            start_idx = dp_rank * dp_item_size
            end_idx = min((dp_rank + 1) * dp_item_size, num_prompts)
            prompt_ids = list(range(start_idx, end_idx))
            if len(prompt_ids) > 0:
                engine_idx_to_prompt_ids[dp_rank] = prompt_ids
        return engine_idx_to_prompt_ids

    # 3. session_id provided, we route by session_id
    for i, cur_sid in enumerate(session_ids):
        engine_idx = hash_with_sha256(str(cur_sid)) % num_inference_engines
        engine_idx_to_prompt_ids.setdefault(engine_idx, []).append(i)
    return engine_idx_to_prompt_ids


def is_single_completion_prompt(prompt: Union[List[int], List[List[int]], List[str], str]) -> bool:
    """Return whether a completion prompt is one string or one flat token-ID list."""
    return isinstance(prompt, str) or _is_list_of_ints(prompt)


def _is_list_of_ints(x: Any) -> bool:
    return isinstance(x, list) and all(isinstance(y, int) for y in x)


def postprocess_completion_request(
    prompt: Union[List[int], List[List[int]], List[str], str],
    session_id_value: Optional[Union[List[int], List[str], int, str]],
) -> tuple[Optional[Union[List[int], List[str], ErrorResponse]], Union[List[List[int]], List[str]]]:
    """
    Postprocess the session_id value and raise error if invalid.

    Returns a list of session_ids, or None if session_id_value is None, or ErrorResponse if invalid.
    Also returns the processed prompt, where if the prompt is a single request, we make it
    a singleton list of a single request. That is, List[int] becomes List[List[int]] of length 1,
    and str becomes List[str] of length 1.

    Postconditions:
    - If session_id_value is None, we return None.
    - If session_id_value and prompt do not match, we return ErrorResponse.
    - The returned session_id_list has the same length as the prompt.
    - The returned prompt is either List[List[int]], or List[str], whether batched or not.
    """

    # Determine if this is a single or batched request (a List[str] of length 1 is considered batched)
    if is_single_completion_prompt(prompt):
        prompt = [prompt]

    if session_id_value is None:
        return None, prompt

    if isinstance(session_id_value, (int, str)):
        session_id_value = [session_id_value]

    if len(session_id_value) != len(prompt):
        return (
            ErrorResponse(
                error=ErrorInfo(
                    message=(
                        "For /completions request with a single prompt, request.session_id must "
                        f"be a single integer/string or a singleton list.\nFor batched requests, "
                        f"request.session_id must have the same length as request.prompt."
                        f"However, received (len(session_id_value): {len(session_id_value)}, len(prompt): {len(prompt)})."
                    ),
                    type=HTTPStatus.BAD_REQUEST.phrase,
                    code=HTTPStatus.BAD_REQUEST.value,
                ),
            ),
            prompt,
        )

    return session_id_value, prompt


def aggregate_completion_usage_info(
    results: List[Dict[str, Any]],
    backend: str,
) -> Dict[str, Any]:
    """
    Aggregate the completion usage info and return the final usage info, since our
    inference engine client breaks down a batched request into sub-requests and routes to engines,
    where each engine only returns its sub-request's usage info. When we return the final response,
    we need to aggregate the usage info.

    NOTE(Charlie): we don't explicitly import vllm here for ease of CPU test. Whether these fields
    are still compatible with newer vllm versions can be checked in our GPU tests, where we explicitly
    check the dictionary with `CompletionResponse.model_validate()`.
    """
    if backend == "vllm":
        # required fields
        usage_info = {
            "prompt_tokens": sum(result["usage"]["prompt_tokens"] for result in results),
            "total_tokens": sum(result["usage"]["total_tokens"] for result in results),
        }
        # optional fields
        if results[0]["usage"].get("completion_tokens") is not None:
            usage_info["completion_tokens"] = sum(result["usage"]["completion_tokens"] for result in results)
        if results[0]["usage"].get("prompt_tokens_details") is not None:
            if results[0]["usage"]["prompt_tokens_details"].get("cached_tokens") is not None:
                usage_info["prompt_tokens_details"] = {
                    "cached_tokens": sum(
                        result["usage"]["prompt_tokens_details"]["cached_tokens"] for result in results
                    )
                }
        return usage_info
    elif backend == "sglang":
        raise NotImplementedError("SGLang is not supported yet")
    else:
        raise ValueError(f"Unsupported backend: {backend}")


_RENDEZVOUS_PORT_START = 20_000
_RENDEZVOUS_PORT_STOP = 30_000


def _find_available_rendezvous_port(excluded_ports: Collection[int] = ()) -> int:
    """Choose a listener port outside Linux's default ephemeral client range."""

    excluded = set(excluded_ports)
    candidates = list(range(_RENDEZVOUS_PORT_START, _RENDEZVOUS_PORT_STOP))
    random.shuffle(candidates)
    for port in candidates:
        if port in excluded:
            continue
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(("", port))
            except OSError:
                continue
        return port
    raise RuntimeError(f"No free rendezvous port in [{_RENDEZVOUS_PORT_START}, {_RENDEZVOUS_PORT_STOP})")


def get_pg_bundle_node_ips(placement_group, pg_indices: Collection[int]) -> List[str]:
    """Return the node IP assigned to each requested placement-group bundle."""

    @ray.remote(num_cpus=0, num_gpus=0)
    def get_node_ip():
        return ray.util.get_node_ip_address()

    refs = [
        get_node_ip.options(
            scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group=placement_group,
                placement_group_capture_child_tasks=True,
                placement_group_bundle_index=idx,
            )
        ).remote()
        for idx in pg_indices
    ]
    return list(ray.get(refs))


def _reserve_available_rendezvous_ports(port_count: int, excluded_ports: Collection[int] = ()) -> list[socket.socket]:
    sockets: list[socket.socket] = []
    unavailable_ports = set(excluded_ports)
    try:
        for _ in range(port_count):
            port = _find_available_rendezvous_port(unavailable_ports)
            reserved = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            reserved.bind(("", port))
            sockets.append(reserved)
            unavailable_ports.add(port)
    except Exception:
        for reserved in sockets:
            reserved.close()
        raise
    return sockets


@ray.remote(num_cpus=0, num_gpus=0)
class RendezvousPortReservation:
    """Hold node-local listener ports until the vLLM coordinator starts."""

    def __init__(self, port_count: int, excluded_ports: Collection[int]):
        self._sockets = _reserve_available_rendezvous_ports(port_count, excluded_ports)

    def details(self) -> Tuple[str, list[int]]:
        """Return the node IP address and held port numbers."""
        return ray.util.get_node_ip_address(), [reserved.getsockname()[1] for reserved in self._sockets]

    def release(self) -> None:
        for reserved in self._sockets:
            reserved.close()
        self._sockets.clear()


@dataclass(frozen=True)
class ReservedRendezvousPorts:
    """Node-local ports held until the inference engine starts listening."""

    address: str
    ports: tuple[int, ...]
    reservation: ActorHandle


def reserve_rendezvous_ports(
    placement_group,
    pg_index: int,
    port_count: int,
    excluded_ports: Collection[int] = (),
) -> ReservedRendezvousPorts:
    """Reserve rendezvous ports on the node containing a placement-group bundle."""

    master_sched = PlacementGroupSchedulingStrategy(
        placement_group=placement_group,
        placement_group_capture_child_tasks=True,
        placement_group_bundle_index=pg_index,
    )
    reservation = RendezvousPortReservation.options(scheduling_strategy=master_sched).remote(port_count, excluded_ports)
    address, ports = ray.get(reservation.details.remote())
    return ReservedRendezvousPorts(address=address, ports=tuple(ports), reservation=reservation)
