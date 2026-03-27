# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Packed vLLM inference context for RL rollout workers."""

from __future__ import annotations

import concurrent.futures
import logging
import os
import socket
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any

import numpy as np
from levanter.compat.hf_checkpoints import load_tokenizer
from levanter.models.lm_model import LmHeadModel
from openai.types.chat.chat_completion import Choice

from marin.rl.environments.inference_ctx.base import BaseInferenceContext, InferenceRequestKind
from marin.rl.environments.inference_ctx.packed_vllm_protocol import (
    PackedChildAckResponse,
    PackedChildActivateRequest,
    PackedChildActivateResponse,
    PackedChildErrorResponse,
    PackedChildGenerateRequest,
    PackedChildGenerateResponse,
    PackedChildInitRequest,
    PackedChildShutdownRequest,
    PackedChildStatusRequest,
    PackedChildStatusResponse,
    PackedReplicaStatus,
    receive_packed_message,
    send_packed_message,
    status_to_metrics,
)
from marin.rl.environments.inference_ctx.staging import prepare_vllm_inference_config_for_inflight
from marin.rl.environments.inference_ctx.vllm import VLLMSamplingConfig, vLLMInferenceContextConfig
from marin.rl.weight_transfer import WeightTransferConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PackedvLLMInferenceContextConfig:
    """Configuration for one packed rollout worker with local TP=2 replicas."""

    model_name: str
    max_model_len: int
    tensor_parallel_size_per_replica: int
    gpu_memory_utilization: float
    sampling_params: VLLMSamplingConfig
    replica_chip_groups: tuple[str, ...]
    canonical_model_name: str | None = None
    load_format: str = "auto"
    enforce_eager: bool = True
    kv_cache_metrics: bool = False
    activation_timeout: float = 600.0
    startup_timeout: float = 600.0
    poll_interval: float = 1.0


@dataclass(frozen=True)
class PackedDispatchPlan:
    """Dispatch decision for one logical rollout batch."""

    dispatch_weight_id: int | None
    activate_weight_id: int | None = None


class PackedReplicaProcess:
    """Control handle for one packed vLLM child process."""

    def __init__(
        self,
        *,
        worker_index: int,
        visible_chips: str,
        inference_config: vLLMInferenceContextConfig,
        weight_transfer_config: WeightTransferConfig,
        coordinator_handle: object | None,
    ):
        self.worker_index = worker_index
        self.visible_chips = visible_chips
        self._inference_config = inference_config
        self._weight_transfer_config = weight_transfer_config
        self._coordinator_handle = coordinator_handle
        self._process: subprocess.Popen[bytes] | None = None
        self._socket: socket.socket | None = None
        self._lock = threading.Lock()

    def start(self, timeout: float) -> PackedReplicaStatus:
        parent_sock, child_sock = socket.socketpair()
        cmd = [
            sys.executable,
            "-u",
            "-m",
            "marin.rl.environments.inference_ctx.packed_vllm_worker",
            "--control-fd",
            str(child_sock.fileno()),
            "--visible-chips",
            self.visible_chips,
            "--worker-index",
            str(self.worker_index),
        ]
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        self._process = subprocess.Popen(
            cmd,
            env=env,
            pass_fds=[child_sock.fileno()],
            stdin=subprocess.DEVNULL,
        )
        child_sock.close()
        self._socket = parent_sock
        init_request = PackedChildInitRequest(
            inference_config=self._inference_config,
            weight_transfer_config=self._weight_transfer_config,
            coordinator_handle=self._coordinator_handle,
            worker_index=self.worker_index,
        )
        response = self._request(init_request, timeout=timeout)
        if not isinstance(response, PackedChildAckResponse):
            raise RuntimeError(f"Expected init ack from packed child {self.worker_index}, got {type(response)}")
        return response.status

    def status(self) -> PackedReplicaStatus:
        response = self._request(PackedChildStatusRequest())
        if not isinstance(response, PackedChildStatusResponse):
            raise RuntimeError(f"Expected status response from packed child {self.worker_index}, got {type(response)}")
        return response.status

    def activate(self, expected_weight_id: int) -> PackedChildActivateResponse:
        response = self._request(PackedChildActivateRequest(expected_weight_id=expected_weight_id))
        if not isinstance(response, PackedChildActivateResponse):
            raise RuntimeError(f"Expected activate response from packed child {self.worker_index}, got {type(response)}")
        return response

    def generate(
        self,
        *,
        request_id: str,
        prompts: list[str] | list[list[dict[str, Any]]],
        request_kind: InferenceRequestKind,
        temperature: float,
        n: int,
        max_tokens: int | None,
        top_k: int | None,
        stop: list[str] | None,
        system_prompt: str | None,
        expected_weight_id: int,
    ) -> PackedChildGenerateResponse:
        response = self._request(
            PackedChildGenerateRequest(
                request_id=request_id,
                prompts=prompts,
                request_kind=request_kind,
                temperature=temperature,
                n=n,
                max_tokens=max_tokens,
                top_k=top_k,
                stop=stop,
                system_prompt=system_prompt,
                expected_weight_id=expected_weight_id,
            )
        )
        if not isinstance(response, PackedChildGenerateResponse):
            raise RuntimeError(f"Expected generate response from packed child {self.worker_index}, got {type(response)}")
        return response

    def shutdown(self) -> None:
        try:
            if self._socket is not None and self.poll() is None:
                self._request(PackedChildShutdownRequest(), timeout=10.0)
        except Exception:
            logger.exception("Failed to send packed child shutdown request for replica %d", self.worker_index)
        finally:
            if self._socket is not None:
                self._socket.close()
                self._socket = None
            if self._process is not None and self._process.poll() is None:
                self._process.terminate()
                try:
                    self._process.wait(timeout=30.0)
                except subprocess.TimeoutExpired:
                    self._process.kill()
                    self._process.wait(timeout=30.0)

    def poll(self) -> int | None:
        if self._process is None:
            return None
        return self._process.poll()

    def _request(self, payload: object, timeout: float | None = None) -> object:
        if self._socket is None or self._process is None:
            raise RuntimeError(f"Packed child {self.worker_index} is not running")
        with self._lock:
            if self._process.poll() is not None:
                raise RuntimeError(f"Packed child {self.worker_index} exited with code {self._process.returncode}")
            previous_timeout = self._socket.gettimeout()
            self._socket.settimeout(timeout)
            try:
                send_packed_message(self._socket, payload)
                response = receive_packed_message(self._socket)
            finally:
                self._socket.settimeout(previous_timeout)

        if isinstance(response, PackedChildErrorResponse):
            detail = response.message
            if response.traceback_text:
                detail = f"{detail}\n{response.traceback_text}"
            raise RuntimeError(detail)
        return response


class PackedvLLMInferenceContext(BaseInferenceContext):
    """Parent inference context that multiplexes two local TP=2 vLLM children."""

    def __init__(
        self,
        inference_config: PackedvLLMInferenceContextConfig,
        *,
        inflight_weight_updates: bool,
        weight_transfer_config: WeightTransferConfig,
        coordinator_handle: object | None,
    ):
        if not inflight_weight_updates:
            raise ValueError("Packed vLLM rollout requires inflight_weight_updates=True")

        self.config = inference_config
        self.mesh = None
        self.axis_mapping = {}
        self._coordinator_handle = coordinator_handle
        self._weight_transfer_config = weight_transfer_config
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=len(inference_config.replica_chip_groups))
        self._replicas: list[PackedReplicaProcess] = []
        self._last_train_dispatch_weight_id = -2
        self._last_eval_dispatch_weight_id = -2
        self._last_statuses: list[PackedReplicaStatus] = []
        self._reservation_condition = threading.Condition()
        self._replica_request_kinds: dict[int, InferenceRequestKind | None] = {
            index: None for index in range(len(inference_config.replica_chip_groups))
        }
        self._eval_waiters = 0
        self._started = False

        self._validate_config(inference_config)
        self._per_replica_inference_config = prepare_vllm_inference_config_for_inflight(
            vLLMInferenceContextConfig(
                model_name=inference_config.model_name,
                canonical_model_name=inference_config.canonical_model_name,
                max_model_len=inference_config.max_model_len,
                tensor_parallel_size=inference_config.tensor_parallel_size_per_replica,
                gpu_memory_utilization=inference_config.gpu_memory_utilization,
                sampling_params=inference_config.sampling_params,
                load_format=inference_config.load_format,
                enforce_eager=inference_config.enforce_eager,
                kv_cache_metrics=inference_config.kv_cache_metrics,
            )
        )
        self.tokenizer = load_tokenizer(self._per_replica_inference_config.model_name)

    def owns_weight_transfer(self) -> bool:
        return True

    def start_server(self, model: LmHeadModel | None) -> None:
        del model
        if self._started:
            return

        try:
            for worker_index, chip_group in enumerate(self.config.replica_chip_groups):
                replica = PackedReplicaProcess(
                    worker_index=worker_index,
                    visible_chips=chip_group,
                    inference_config=self._per_replica_inference_config,
                    weight_transfer_config=self._weight_transfer_config,
                    coordinator_handle=self._coordinator_handle,
                )
                status = replica.start(timeout=self.config.startup_timeout)
                self._replicas.append(replica)
                self._last_statuses.append(status)
                logger.info(
                    "Packed rollout child %d started on chips %s (active=%d)",
                    worker_index,
                    chip_group,
                    status.active_weight_id,
                )
        except Exception:
            self.shutdown()
            raise

        self._started = True

    def wait_for_initial_weights(self, timeout: float) -> int | None:
        dispatch_weight_id = self._resolve_dispatch_weight(timeout=timeout, replica_indices=(0, 1))
        self._last_train_dispatch_weight_id = dispatch_weight_id
        return dispatch_weight_id

    def current_weight_id(self, request_kind: InferenceRequestKind = InferenceRequestKind.TRAIN) -> int | None:
        if request_kind == InferenceRequestKind.TRAIN:
            return self._last_train_dispatch_weight_id
        return self._last_eval_dispatch_weight_id

    def reload_model(self, model: LmHeadModel | None, state_dict: dict) -> LmHeadModel | None:
        del model, state_dict
        raise RuntimeError("Packed vLLM rollout manages weight updates inside child replicas")

    def shutdown(self) -> None:
        with self._reservation_condition:
            for replica_index in self._replica_request_kinds:
                self._replica_request_kinds[replica_index] = None
            self._reservation_condition.notify_all()
        for replica in reversed(self._replicas):
            replica.shutdown()
        self._replicas.clear()
        self._executor.shutdown(wait=True)
        self._started = False

    def get_metrics(self) -> dict[str, float]:
        statuses = self._statuses()
        with self._reservation_condition:
            reservation_snapshot = dict(self._replica_request_kinds)
            eval_active = any(
                request_kind in {InferenceRequestKind.EVAL, InferenceRequestKind.MICRO_EVAL}
                for request_kind in reservation_snapshot.values()
            )
        metrics: dict[str, float] = {
            "packed/replica_count": float(len(statuses)),
            "packed/last_dispatch_weight_id": float(self._last_train_dispatch_weight_id),
            "packed/last_train_dispatch_weight_id": float(self._last_train_dispatch_weight_id),
            "packed/last_eval_dispatch_weight_id": float(self._last_eval_dispatch_weight_id),
            "packed/eval_active": float(int(eval_active)),
        }
        for replica_index, request_kind in reservation_snapshot.items():
            metrics[f"packed/replica_{replica_index}/reserved"] = float(int(request_kind is not None))
            metrics[f"packed/replica_{replica_index}/eval_reserved"] = float(
                int(request_kind in {InferenceRequestKind.EVAL, InferenceRequestKind.MICRO_EVAL})
            )
        for status in statuses:
            metrics.update(status_to_metrics(f"packed/replica_{status.worker_index}", status))
        return metrics

    def tokenize_prompt(self, prompt: str, choice: Choice | None = None, system_prompt: str | None = None) -> np.ndarray:
        del prompt, system_prompt
        if choice is None or not hasattr(choice, "prompt_token_ids"):
            raise ValueError("Packed vLLM rollout requires prompt_token_ids from vLLM choices")
        return np.array(choice.prompt_token_ids, dtype=np.int32)

    def response_tokens_from_choice(self, choice: Choice) -> np.ndarray:
        if not hasattr(choice, "response_token_ids"):
            raise ValueError("Packed vLLM rollout requires response_token_ids from vLLM choices")
        return np.array(choice.response_token_ids, dtype=np.int32)

    def batch_completions(
        self,
        prompts: list[str] | list[list[dict[str, Any]]],
        request_kind: InferenceRequestKind,
        temperature: float,
        n: int,
        max_tokens: int | None = None,
        top_k: int | None = None,
        stop: list[str] | None = None,
        system_prompt: str | None = None,
    ) -> list[object]:
        replica_indices: tuple[int, ...] = ()
        try:
            replica_indices = self._reserve_replicas(request_kind)
            dispatch_weight_id = self._resolve_dispatch_weight(
                timeout=self.config.activation_timeout,
                replica_indices=replica_indices,
            )
            selected_replicas = [self._replicas[replica_index] for replica_index in replica_indices]
            shards = split_packed_prompt_batch(prompts, len(selected_replicas))

            futures: list[tuple[int, concurrent.futures.Future[PackedChildGenerateResponse]]] = []
            request_id_prefix = uuid.uuid4().hex[:8]
            for replica, shard in zip(selected_replicas, shards, strict=True):
                indices, shard_prompts = shard
                if not indices:
                    continue
                future = self._executor.submit(
                    replica.generate,
                    request_id=f"{request_id_prefix}-{replica.worker_index}",
                    prompts=shard_prompts,
                    request_kind=request_kind,
                    temperature=temperature,
                    n=n,
                    max_tokens=max_tokens,
                    top_k=top_k,
                    stop=stop,
                    system_prompt=system_prompt,
                    expected_weight_id=dispatch_weight_id,
                )
                futures.append((replica.worker_index, future))

            completion_shards: list[tuple[list[int], list[object]]] = []
            for (indices, _), (worker_index, future) in zip(
                [shard for shard in shards if shard[0]],
                futures,
                strict=True,
            ):
                response = future.result()
                self._replace_status(response.status)
                completion_shards.append((indices, response.completions))
                logger.info(
                    "Packed rollout child %d generated %d prompts for %s at weight %d",
                    worker_index,
                    len(indices),
                    request_kind,
                    dispatch_weight_id,
                )
        except Exception:
            self.shutdown()
            raise
        finally:
            self._release_replicas(replica_indices)

        if request_kind == InferenceRequestKind.TRAIN:
            self._last_train_dispatch_weight_id = dispatch_weight_id
        else:
            self._last_eval_dispatch_weight_id = dispatch_weight_id
        return merge_packed_completion_shards(completion_shards, len(prompts))

    def _statuses(self, replica_indices: tuple[int, ...] | None = None) -> list[PackedReplicaStatus]:
        target_replica_indices = replica_indices or tuple(range(len(self._replicas)))
        statuses = [self._replicas[replica_index].status() for replica_index in target_replica_indices]
        for status in statuses:
            self._replace_status(status)
        return statuses

    def _replace_status(self, updated_status: PackedReplicaStatus) -> None:
        for index, status in enumerate(self._last_statuses):
            if status.worker_index == updated_status.worker_index:
                self._last_statuses[index] = updated_status
                return
        self._last_statuses.append(updated_status)

    def _resolve_dispatch_weight(self, timeout: float, replica_indices: tuple[int, ...]) -> int:
        deadline = time.monotonic() + timeout
        selected_replicas = [self._replicas[replica_index] for replica_index in replica_indices]
        while True:
            statuses = self._statuses(replica_indices)
            plan = choose_packed_dispatch_plan(statuses)
            if plan.activate_weight_id is not None:
                futures = [
                    self._executor.submit(replica.activate, plan.activate_weight_id) for replica in selected_replicas
                ]
                responses = [future.result() for future in futures]
                for response in responses:
                    self._replace_status(response.status)
                if all(response.applied for response in responses):
                    return plan.activate_weight_id
            elif plan.dispatch_weight_id is not None:
                return plan.dispatch_weight_id

            if time.monotonic() >= deadline:
                status_summary = ", ".join(
                    (
                        f"replica_{status.worker_index}("
                        f"active={status.active_weight_id}, pending={status.pending_weight_id})"
                    )
                    for status in statuses
                )
                raise TimeoutError(f"Timed out waiting for synchronized packed rollout weights: {status_summary}")
            time.sleep(self.config.poll_interval)

    def _reserve_replicas(self, request_kind: InferenceRequestKind) -> tuple[int, ...]:
        count_as_eval_waiter = request_kind in {InferenceRequestKind.EVAL, InferenceRequestKind.MICRO_EVAL}
        with self._reservation_condition:
            if count_as_eval_waiter:
                self._eval_waiters += 1
            try:
                while True:
                    replica_indices = choose_packed_target_replica_indices(
                        request_kind=request_kind,
                        reserved_request_kinds=self._replica_request_kinds,
                        eval_waiters=self._eval_waiters,
                    )
                    if replica_indices is not None:
                        for replica_index in replica_indices:
                            self._replica_request_kinds[replica_index] = request_kind
                        return replica_indices
                    self._reservation_condition.wait(timeout=self.config.poll_interval)
            finally:
                if count_as_eval_waiter:
                    self._eval_waiters -= 1

    def _release_replicas(self, replica_indices: tuple[int, ...]) -> None:
        with self._reservation_condition:
            for replica_index in replica_indices:
                self._replica_request_kinds[replica_index] = None
            self._reservation_condition.notify_all()

    @staticmethod
    def _validate_config(inference_config: PackedvLLMInferenceContextConfig) -> None:
        if len(inference_config.replica_chip_groups) != 2:
            raise ValueError("Packed vLLM rollout currently requires exactly two replica chip groups")
        for chip_group in inference_config.replica_chip_groups:
            chips = [chip.strip() for chip in chip_group.split(",") if chip.strip()]
            if len(chips) != inference_config.tensor_parallel_size_per_replica:
                raise ValueError(
                    f"Chip group {chip_group!r} must contain exactly "
                    f"{inference_config.tensor_parallel_size_per_replica} chips"
                )


def split_packed_prompt_batch(
    prompts: list[str] | list[list[dict[str, Any]]],
    num_replicas: int,
) -> list[tuple[list[int], list[str] | list[list[dict[str, Any]]]]]:
    """Split prompts into contiguous shards while preserving original order."""
    total = len(prompts)
    base = total // num_replicas
    remainder = total % num_replicas
    shards: list[tuple[list[int], list[str] | list[list[dict[str, Any]]]]] = []
    start = 0
    for replica_index in range(num_replicas):
        shard_size = base + (1 if replica_index < remainder else 0)
        end = start + shard_size
        indices = list(range(start, end))
        shards.append((indices, prompts[start:end]))
        start = end
    return shards


def merge_packed_completion_shards(
    completion_shards: list[tuple[list[int], list[object]]],
    total_prompts: int,
) -> list[object]:
    """Merge per-replica completion shards back into original prompt order."""
    merged: list[object | None] = [None] * total_prompts
    for indices, completions in completion_shards:
        if len(indices) != len(completions):
            raise ValueError("Packed completion shard length mismatch")
        for index, completion in zip(indices, completions, strict=True):
            merged[index] = completion
    if any(completion is None for completion in merged):
        raise ValueError("Packed completion merge left gaps in the prompt ordering")
    return [completion for completion in merged if completion is not None]


def choose_packed_dispatch_plan(statuses: list[PackedReplicaStatus]) -> PackedDispatchPlan:
    """Choose the next synchronized weight step to dispatch."""
    errors = [status.error for status in statuses if status.error]
    if errors:
        raise RuntimeError("; ".join(errors))

    active_ids = {status.active_weight_id for status in statuses}
    if len(active_ids) != 1:
        raise RuntimeError(f"Packed rollout replicas disagree on active weights: {sorted(active_ids)}")
    active_weight_id = next(iter(active_ids))

    shared_pending_weight = statuses[0].pending_weight_id
    if shared_pending_weight is not None and all(
        status.pending_weight_id == shared_pending_weight and not status.busy for status in statuses
    ):
        if shared_pending_weight != active_weight_id:
            return PackedDispatchPlan(
                dispatch_weight_id=shared_pending_weight,
                activate_weight_id=shared_pending_weight,
            )

    if active_weight_id >= -1:
        return PackedDispatchPlan(dispatch_weight_id=active_weight_id)
    return PackedDispatchPlan(dispatch_weight_id=None)


def choose_packed_target_replica_indices(
    *,
    request_kind: InferenceRequestKind,
    reserved_request_kinds: dict[int, InferenceRequestKind | None],
    eval_waiters: int,
) -> tuple[int, ...] | None:
    """Choose which local packed replicas should serve the next request."""
    replica_0_request = reserved_request_kinds[0]
    replica_1_request = reserved_request_kinds[1]

    if request_kind == InferenceRequestKind.TRAIN:
        if replica_0_request is not None:
            return None
        if replica_1_request is None and eval_waiters == 0:
            return (0, 1)
        return (0,)

    if request_kind in {InferenceRequestKind.EVAL, InferenceRequestKind.MICRO_EVAL}:
        if replica_1_request is not None:
            return None
        return (1,)

    raise ValueError(f"Unsupported packed request kind: {request_kind}")
