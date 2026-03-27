# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Child worker for packed vLLM rollout replicas."""

from __future__ import annotations

import argparse
import logging
import os
import socket
import threading
import time
import traceback
from dataclasses import dataclass, field
from typing import Any

from marin.rl.environments.inference_ctx.base import InferenceRequestKind
from .packed_vllm_protocol import (
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
)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-fd", type=int, required=True, help=argparse.SUPPRESS)
    parser.add_argument("--visible-chips", required=True, help=argparse.SUPPRESS)
    parser.add_argument("--worker-index", type=int, required=True, help=argparse.SUPPRESS)
    return parser.parse_args()


@dataclass
class _ChildRuntime:
    worker_index: int
    inference_ctx: Any
    transfer_client: Any
    cleanup_ctx: Any | None = None
    active_weight_id: int = -2
    pending_weight_id: int | None = None
    pending_state_dict: dict[str, Any] | None = None
    busy: bool = False
    total_weight_fetches: int = 0
    total_weight_activations: int = 0
    total_generate_requests: int = 0
    total_train_generate_requests: int = 0
    total_eval_generate_requests: int = 0
    total_micro_eval_generate_requests: int = 0
    last_generation_seconds: float | None = None
    error: str | None = None
    running: bool = True
    first_weights_ready: threading.Event = field(default_factory=threading.Event)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def status(self) -> PackedReplicaStatus:
        transfer_metrics = self.transfer_client.get_metrics()
        return PackedReplicaStatus(
            worker_index=self.worker_index,
            active_weight_id=self.active_weight_id,
            pending_weight_id=self.pending_weight_id,
            busy=self.busy,
            error=self.error,
            total_weight_fetches=self.total_weight_fetches,
            total_weight_activations=self.total_weight_activations,
            total_generate_requests=self.total_generate_requests,
            total_train_generate_requests=self.total_train_generate_requests,
            total_eval_generate_requests=self.total_eval_generate_requests,
            total_micro_eval_generate_requests=self.total_micro_eval_generate_requests,
            last_generation_seconds=self.last_generation_seconds,
            transfer_metrics=transfer_metrics,
        )


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    _configure_tpu_env(args.visible_chips)
    control = socket.socket(fileno=args.control_fd)

    try:
        runtime = _initialize_runtime(control, args.worker_index)
    except Exception:
        _send_error(control, "Packed vLLM child failed during initialization")
        raise

    sync_thread = threading.Thread(target=_weight_sync_loop, args=(runtime,), daemon=True, name="packed-vllm-sync")
    sync_thread.start()

    try:
        _serve(control, runtime)
    finally:
        runtime.running = False
        sync_thread.join(timeout=10.0)
        try:
            runtime.transfer_client.cleanup()
        except Exception:
            logger.exception("Failed to clean up packed child transfer client")
        try:
            runtime.inference_ctx.shutdown()
        except Exception:
            logger.exception("Failed to shut down packed child inference context")
        if runtime.cleanup_ctx is not None:
            try:
                runtime.cleanup_ctx.__exit__(None, None, None)
            except Exception:
                logger.exception("Failed to clean up packed child TPU lockfile context")
        control.close()


def _configure_tpu_env(visible_chips: str) -> None:
    chips = [chip.strip() for chip in visible_chips.split(",") if chip.strip()]
    if not chips:
        raise ValueError("visible_chips must not be empty")
    os.environ["TPU_PROCESS_BOUNDS"] = "1,1,1"
    os.environ["TPU_CHIPS_PER_PROCESS_BOUNDS"] = f"1,{len(chips)},1"
    os.environ["TPU_VISIBLE_CHIPS"] = visible_chips
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"


def _initialize_runtime(control: socket.socket, worker_index: int) -> _ChildRuntime:
    init_request = receive_packed_message(control)
    if not isinstance(init_request, PackedChildInitRequest):
        raise TypeError(f"Expected PackedChildInitRequest, got {type(init_request)}")

    from iris.logging import configure_logging
    from marin.rl.environments.inference_ctx.async_vllm import AsyncvLLMInferenceContext
    from marin.rl.weight_transfer import create_weight_transfer_client
    from marin.utils import remove_tpu_lockfile_on_exit

    configure_logging(level=logging.INFO)
    cleanup_ctx = remove_tpu_lockfile_on_exit()
    cleanup_ctx.__enter__()

    logger.info("Packed vLLM child %d starting on chips %s", worker_index, os.environ["TPU_VISIBLE_CHIPS"])

    inference_ctx = AsyncvLLMInferenceContext(init_request.inference_config)
    inference_ctx.start_server(None)
    transfer_client = create_weight_transfer_client(
        init_request.weight_transfer_config,
        mesh=inference_ctx.mesh,
        axis_mapping=inference_ctx.axis_mapping,
        coordinator_handle=init_request.coordinator_handle,
    )
    runtime = _ChildRuntime(
        worker_index=worker_index,
        inference_ctx=inference_ctx,
        transfer_client=transfer_client,
        cleanup_ctx=cleanup_ctx,
    )
    send_packed_message(control, PackedChildAckResponse(status=runtime.status()))
    return runtime


def _weight_sync_loop(runtime: _ChildRuntime) -> None:
    while runtime.running:
        try:
            update = runtime.transfer_client.receive_weights(None)
        except Exception:
            logger.exception("Packed child %d failed while fetching weights", runtime.worker_index)
            with runtime.lock:
                runtime.error = traceback.format_exc()
                runtime.running = False
            return

        if update is not None:
            with runtime.lock:
                runtime.pending_weight_id = update.weight_id
                runtime.pending_state_dict = update.state_dict
                runtime.total_weight_fetches += 1
                runtime.first_weights_ready.set()
                logger.info(
                    "Packed child %d fetched pending weight %d",
                    runtime.worker_index,
                    update.weight_id,
                )

        if not runtime.running:
            return
        time.sleep(1.0)


def _serve(control: socket.socket, runtime: _ChildRuntime) -> None:
    while runtime.running:
        request = receive_packed_message(control)
        try:
            response = _handle_request(runtime, request)
        except Exception:
            logger.exception("Packed child %d request failed", runtime.worker_index)
            runtime.error = traceback.format_exc()
            send_packed_message(
                control,
                PackedChildErrorResponse(
                    message=f"Packed child {runtime.worker_index} request failed",
                    status=runtime.status(),
                    traceback_text=runtime.error,
                ),
            )
            runtime.running = False
            return

        send_packed_message(control, response)
        if isinstance(request, PackedChildShutdownRequest):
            runtime.running = False
            return


def _handle_request(runtime: _ChildRuntime, request: object) -> object:
    if runtime.error is not None:
        return PackedChildErrorResponse(
            message=f"Packed child {runtime.worker_index} is in a failed state",
            status=runtime.status(),
            traceback_text=runtime.error,
        )

    if isinstance(request, PackedChildStatusRequest):
        return PackedChildStatusResponse(status=runtime.status())

    if isinstance(request, PackedChildActivateRequest):
        return _handle_activate(runtime, request)

    if isinstance(request, PackedChildGenerateRequest):
        return _handle_generate(runtime, request)

    if isinstance(request, PackedChildShutdownRequest):
        return PackedChildAckResponse(status=runtime.status())

    raise TypeError(f"Unsupported packed child request type: {type(request)}")


def _handle_activate(runtime: _ChildRuntime, request: PackedChildActivateRequest) -> PackedChildActivateResponse:
    with runtime.lock:
        if runtime.busy:
            return PackedChildActivateResponse(applied=False, status=runtime.status())
        if runtime.pending_weight_id != request.expected_weight_id or runtime.pending_state_dict is None:
            return PackedChildActivateResponse(applied=False, status=runtime.status())
        state_dict = runtime.pending_state_dict
        runtime.pending_state_dict = None
        runtime.pending_weight_id = None

    runtime.inference_ctx.reload_model(None, state_dict)

    with runtime.lock:
        runtime.active_weight_id = request.expected_weight_id
        runtime.total_weight_activations += 1
        logger.info(
            "Packed child %d activated weight %d",
            runtime.worker_index,
            request.expected_weight_id,
        )
        return PackedChildActivateResponse(applied=True, status=runtime.status())


def _handle_generate(runtime: _ChildRuntime, request: PackedChildGenerateRequest) -> PackedChildGenerateResponse:
    with runtime.lock:
        if runtime.busy:
            raise RuntimeError(f"Packed child {runtime.worker_index} received overlapping generate requests")
        if runtime.active_weight_id != request.expected_weight_id:
            raise RuntimeError(
                f"Packed child {runtime.worker_index} active weight {runtime.active_weight_id} "
                f"does not match expected {request.expected_weight_id}"
            )
        runtime.busy = True

    start = time.time()
    try:
        completions = runtime.inference_ctx.batch_completions(
            prompts=request.prompts,
            request_kind=request.request_kind,
            temperature=request.temperature,
            n=request.n,
            max_tokens=request.max_tokens,
            top_k=request.top_k,
            stop=request.stop,
            system_prompt=request.system_prompt,
        )
    finally:
        elapsed = time.time() - start
        with runtime.lock:
            runtime.busy = False
            runtime.total_generate_requests += 1
            if request.request_kind == InferenceRequestKind.TRAIN:
                runtime.total_train_generate_requests += 1
            elif request.request_kind == InferenceRequestKind.EVAL:
                runtime.total_eval_generate_requests += 1
            elif request.request_kind == InferenceRequestKind.MICRO_EVAL:
                runtime.total_micro_eval_generate_requests += 1
            runtime.last_generation_seconds = elapsed

    return PackedChildGenerateResponse(completions=completions, status=runtime.status())


def _send_error(control: socket.socket, message: str) -> None:
    try:
        send_packed_message(
            control,
            PackedChildErrorResponse(
                message=message,
                traceback_text=traceback.format_exc(),
            ),
        )
    except Exception:
        logger.exception("Failed to send packed child error response")


if __name__ == "__main__":
    main()
