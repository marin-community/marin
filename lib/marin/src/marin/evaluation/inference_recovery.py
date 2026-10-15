# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Keep evaluator attempts from running through inference outages."""

import logging
from collections.abc import Callable

from marin.inference.iris import InferenceBackendState, RemoteInferenceSession

logger = logging.getLogger(__name__)
# Stop evaluator attempts before their HTTP retries exhaust during a model reload.
BACKEND_POLL_SECONDS = 5.0


class InferenceUnavailable(RuntimeError):
    """An evaluator attempt must stop while its inference backends recover."""


class InferenceBackendFinished(RuntimeError):
    """Inference ended permanently while an evaluator depended on it."""


def inference_backend_state(sessions: tuple[RemoteInferenceSession, ...]) -> InferenceBackendState:
    """Combine dependency states, with terminal failure taking precedence over recovery."""
    states = tuple(session.backend_state() for session in sessions)
    if InferenceBackendState.FINISHED in states:
        return InferenceBackendState.FINISHED
    if InferenceBackendState.RECOVERING in states:
        return InferenceBackendState.RECOVERING
    return InferenceBackendState.READY


def check_inference_backend(state: InferenceBackendState) -> None:
    """Interrupt an attempt on a recoverable outage; fail on a terminal backend."""
    if state is InferenceBackendState.RECOVERING:
        raise InferenceUnavailable("inference backends are not ready")
    if state is InferenceBackendState.FINISHED:
        raise InferenceBackendFinished("inference backend finished while evaluation was running")


def run_with_inference_recovery[T](
    attempt: Callable[[], T],
    prepare_retry: Callable[[], None],
    sessions: tuple[RemoteInferenceSession, ...],
) -> T:
    """Wait through outages and retry with the adapter's durable progress preserved.

    The attempt must monitor dependencies and cancel its work before raising
    InferenceUnavailable. This function does not monitor a running attempt.
    After recovery, prepare_retry keeps reusable progress or selects a fresh archive.
    Other errors propagate without retry.
    """

    def check_dependencies() -> None:
        if inference_backend_state(sessions) is InferenceBackendState.FINISHED:
            check_inference_backend(InferenceBackendState.FINISHED)

    def wait_for_backends() -> None:
        for session in sessions:
            session.wait_until_ready(check_dependencies=check_dependencies)

    while True:
        state = inference_backend_state(sessions)
        if state is InferenceBackendState.RECOVERING:
            wait_for_backends()
        else:
            check_inference_backend(state)
        try:
            return attempt()
        except InferenceUnavailable:
            logger.warning("pausing evaluation while inference recovers")
            wait_for_backends()
            prepare_retry()
