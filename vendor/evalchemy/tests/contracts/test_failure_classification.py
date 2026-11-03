"""Classification of escaped endpoint-transport errors onto the failure taxonomy.

An HTTP client error raised against the model endpoint (connection reset, timeout,
or any status the client surfaced) is model transport, not grader infrastructure:
the taxonomy's job is to point the debugging finger at the right component. A
500-only endpoint drove whole tasks to ``grader_infrastructure`` because only the
builtin ``ConnectionError`` mapped to ``model_transport``.
"""

import aiohttp
import pytest
import requests

from eval.contracts.failures import FailureCategory, FailurePhase, classify_task_exception


def _http_error():
    return requests.exceptions.HTTPError("500 Server Error: Internal Server Error")


def _requests_connection_error():
    return requests.exceptions.ConnectionError("connection refused")


def _aiohttp_response_error(status=502):
    return aiohttp.ClientResponseError(
        request_info=None,
        history=(),
        status=status,
        message="Bad Gateway",
    )


@pytest.mark.parametrize(
    "error",
    [
        ConnectionError("connection reset"),
        _http_error(),
        _requests_connection_error(),
        _aiohttp_response_error(502),
        _aiohttp_response_error(404),
    ],
)
def test_evaluation_phase_endpoint_transport_errors_are_model_transport(error):
    assert classify_task_exception(FailurePhase.EVALUATION, error) is FailureCategory.MODEL_TRANSPORT


def test_generation_phase_endpoint_transport_errors_are_model_transport():
    assert classify_task_exception(FailurePhase.GENERATION, _requests_connection_error()) is (
        FailureCategory.MODEL_TRANSPORT
    )


def test_grading_phase_endpoint_errors_stay_grader_infrastructure():
    # The grading phase talks to the judge, so an HTTP error there is the
    # grader's infrastructure, never model transport.
    assert classify_task_exception(FailurePhase.GRADING, _http_error()) is (FailureCategory.GRADER_INFRASTRUCTURE)


def test_evaluation_phase_unrelated_errors_stay_grader_infrastructure():
    assert classify_task_exception(FailurePhase.EVALUATION, ValueError("bad result document")) is (
        FailureCategory.GRADER_INFRASTRUCTURE
    )


def test_timeout_classifies_as_agent_timeout_before_transport():
    import requests

    assert classify_task_exception(FailurePhase.EVALUATION, TimeoutError()) is FailureCategory.AGENT_TIMEOUT
    assert classify_task_exception(FailurePhase.EVALUATION, requests.exceptions.Timeout("t")) is (
        FailureCategory.MODEL_TRANSPORT
    )
