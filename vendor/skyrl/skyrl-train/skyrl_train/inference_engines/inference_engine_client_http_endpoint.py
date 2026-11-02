"""
OpenAI-compatible HTTP endpoint using InferenceEngineClient as backend.

This module provides a FastAPI-based HTTP endpoint that exposes OpenAI's completion APIs and
vLLM-compatible chat tokenization while routing requests to InferenceEngineClient.

Main functions:
- serve(): Start the HTTP endpoint.
- wait_for_server_ready(): Wait for server to be ready.
- shutdown_server(): Shutdown the server.
"""

import asyncio
import json
import logging
import time
import requests
import traceback
from contextlib import asynccontextmanager
from enum import StrEnum
from http import HTTPStatus
from typing import Any, Coroutine, Dict, Optional, TypeVar

import fastapi
import uvicorn
from fastapi import Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from skyrl_train.inference_engines.inference_http_backend import InferenceHTTPBackend
from skyrl_train.inference_engines.vllm.stats import HTTPBridgeStatsAccumulator


logger = logging.getLogger(__name__)

_ResponseT = TypeVar("_ResponseT")
TOKENIZE_ENDPOINT = "/tokenize"
MODELS_ENDPOINT = "/v1/models"
CHAT_COMPLETIONS_ENDPOINT = "/v1/chat/completions"
COMPLETIONS_ENDPOINT = "/v1/completions"
_INFERENCE_ENDPOINTS = frozenset({TOKENIZE_ENDPOINT, CHAT_COMPLETIONS_ENDPOINT, COMPLETIONS_ENDPOINT})
_SERVER_CREATED_TIME = int(time.time())


# Global state to hold the inference engine client and backend
_global_inference_engine_client: Optional[InferenceHTTPBackend] = None
_global_uvicorn_server: Optional[uvicorn.Server] = None


# Mirrors vLLM's nested OpenAI error response shape.
class ErrorInfo(BaseModel):
    message: str
    type: str
    param: Optional[str] = None
    code: int


class ErrorResponse(BaseModel):
    error: ErrorInfo


class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str = "skyrl"
    max_model_len: Optional[int]


class ModelList(BaseModel):
    object: str = "list"
    data: list[ModelCard]


class _RequestOutcome(StrEnum):
    INCOMPLETE = "incomplete"
    COMPLETED = "completed"
    CLIENT_DISCONNECT = "client_disconnect"
    SEND_FAILURE = "send_failure"
    SERVER_CANCELLED = "server_cancelled"
    APPLICATION_ERROR = "application_error"


class _RequestOutcomeMiddleware:
    """Record the close outcome visible at the ASGI boundary for inference requests."""

    def __init__(self, app, bridge_stats: HTTPBridgeStatsAccumulator):
        self.app = app
        self.bridge_stats = bridge_stats

    async def __call__(self, scope, receive, send):
        endpoint = scope.get("path")
        if scope["type"] != "http" or endpoint not in _INFERENCE_ENDPOINTS:
            return await self.app(scope, receive, send)

        reason = _RequestOutcome.INCOMPLETE

        async def observed_receive():
            nonlocal reason
            message = await receive()
            if message["type"] == "http.disconnect":
                reason = _RequestOutcome.CLIENT_DISCONNECT
            return message

        async def observed_send(message):
            nonlocal reason
            try:
                await send(message)
            except OSError:
                reason = _RequestOutcome.SEND_FAILURE
                raise
            if (
                reason is _RequestOutcome.INCOMPLETE
                and message["type"] == "http.response.body"
                and not message.get("more_body", False)
            ):
                reason = _RequestOutcome.COMPLETED

        try:
            await self.app(scope, observed_receive, observed_send)
        except asyncio.CancelledError:
            if reason is _RequestOutcome.INCOMPLETE:
                reason = _RequestOutcome.SERVER_CANCELLED
            raise
        except Exception:
            if reason is _RequestOutcome.INCOMPLETE:
                reason = _RequestOutcome.APPLICATION_ERROR
            raise
        finally:
            self.bridge_stats.record_request_outcome(endpoint, reason.value)


def is_engine_error_response(response: Dict[str, Any]) -> bool:
    """Recognize the error response shapes returned by vLLM and SGLang."""
    return "error" in response or response.get("object", "") == "error"


def set_global_state(inference_engine_client: InferenceHTTPBackend, uvicorn_server: uvicorn.Server | None):
    """Set the global inference engine client."""
    global _global_inference_engine_client
    global _global_uvicorn_server
    _global_inference_engine_client = inference_engine_client
    _global_uvicorn_server = uvicorn_server


def _validate_backend_model(model: Optional[str]) -> Optional[ErrorResponse]:
    if _global_inference_engine_client is None:
        return ErrorResponse(
            error=ErrorInfo(
                message="Inference engine client not initialized",
                type=HTTPStatus.INTERNAL_SERVER_ERROR.phrase,
                code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
            ),
        )
    if model is not None and _global_inference_engine_client.model_name != model:
        # `served_model_name` is supported in generator.engine_init_kwargs. Both
        # vllm_engine.py and InferenceEngineClient use it for Harbor/LiteLLM compatibility.
        return ErrorResponse(
            error=ErrorInfo(
                message=f"Model name mismatch: loaded model name {_global_inference_engine_client.model_name} "
                f"!= model name in request {model}",
                type=HTTPStatus.BAD_REQUEST.phrase,
                code=HTTPStatus.BAD_REQUEST.value,
            ),
        )
    return None


def _validate_openai_request(request_json: Dict[str, Any], endpoint: str) -> Optional[ErrorResponse]:
    """Common validation for /chat/completions and /completions endpoints."""
    assert endpoint in ["/completions", "/chat/completions"]

    backend_error = _validate_backend_model(request_json.get("model"))
    if backend_error is not None:
        return backend_error
    if "model" not in request_json:
        return ErrorResponse(
            error=ErrorInfo(
                message=f"The field `model` is required in your `{endpoint}` request.",
                type=HTTPStatus.BAD_REQUEST.phrase,
                code=HTTPStatus.BAD_REQUEST.value,
            ),
        )
    if endpoint == "/completions" and "n" in request_json and request_json["n"] > 1:
        # TODO(Charlie): this constraint can be removed when we leave DP routing to
        # inference frameworks. Or we could try to resolve it when needed.
        return ErrorResponse(
            error=ErrorInfo(
                message="n is not supported in SkyRL for /completions request yet, please set n to 1.",
                type=HTTPStatus.BAD_REQUEST.phrase,
                code=HTTPStatus.BAD_REQUEST.value,
            ),
        )
    if endpoint == "/chat/completions" and "messages" not in request_json:
        return ErrorResponse(
            error=ErrorInfo(
                message="The field `messages` is required in your `/chat/completions` request.",
                type=HTTPStatus.BAD_REQUEST.phrase,
                code=HTTPStatus.BAD_REQUEST.value,
            ),
        )
    if endpoint == "/chat/completions" and request_json["messages"] == []:
        return ErrorResponse(
            error=ErrorInfo(
                message="The field `messages` in `/chat/completions` cannot be an empty list.",
                type=HTTPStatus.BAD_REQUEST.phrase,
                code=HTTPStatus.BAD_REQUEST.value,
            ),
        )
    return None


def _status_class(status_code: int) -> str:
    return f"{status_code // 100}xx"


def _json_response(
    content: Any,
    *,
    endpoint: str,
    bridge_stats: HTTPBridgeStatsAccumulator,
    status_code: int = HTTPStatus.OK.value,
) -> JSONResponse:
    started = time.perf_counter()
    response = JSONResponse(content=content, status_code=status_code)
    serialization_seconds = time.perf_counter() - started
    attributes = {"endpoint": endpoint, "transport": "json", "status": _status_class(status_code)}
    bridge_stats.observe("json_serialization_seconds", serialization_seconds, attributes=attributes)
    bridge_stats.observe("response_bytes", float(len(response.body)), attributes=attributes)
    return response


async def _safe_sse_stream(generator, *, endpoint: str, bridge_stats: HTTPBridgeStatsAccumulator):
    """Wrap an async generator so the SSE body always completes.

    Guarantees ``data: [DONE]\\n\\n`` is the last chunk even when the
    underlying generator raises mid-stream (Ray actor death, vLLM error,
    serialization failure, etc.).  Without this, ``StreamingResponse``
    closes the connection on any unhandled exception, producing
    ``RemoteProtocolError: incomplete chunked read`` on the client side.
    """
    done_sent = False
    response_bytes = 0
    status = HTTPStatus.OK.value
    try:
        async for chunk in generator:
            if isinstance(chunk, str) and "[DONE]" in chunk:
                done_sent = True
            response_bytes += len(chunk.encode()) if isinstance(chunk, str) else len(chunk)
            yield chunk
    except asyncio.CancelledError:
        # Client disconnected — re-raise so Starlette stops sending.
        raise
    except Exception as e:
        status = HTTPStatus.INTERNAL_SERVER_ERROR.value
        tb = traceback.format_exc()
        logger.error(f"Mid-stream SSE error:\n{tb}")
        if not done_sent:
            err = ErrorResponse(
                error=ErrorInfo(
                    message=f"Streaming error: {str(e)}",
                    type=HTTPStatus.INTERNAL_SERVER_ERROR.phrase,
                    code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
                ),
            ).model_dump()
            error_chunk = f"data: {json.dumps(err)}\n\n"
            done_chunk = "data: [DONE]\n\n"
            response_bytes += len(error_chunk.encode()) + len(done_chunk.encode())
            yield error_chunk
            yield done_chunk
        return
    else:
        # Normal completion — ensure [DONE] if the generator didn't send it.
        if not done_sent:
            done_chunk = "data: [DONE]\n\n"
            response_bytes += len(done_chunk.encode())
            yield done_chunk
    finally:
        bridge_stats.observe(
            "response_bytes",
            float(response_bytes),
            attributes={"endpoint": endpoint, "transport": "sse", "status": _status_class(status)},
        )


async def _wait_for_disconnect(raw_request: Request) -> None:
    """Wait for the ASGI server to report that the HTTP client disconnected."""
    while True:
        message = await raw_request.receive()
        if message["type"] == "http.disconnect":
            return


async def _await_with_disconnect(raw_request: Request, backend_request: Coroutine[Any, Any, _ResponseT]) -> _ResponseT:
    """Cancel backend work when its HTTP client abandons the request."""
    backend_task = asyncio.create_task(backend_request)
    disconnect_task = asyncio.create_task(_wait_for_disconnect(raw_request))
    try:
        done, _ = await asyncio.wait(
            {backend_task, disconnect_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        if backend_task in done:
            return await backend_task
        raise asyncio.CancelledError
    finally:
        backend_task.cancel()
        disconnect_task.cancel()
        await asyncio.gather(backend_task, disconnect_task, return_exceptions=True)


async def handle_openai_request(
    raw_request: Request,
    endpoint: str,
    bridge_stats: HTTPBridgeStatsAccumulator,
):
    """Handle a request implemented by the policy model's serving backend.

    Returns ``StreamingResponse`` for ``stream:true`` chat-completions and
    ``JSONResponse`` for everything else (byte-identical to pre-streaming behavior).
    """
    assert endpoint in ["/completions", "/chat/completions", TOKENIZE_ENDPOINT]
    try:
        request_json = await raw_request.json()

        # SkyRL-side validation
        if endpoint == TOKENIZE_ENDPOINT:
            error_response = _validate_backend_model(request_json.get("model"))
        else:
            error_response = _validate_openai_request(request_json, endpoint=endpoint)
        if error_response is not None:
            return _json_response(
                error_response.model_dump(),
                endpoint=endpoint,
                bridge_stats=bridge_stats,
                status_code=error_response.error.code,
            )

        # Serialize fastapi.Request because it is not pickable, which causes ray methods to fail.
        payload = {
            "json": request_json,
            "headers": dict(raw_request.headers) if hasattr(raw_request, "headers") else {},
        }

        # ── Streaming branch ──────────────────────────────────────────────
        if request_json.get("stream", False) and endpoint == "/chat/completions":
            raw_gen = _global_inference_engine_client.chat_completion_stream(payload)
            return StreamingResponse(
                content=_safe_sse_stream(raw_gen, endpoint=endpoint, bridge_stats=bridge_stats),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        # Non-streaming requests stay attached to the client until completion.
        if endpoint == "/chat/completions":
            backend_request = _global_inference_engine_client.chat_completion(payload)
        elif endpoint == "/completions":
            backend_request = _global_inference_engine_client.completion(payload)
        else:
            backend_request = _global_inference_engine_client.tokenize(payload)
        response = await _await_with_disconnect(raw_request, backend_request)

        if is_engine_error_response(response):
            # former is vllm format, latter is sglang format
            error_code = response["error"]["code"] if "error" in response else response["code"]
            return _json_response(response, endpoint=endpoint, bridge_stats=bridge_stats, status_code=error_code)
        else:
            return _json_response(response, endpoint=endpoint, bridge_stats=bridge_stats)

    except json.JSONDecodeError as e:
        # To catch possible raw_request.json() errors
        error_response = ErrorResponse(
            error=ErrorInfo(
                message=f"Invalid JSON error: {str(e)}",
                type=HTTPStatus.BAD_REQUEST.phrase,
                code=HTTPStatus.BAD_REQUEST.value,
            ),
        )
        return _json_response(
            error_response.model_dump(),
            endpoint=endpoint,
            bridge_stats=bridge_stats,
            status_code=HTTPStatus.BAD_REQUEST.value,
        )
    except Exception as e:
        # Include full traceback for debugging
        tb = traceback.format_exc()
        logger.error(f"Error when handling {endpoint} request in SkyRL:\n{tb}")
        error_response = ErrorResponse(
            error=ErrorInfo(
                message=f"Error when handling {endpoint} request in SkyRL: {str(e)}\n\nTraceback:\n{tb}",
                type=HTTPStatus.INTERNAL_SERVER_ERROR.phrase,
                code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
            ),
        )
        return _json_response(
            error_response.model_dump(),
            endpoint=endpoint,
            bridge_stats=bridge_stats,
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
        )


def shutdown_server(host: str = "127.0.0.1", port: int = 8000, max_wait_seconds: int = 30) -> None:
    """Shutdown the server.

    Args:
        host: Server host.
        port: Server port.
        max_wait_seconds: How long to wait until the server stops listening.

    Raises:
        Exception: If the server is still responding after *max_wait_seconds*.
    """
    if _global_uvicorn_server is not None:
        _global_uvicorn_server.should_exit = True

    health_url = f"http://{host}:{port}/health"

    for i in range(max_wait_seconds):
        try:
            # If this succeeds, server is still alive
            requests.get(health_url, timeout=1)
        except requests.exceptions.RequestException:
            # A network error / connection refused means server is down.
            logger.info(f"Server shut down after {i + 1} seconds")
            return
        time.sleep(1)

    raise Exception(f"Server failed to shut down within {max_wait_seconds} seconds")


def wait_for_server_ready(host: str = "127.0.0.1", port: int = 8000, max_wait_seconds: int = 30) -> None:
    """
    Wait for the HTTP endpoint to be ready by polling the health endpoint.

    Args:
        host: Host where the server is running
        port: Port where the server is running
        max_wait_seconds: Maximum time to wait in seconds

    Raises:
        Exception: If server doesn't become ready within max_wait_seconds
    """
    max_retries = max_wait_seconds
    health_url = f"http://{host}:{port}/health"

    for i in range(max_retries):
        try:
            response = requests.get(health_url, timeout=1)
            if response.status_code == 200:
                logger.info(f"Server ready after {i + 1} attempts ({i + 1} seconds)")
                return
        except (requests.exceptions.RequestException, requests.exceptions.ConnectionError):
            if i == max_retries - 1:
                raise Exception(f"Server failed to start within {max_wait_seconds} seconds")
            time.sleep(1)  # Wait 1 second between retries


async def _monitor_event_loop_lag(
    bridge_stats: HTTPBridgeStatsAccumulator,
    interval_seconds: float,
) -> None:
    """Measure delay between a scheduled wake-up and the event loop running it."""
    try:
        loop = asyncio.get_running_loop()
        expected = loop.time() + interval_seconds
        while True:
            await asyncio.sleep(interval_seconds)
            now = loop.time()
            bridge_stats.observe("event_loop_lag_seconds", max(0.0, now - expected))
            expected = now + interval_seconds
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.exception("Inference HTTP event-loop lag monitor failed")
        raise


async def handle_models_request(
    *,
    bridge_stats: HTTPBridgeStatsAccumulator,
) -> JSONResponse:
    """Return the served model name and live engine context limit."""
    backend = _global_inference_engine_client
    assert backend is not None
    response = ModelList(
        data=[
            ModelCard(
                id=backend.model_name,
                created=_SERVER_CREATED_TIME,
                max_model_len=backend.max_model_len,
            )
        ]
    )
    return _json_response(
        response.model_dump(),
        endpoint=MODELS_ENDPOINT,
        bridge_stats=bridge_stats,
    )


def create_app(
    bridge_stats: HTTPBridgeStatsAccumulator | None = None,
    *,
    event_loop_lag_interval_seconds: float = 0.5,
) -> fastapi.FastAPI:
    """Create the FastAPI application."""
    bridge_stats = bridge_stats or HTTPBridgeStatsAccumulator()

    @asynccontextmanager
    async def lifespan(app: fastapi.FastAPI):
        logger.info("Starting inference HTTP endpoint...")
        monitor = asyncio.create_task(
            _monitor_event_loop_lag(bridge_stats, event_loop_lag_interval_seconds),
            name="inference-http-event-loop-lag",
        )
        try:
            yield
        finally:
            monitor.cancel()
            try:
                await monitor
            except asyncio.CancelledError:
                pass

    app = fastapi.FastAPI(
        title="InferenceEngine OpenAI-Compatible API",
        description="OpenAI-compatible chat completion API using InferenceEngineClient",
        version="1.0.0",
        lifespan=lifespan,
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(_RequestOutcomeMiddleware, bridge_stats=bridge_stats)

    @app.post(CHAT_COMPLETIONS_ENDPOINT)
    async def chat_completion(raw_request: Request):
        """
        Takes in OpenAI's `ChatCompletionRequest` and returns OpenAI's `ChatCompletionResponse`.

        Note that the specific fields inside the request and response depend on the backend you use.
        If `config.generator.backend` is `vllm`, then the request and response will be vLLM's.
        Same for SGLang. SkyRL does not perform field validation beyond `model` and `session_id`,
        and otherwise depends on the underlying engines' validation.

        Make sure you add in `session_id` (a string or an integer) to ensure load balancing and
        sticky routing. The same agentic rollout / session should share the same `session_id` so
        they get routed to the same engine for better prefix caching. If unprovided, we will route
        to a random engine which is not performant.

        API reference:
        - https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html
        - https://docs.sglang.ai/basic_usage/openai_api_completions.html
        """
        return await handle_openai_request(
            raw_request,
            endpoint="/chat/completions",
            bridge_stats=bridge_stats,
        )

    @app.post(COMPLETIONS_ENDPOINT)
    async def completions(raw_request: Request):
        """
        Takes in OpenAI's `CompletionRequest` and returns OpenAI's `CompletionResponse`.

        Note that the specific fields inside the request and response depend on the backend you use.
        If `config.generator.backend` is `vllm`, then the request and response will be vLLM's.
        SkyRL only validates the fields `model` and `session_id`, and otherwise offloads
        field validation to the underlying engines.

        Make sure you add in `session_id` to ensure load balancing and sticky routing. Since
        `request["prompt"]` can be `Union[list[int], list[list[int]], str, list[str]]`, i.e.
        {batched, single} x {string, token IDs}, we follow the following logic for request routing:
        - For batched request: `session_id`, if provided, must have the same length as `request["prompt"]`
          so that each `request["prompt"][i]` is routed based on `session_id[i]`.
        - For single request: `session_id`, if provided, must be a single integer or a singleton
          list, where each `session_id` is a string or an integer.

        API reference:
        - https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html
        - https://docs.sglang.ai/basic_usage/openai_api_completions.html
        """
        return await handle_openai_request(raw_request, endpoint="/completions", bridge_stats=bridge_stats)

    @app.post(TOKENIZE_ENDPOINT)
    async def tokenize(raw_request: Request):
        """Delegate chat tokenization to the inference backend's serving renderer."""
        return await handle_openai_request(raw_request, endpoint=TOKENIZE_ENDPOINT, bridge_stats=bridge_stats)

    @app.get(MODELS_ENDPOINT)
    async def models():
        """Return the served model identity and configured context limit."""
        return await handle_models_request(bridge_stats=bridge_stats)

    # Health check endpoint
    # All inference engine replicas are initialized before creating `InferenceEngineClient`, and thus
    # we can start receiving requests as soon as the FastAPI server starts
    @app.get("/health")
    async def health_check():
        return {"status": "healthy"}

    # This handler only catches unexpected server-side exceptions
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        logger.error(f"Unhandled exception: {str(exc)}\n{traceback.format_exc()}")
        error_response = ErrorResponse(
            error=ErrorInfo(
                message=f"Unhandled exception: {str(exc)}",
                type=HTTPStatus.INTERNAL_SERVER_ERROR.phrase,
                code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
            ),
        )
        return _json_response(
            error_response.model_dump(),
            endpoint=request.url.path,
            bridge_stats=bridge_stats,
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
        )

    return app


def serve(
    inference_engine_client: InferenceHTTPBackend,
    host: str = "0.0.0.0",
    port: int = 8000,
    log_level: str = "info",
    bridge_stats: HTTPBridgeStatsAccumulator | None = None,
):
    """
    Start the HTTP endpoint.

    Args:
        inference_engine_client: The InferenceEngineClient to use as backend
        host: Host to bind to (default: "0.0.0.0")
        port: Port to bind to (default: 8000)
        log_level: Logging level (default: "info")
        bridge_stats: Shared accumulator for HTTP bridge metrics
    """
    app = create_app(bridge_stats)

    # Configure logging
    logging.basicConfig(level=getattr(logging, log_level.upper()))

    logger.info(f"Starting server on {host}:{port}")

    # Run server
    config = uvicorn.Config(app, host=host, port=port, log_level=log_level, access_log=True)
    server = uvicorn.Server(config)

    # Expose server for external shutdown control (tests)
    set_global_state(inference_engine_client, server)

    try:
        # Run until shutdown
        asyncio.run(server.serve())
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down...")
    except Exception as e:
        logger.error(f"Server error: {e}")
        raise
