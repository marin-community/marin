# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""vLLM middleware for recording benchmark requests and literal responses."""

import json
import os
import uuid
from pathlib import Path

from starlette.types import ASGIApp, Message, Receive, Scope, Send


class RecordRequests:
    """Save exchanges in Iris outputs while preserving response bytes and streaming."""

    def __init__(self, app: ASGIApp, output_dir: Path | None = None):
        self.app = app
        self.output_dir = output_dir if output_dir is not None else Path(os.environ["IRIS_OUTPUT_DIR"]) / "requests"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if (
            scope["type"] != "http"
            or scope["method"] != "POST"
            or scope["path"]
            not in {
                "/v1/chat/completions",
                "/v1/completions",
            }
        ):
            await self.app(scope, receive, send)
            return
        body = bytearray()
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            body.extend(message.get("body", b""))
            if not message.get("more_body", False):
                break
        payload = json.loads(body)
        payload["return_token_ids"] = True
        effective_body = json.dumps(payload).encode()
        request_scope = dict(scope)
        request_scope["headers"] = [
            (key, value) for key, value in scope["headers"] if key.lower() != b"content-length"
        ] + [(b"content-length", str(len(effective_body)).encode())]
        destination = self.output_dir / uuid.uuid4().hex
        destination.mkdir()
        (destination / "request.json").write_bytes(body)
        (destination / "effective-request.json").write_bytes(effective_body)
        delivered = False

        async def receive_request() -> Message:
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": effective_body, "more_body": False}
            return await receive()

        with (destination / "response.bin").open("wb") as output:

            async def send_response(message: Message) -> None:
                if message["type"] == "http.response.start":
                    (destination / "status.json").write_text(json.dumps({"status": message["status"]}) + "\n")
                elif message["type"] == "http.response.body":
                    output.write(message.get("body", b""))
                await send(message)

            await self.app(request_scope, receive_request, send_response)
        (destination / "complete.json").write_text(json.dumps({"path": scope["path"]}) + "\n")
