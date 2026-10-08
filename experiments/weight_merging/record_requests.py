# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""vLLM middleware for recording benchmark requests and literal responses."""

import asyncio
import json
import os
import tarfile
import uuid
from pathlib import Path

from rigging.filesystem.buckets import filesystem_for
from starlette.types import ASGIApp, Message, Receive, Scope, Send

DURABLE_CAPTURE_ROOT = "s3://marin-us-east-02a/marin/experiments/weight-merging-20261007/routing-captures/durable-v1"


def persist_exchange(destination: Path, uri: str) -> None:
    """Publish a completed exchange as one compressed object."""
    fs, path = filesystem_for(uri)
    with fs.open(path, "wb") as output, tarfile.open(fileobj=output, mode="w|gz") as archive:
        for file in sorted(destination.iterdir()):
            archive.add(file, arcname=file.name)


class RecordRequests:
    """Save exchanges in Iris outputs while preserving response bytes and streaming."""

    def __init__(self, app: ASGIApp, output_dir: Path | None = None, remote_prefix: str | None = None):
        self.app = app
        self.remote_prefix = remote_prefix
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
        headers = dict(scope["headers"])
        (destination / "metadata.json").write_text(
            json.dumps(
                {
                    "model": payload["model"],
                    "stream": payload.get("stream", False),
                    "trial_id": headers.get(b"x-ot-trial-id", b"").decode(),
                    "path": scope["path"],
                }
            )
            + "\n"
        )
        delivered = False

        async def receive_request() -> Message:
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": effective_body, "more_body": False}
            return await receive()

        finalized = False
        stream_tail = b""
        with (destination / "response.bin").open("wb") as output:

            async def send_response(message: Message) -> None:
                nonlocal finalized, stream_tail
                if message["type"] == "http.response.start":
                    (destination / "status.json").write_text(json.dumps({"status": message["status"]}) + "\n")
                elif message["type"] == "http.response.body":
                    chunk = message.get("body", b"")
                    output.write(chunk)
                    stream_tail = (stream_tail + chunk)[-64:]
                    stream_done = payload.get("stream", False) and b"data: [DONE]" in stream_tail
                    if not finalized and (stream_done or not message.get("more_body", False)):
                        output.flush()
                        (destination / "complete.json").write_text(json.dumps({"path": scope["path"]}) + "\n")
                        if self.remote_prefix is not None:
                            uri = f"{self.remote_prefix.rstrip('/')}/{destination.name}.tar.gz"
                            await asyncio.to_thread(persist_exchange, destination, uri)
                        finalized = True
                await send(message)

            await self.app(request_scope, receive_request, send_response)


class DurableRecordRequests(RecordRequests):
    """Persist campaign captures before clients can finish their requests."""

    def __init__(self, app: ASGIApp):
        super().__init__(app, remote_prefix=f"{DURABLE_CAPTURE_ROOT}/{uuid.uuid4().hex}")
        (self.output_dir.parent / "capture-location.json").write_text(
            json.dumps({"remote_prefix": self.remote_prefix}) + "\n"
        )
