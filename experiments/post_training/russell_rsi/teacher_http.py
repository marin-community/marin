# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Store exact native teacher HTTP requests and responses before parsing."""

import base64
import hashlib
import json
from collections.abc import Callable

import httpx
from rigging.filesystem.storage_path import StoragePath

from experiments.post_training.russell_rsi.bootstrap_loop import write_once


async def journal_teacher_response(
    client: httpx.AsyncClient,
    resolve_base_url: Callable[[], str],
    directory: StoragePath,
    body: dict,
    identity: dict,
) -> bytes:
    """Issue one reserved request, or parse its saved response without another HTTP request."""
    issued = directory / "issued.json"
    response_path = directory / "response.json"
    request_path = directory / "request.json"
    wire_path = directory / "request-wire.bin"
    wire_record_path = directory / "request-wire.json"
    if issued.exists():
        if json.loads(issued.read_text()) != identity:
            raise ValueError("Saved teacher issuance has a different request identity")
        if json.loads(request_path.read_text()) != body:
            raise ValueError("Saved teacher request differs from its reserved request")
        if not response_path.exists():
            raise RuntimeError("Teacher request is ambiguous; no replacement request is permitted")
        wire = wire_path.read_bytes()
        wire_record = json.loads(wire_record_path.read_text())
        if hashlib.sha256(wire).hexdigest() != wire_record["sha256"] or json.loads(wire) != body:
            raise ValueError("Saved teacher wire bytes differ from their request identity")
        record = json.loads(response_path.read_text())
    else:
        if response_path.exists():
            raise ValueError("Saved teacher response lacks its request reservation")
        write_once(request_path, body)
        url = resolve_base_url().rstrip("/") + "/chat/completions"
        outbound = client.build_request("POST", url, json=body)
        wire = outbound.content
        if wire_path.exists():
            if wire_path.read_bytes() != wire:
                raise ValueError("Saved teacher wire bytes differ from the pending request")
        else:
            wire_path.write_bytes(wire)
        write_once(
            wire_record_path,
            {"sha256": hashlib.sha256(wire).hexdigest(), "content_type": outbound.headers["Content-Type"]},
        )
        write_once(issued, identity)
        response = await client.send(outbound)
        record = {
            "identity": identity,
            "url": url,
            "status_code": response.status_code,
            "response_headers": dict(response.headers),
            "body_base64": base64.b64encode(response.content).decode("ascii"),
        }
        write_once(response_path, record)
    if record["identity"] != identity:
        raise ValueError("Saved teacher response has a different request identity")
    raw = base64.b64decode(record["body_base64"], validate=True)
    httpx.Response(record["status_code"], content=raw, request=httpx.Request("POST", record["url"])).raise_for_status()
    return raw
