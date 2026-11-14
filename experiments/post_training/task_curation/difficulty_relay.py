# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Apply the protocol to native Harbor requests without changing source packages."""

import hashlib
import json
import logging
from collections import Counter
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock, Thread

import httpx
from finestore.rl.difficulty import PROTOCOL_ID, PROTOCOLS
from harbor.environments.model_relay import _upstream_error
from rigging.chat_templates import chat_template_messages
from transformers import AutoTokenizer

LOGGER = logging.getLogger(__name__)
REQUEST_TIMEOUT = 600


@dataclass(frozen=True)
class RequestBudget:
    context_tokens: int
    input_tokens: int
    output_tokens: int


CANDIDATE_BUDGET = RequestBudget(*(int(value) for value in PROTOCOLS[PROTOCOL_ID].sampling[:3]))


class ReceiptHTTPServer(ThreadingHTTPServer):
    daemon_threads = False


@contextmanager
def budget_relay(
    *,
    endpoint: str,
    model: str,
    api_key: str,
    tokenizer_model: str,
    tokenizer_revision: str,
    parameters: dict,
    budget: RequestBudget | None,
    receipts: Path,
):
    """Serve task-labelled, non-streaming chat and pinned CPU tokenization locally.

    The upstream credential remains in the trusted host. Every actual request and
    response is retained before returning it to Harbor. Server usage is authoritative.
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model, revision=tokenizer_revision)
    numbers = Counter()
    lock = Lock()
    with httpx.Client(headers={"Authorization": f"Bearer {api_key}"}, timeout=REQUEST_TIMEOUT) as upstream:

        class Handler(BaseHTTPRequestHandler):
            def setup(self):
                super().setup()
                self.connection.settimeout(REQUEST_TIMEOUT)

            def log_message(self, message_format, *args):
                pass

            def respond(self, status, payload):
                data = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                metadata = {} if budget is None else {"max_model_len": budget.context_tokens}
                self.respond(200, {"data": [{"id": model, **metadata}]})

            def do_POST(self):
                length = int(self.headers.get("Content-Length", "0"))
                if length > 16 * 1024 * 1024:
                    self.respond(413, {"error": "request body exceeds limit"})
                    return
                body = json.loads(self.rfile.read(length))
                template = {
                    **body.get("chat_template_kwargs", {}),
                    **parameters.get("extra_body", {}).get("chat_template_kwargs", {}),
                }
                tokens = []
                if (budget is not None or self.path.endswith("/tokenize")) and "messages" in body:
                    tokens = tokenizer.apply_chat_template(
                        chat_template_messages(body["messages"]),
                        tools=body.get("tools") or None,
                        add_generation_prompt=body.get("add_generation_prompt", True),
                        tokenize=True,
                        return_dict=False,
                        **template,
                    )
                elif budget is not None or self.path.endswith("/tokenize"):
                    tokens = tokenizer.encode(body["prompt"], add_special_tokens=body.get("add_special_tokens", True))
                if self.path.endswith("/tokenize"):
                    metadata = {} if budget is None else {"max_model_len": budget.context_tokens}
                    self.respond(200, {"tokens": tokens, "count": len(tokens), **metadata})
                    return
                if not self.path.endswith("/v1/chat/completions") or body.get("stream"):
                    self.respond(400, {"error": "non-streaming chat completions required"})
                    return
                requested_body = deepcopy(body)
                body.update({key: value for key, value in parameters.items() if key != "extra_body"})
                body.update(parameters.get("extra_body", {}))
                body["model"] = model
                if budget is not None:
                    body["max_tokens"] = min(
                        body.get("max_tokens", budget.output_tokens),
                        budget.output_tokens,
                        budget.context_tokens - len(tokens),
                    )
                task = hashlib.sha256(self.path.removesuffix("/v1/chat/completions").encode()).hexdigest()
                with lock:
                    number = numbers[task]
                    numbers[task] += 1
                root = receipts / task / f"{number:06d}"
                root.mkdir(parents=True, exist_ok=False)
                (root / "request.json").write_text(
                    json.dumps(
                        {
                            "route": self.path,
                            "prompt_tokens": None if budget is None else len(tokens),
                            "requested_body": requested_body,
                            "body": body,
                        }
                    )
                    + "\n"
                )
                if budget is not None and len(tokens) > budget.input_tokens:
                    payload = {
                        "error": {
                            "code": "input_budget_unserved",
                            "type": "ContextBudgetExceededError",
                            "message": "Protocol input budget rejects this unserved prompt",
                        }
                    }
                    status = 400
                else:
                    try:
                        response = upstream.post(endpoint.rstrip("/") + "/chat/completions", json=body)
                        (root / "upstream-response.body").write_bytes(response.content)
                        (root / "upstream-response-status.json").write_text(json.dumps({"status": response.status_code}))
                        status, payload = response.status_code, response.json()
                        if not response.is_success:
                            safe = _upstream_error(status, response.content)
                            status, payload = safe.status, json.loads(safe.body)
                    except httpx.HTTPError as error:
                        # Preserve transport classification; source commands are never replayed here.
                        status, payload = 502, {
                            "error": {"type": type(error).__name__, "message": "Upstream model transport failed"}
                        }
                (root / "response.json").write_text(json.dumps({"status": status, "body": payload}) + "\n")
                usage = payload.get("usage")
                if (
                    status == 200
                    and budget is not None
                    and (
                        usage is None
                        or usage["prompt_tokens"] > budget.input_tokens
                        or usage["completion_tokens"] > budget.output_tokens
                        or usage["prompt_tokens"] + usage["completion_tokens"] > budget.context_tokens
                    )
                ):
                    status, payload = 502, {
                        "error": {
                            "type": "ProtocolUsageViolation",
                            "message": "Server usage exceeds the protocol budget or is missing",
                        }
                    }
                self.respond(status, payload)

        server = ReceiptHTTPServer(("127.0.0.1", 0), Handler)
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield f"http://127.0.0.1:{server.server_port}"
        finally:
            server.shutdown()
            thread.join()
            server.server_close()
