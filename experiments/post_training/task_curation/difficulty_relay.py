# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Apply the protocol to native Harbor requests without changing source packages."""

import hashlib
import json
import logging
from collections import Counter
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock, Thread

import httpx
from finestore.rl.difficulty import PROTOCOL_ID, PROTOCOLS
from harbor.environments.model_relay import _upstream_error
from rigging.chat_templates import chat_template_messages
from transformers import AutoTokenizer

CONTEXT_TOKENS, INPUT_TOKENS, OUTPUT_TOKENS = (int(value) for value in PROTOCOLS[PROTOCOL_ID].sampling[:3])

LOGGER = logging.getLogger(__name__)
REQUEST_TIMEOUT = 600


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
                self.respond(200, {"data": [{"id": model, "max_model_len": CONTEXT_TOKENS}]})

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
                if "messages" in body:
                    tokens = tokenizer.apply_chat_template(
                        chat_template_messages(body["messages"]),
                        tools=body.get("tools") or None,
                        add_generation_prompt=body.get("add_generation_prompt", True),
                        tokenize=True,
                        return_dict=False,
                        **template,
                    )
                else:
                    tokens = tokenizer.encode(body["prompt"], add_special_tokens=body.get("add_special_tokens", True))
                if self.path.endswith("/tokenize"):
                    self.respond(200, {"tokens": tokens, "count": len(tokens), "max_model_len": CONTEXT_TOKENS})
                    return
                if not self.path.endswith("/v1/chat/completions") or body.get("stream"):
                    self.respond(400, {"error": "non-streaming chat completions required"})
                    return
                body.update({key: value for key, value in parameters.items() if key != "extra_body"})
                body.update(parameters.get("extra_body", {}))
                body["model"] = model
                body["max_tokens"] = min(
                    body.get("max_tokens", OUTPUT_TOKENS), OUTPUT_TOKENS, CONTEXT_TOKENS - len(tokens)
                )
                task = hashlib.sha256(self.path.removesuffix("/v1/chat/completions").encode()).hexdigest()
                with lock:
                    number = numbers[task]
                    numbers[task] += 1
                root = receipts / task / f"{number:06d}"
                root.mkdir(parents=True, exist_ok=False)
                (root / "request.json").write_text(
                    json.dumps({"route": self.path, "prompt_tokens": len(tokens), "body": body}) + "\n"
                )
                if len(tokens) > INPUT_TOKENS:
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
                if status == 200 and (
                    usage is None
                    or usage["prompt_tokens"] > INPUT_TOKENS
                    or usage["completion_tokens"] > OUTPUT_TOKENS
                    or usage["prompt_tokens"] + usage["completion_tokens"] > CONTEXT_TOKENS
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
