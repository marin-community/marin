# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import cast


@dataclass(frozen=True)
class JudgeReply:
    content: str
    finish_reason: str = "stop"


class ReviewHTTPServer(ThreadingHTTPServer):
    def __init__(self):
        self.requests: list[dict] = []
        self.review_available = True
        self.assessment_available = True
        self.judge_replies: list[JudgeReply] = []
        self.model_ids: list[str] = []
        self.bare_model_list = False
        super().__init__(("127.0.0.1", 0), ReviewHandler)


@contextmanager
def running_review_server():
    server = ReviewHTTPServer()
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


class ReviewHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        server = cast(ReviewHTTPServer, self.server)
        models = [{"id": model} for model in server.model_ids]
        self.respond(models if server.bare_model_list else {"data": models})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        cast(ReviewHTTPServer, self.server).requests.append(body)
        if "tool_choice" not in body:
            reply = cast(ReviewHTTPServer, self.server).judge_replies.pop(0)
            self.respond(
                {
                    "id": "chatcmpl-fixture",
                    "object": "chat.completion",
                    "created": 0,
                    "model": body["model"],
                    "usage": {"prompt_tokens": 12, "completion_tokens": 4, "total_tokens": 16},
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": reply.finish_reason,
                            "message": {"role": "assistant", "content": reply.content},
                        }
                    ],
                }
            )
            return
        task = json.loads(body["messages"][-1]["content"])
        tool = body["tool_choice"]["function"]["name"]
        server = cast(ReviewHTTPServer, self.server)
        if (tool == "review_task" and not server.review_available) or (
            tool == "assess_task" and not server.assessment_available
        ):
            self.send_error(503)
            return
        verdict = (
            {
                "task_id": task["id"],
                "quality": "good",
                "confidence": "high",
                "reference_status": "consistent",
                "defects": [],
                "evidence": "Reference agrees with the task.",
            }
            if tool == "review_task"
            else {
                "task_id": task["task_id"],
                "verdict": "good" if "task" in task["evidence"] else "unknown",
                "explanation": "The reference agrees with the question.",
                "evidence_ids": ["task" if "task" in task["evidence"] else "normalize"],
            }
        )
        result = {
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "review",
                                "type": "function",
                                "function": {
                                    "name": tool,
                                    "arguments": json.dumps(verdict),
                                },
                            }
                        ],
                    },
                }
            ],
        }
        self.respond(result)

    def respond(self, result: dict | list) -> None:
        payload = json.dumps(result).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, message, *args):
        pass
