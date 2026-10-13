# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import cast

class ReviewHTTPServer(ThreadingHTTPServer):
    def __init__(self):
        self.requests: list[dict] = []
        self.review_available = True
        self.assessment_available = True
        super().__init__(("127.0.0.1", 0), ReviewHandler)


class ReviewHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        cast(ReviewHTTPServer, self.server).requests.append(body)
        task = json.loads(body["messages"][-1]["content"])
        tool = body["tool_choice"]["function"]["name"]
        server = cast(ReviewHTTPServer, self.server)
        if (tool == "review_task" and not server.review_available) or (tool == "assess_task" and not server.assessment_available):
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
        payload = json.dumps(result).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, message, *args):
        pass


