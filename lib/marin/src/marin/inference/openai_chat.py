# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""One-attempt HTTP transport for OpenAI-compatible chat requests."""

import json
import urllib.error
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class OpenAIChatClient:
    base_url: str
    token: str = field(repr=False)
    timeout: float
    priority: str = "bulk"

    def complete(self, body: Mapping[str, Any]) -> dict[str, Any]:
        """Send exactly one request; its caller owns durable reservation and retry policy."""
        request = urllib.request.Request(
            self.base_url.rstrip("/") + "/chat/completions",
            data=json.dumps(body, ensure_ascii=False).encode(),
            headers={
                "Authorization": f"Bearer {self.token}",
                "x-priority": self.priority,
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                result = json.load(response)
        except urllib.error.HTTPError as error:
            raise ConnectionError(f"Chat request returned HTTP {error.code}") from None
        except urllib.error.URLError as error:
            raise ConnectionError("Chat request failed before a response was available") from error
        if not isinstance(result, dict):
            raise ValueError("Chat response must be a JSON object")
        return result
