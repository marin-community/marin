# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Native mini-swe-agent environment client for a private QEMU bridge."""

import json
import socket
from typing import Any

from minisweagent.environments.local import LocalEnvironment
from minisweagent.utils.serialize import recursive_merge


class MiniQemuEnvironment(LocalEnvironment):
    """Keep native completion handling while commands execute in the guest."""

    def __init__(self, *, socket_path: str, **kwargs):
        super().__init__(**kwargs)
        self.socket_path = socket_path

    def _request(self, request: dict) -> dict:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
            connection.connect(self.socket_path)
            with connection.makefile("rwb") as stream:
                stream.write(json.dumps(request).encode() + b"\n")
                stream.flush()
                response = json.loads(stream.readline())
        if "error" in response:
            raise RuntimeError(response["error"])
        return response

    def execute(self, action: dict, cwd: str = "", *, timeout: int | None = None) -> dict[str, Any]:
        output = self._request(
            {
                "operation": "execute",
                "command": action.get("command", ""),
                "cwd": cwd or self.config.cwd,
                "env": self.config.env,
                "timeout": timeout or self.config.timeout,
            }
        )
        self._check_finished(output)
        return output

    def get_template_vars(self, **kwargs) -> dict[str, Any]:
        guest = self._request({"operation": "template", "env": self.config.env})
        return recursive_merge(self.config.model_dump(), guest, kwargs)
