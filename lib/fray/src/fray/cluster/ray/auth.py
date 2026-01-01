# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Ray token-auth utilities shared across CLI and library code."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

DEFAULT_RAY_AUTH_TOKEN_SECRET = "RAY_AUTH_TOKEN"


def ray_auth_secret(secret_override: str | None = None) -> str:
    """Return the Secret Manager secret name to use for Ray auth token retrieval."""
    return secret_override or DEFAULT_RAY_AUTH_TOKEN_SECRET


def maybe_fetch_local_ray_token(*, config_path: str | None) -> str:
    """Ensure a Ray auth token is available locally and return a token file path.

    Priority:
    - Respect an existing RAY_AUTH_TOKEN_PATH if it points to a file.
    - If a token exists at the default path (~/.ray/auth_token), use it.
    - Otherwise, fetch it with `gcloud` into the default path (~/.ray/auth_token).
    """
    token_path_env = os.environ.get("RAY_AUTH_TOKEN_PATH")
    if token_path_env and Path(token_path_env).expanduser().exists():
        return str(Path(token_path_env).expanduser())

    default_path = Path.home() / ".ray" / "auth_token"
    if default_path.exists():
        return str(default_path)

    if not config_path:
        raise RuntimeError(
            "Ray token authentication is enabled but no local token was found. "
            "Create a token file at ~/.ray/auth_token or set RAY_AUTH_TOKEN_PATH."
        )

    secret = ray_auth_secret()
    default_path.parent.mkdir(parents=True, exist_ok=True)

    token = subprocess.check_output(
        ["gcloud", "secrets", "versions", "access", "latest", f"--secret={secret}"],
        text=True,
    ).strip()
    if not token:
        raise RuntimeError(f"Secret {secret} returned empty token")

    default_path.write_text(token)
    default_path.chmod(0o600)
    return str(default_path)
