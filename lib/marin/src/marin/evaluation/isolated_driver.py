# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared process boundary for pinned external evaluation drivers."""

import subprocess
from collections.abc import Mapping, Sequence

from marin.evaluation.eval_env import env_vars_from_keys

ISOLATED_REQUEST_MODE = 0o600
_SYSTEM_ENV_KEYS = (
    "CURL_CA_BUNDLE",
    "HOME",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "NO_PROXY",
    "PATH",
    "PYTHONHASHSEED",
    "REQUESTS_CA_BUNDLE",
    "SSL_CERT_DIR",
    "SSL_CERT_FILE",
    "TMPDIR",
    "UV_CACHE_DIR",
    "XDG_CACHE_HOME",
    "http_proxy",
    "https_proxy",
    "no_proxy",
)


def isolated_driver_environment(
    additional_keys: Sequence[str] = (),
    overrides: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Return the allowlisted environment for a pinned evaluator process."""
    environment = env_vars_from_keys((*_SYSTEM_ENV_KEYS, *additional_keys))
    environment.update(overrides or {})
    return environment


def driver_failure(exc: subprocess.CalledProcessError) -> ValueError:
    """Convert a failed evaluator process into a useful validation error."""
    stderr = exc.stderr.strip() if isinstance(exc.stderr, str) else ""
    stdout = exc.stdout.strip() if isinstance(exc.stdout, str) else ""
    detail = stderr or stdout or f"driver exited with status {exc.returncode}"
    return ValueError(detail)


def capture_driver(command: Sequence[str], environment: Mapping[str, str]) -> subprocess.CompletedProcess[str]:
    """Run a pinned evaluator command and capture its text output."""
    try:
        return subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            env=environment,
        )
    except subprocess.CalledProcessError as exc:
        raise driver_failure(exc) from exc
