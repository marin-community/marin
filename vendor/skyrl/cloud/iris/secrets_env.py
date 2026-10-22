"""Load a launch-host secrets file (``KEY=VALUE`` lines) into ``os.environ``."""

from __future__ import annotations

import os
from pathlib import Path


def default_secrets_env() -> str | None:
    """Return the approved launch-host secrets file when it exists."""
    candidate = os.environ.get("OT_AGENT_SECRETS_ENV") or "~/Documents/secrets.env"
    path = Path(candidate).expanduser()
    return str(path) if path.is_file() else None


def load_secrets_env_into_os_environ(secrets_env: str | None) -> int:
    """Read ``secrets_env`` (KEY=VALUE) into ``os.environ`` on the launch host.

    File values override any pre-existing shell values. Returns the number of
    entries loaded (0 when no file is given or it does not exist).
    """
    if not secrets_env:
        return 0
    path = Path(secrets_env).expanduser().resolve()
    if not path.is_file():
        return 0
    loaded = 0
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        if "=" not in line:
            continue
        k, _, v = line.partition("=")
        k = k.strip()
        v = v.strip()
        if len(v) >= 2 and v[0] == v[-1] and v[0] in ("'", '"'):
            v = v[1:-1]
        if not k:
            continue
        os.environ[k] = v  # file overrides shell
        loaded += 1
    return loaded
