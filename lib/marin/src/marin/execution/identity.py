# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Resolution of the Marin user identity for scoping output paths."""

import getpass
import os
import re

USER_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")

# Generic / service identities that must not be used to scope a user's outputs.
GENERIC_USER_NAMES = frozenset({"root", "runner", "nobody", "user", ""})


def resolve_marin_user(override: str | None = None) -> str | None:
    """Resolve the Marin user used to scope output paths.

    Resolution order: ``override`` (if non-empty after strip) > ``MARIN_USER`` env
    (if set and non-empty) > iris ``resolve_job_user()`` (import-guarded) >
    ``getpass.getuser()``.

    The resolved name is classified into three outcomes:

    - MALFORMED (contains ``/`` or ``..``, is non-printable, or otherwise fails
      ``^[A-Za-z0-9][A-Za-z0-9_-]*$``): raise ``ValueError``. A traversal-y or
      otherwise illegal name is a security/misconfig signal, never silently
      accepted.
    - EMPTY or a generic/service identity (``root``, ``runner``, ``nobody``,
      ``user``, empty): return ``None`` — there is no usable per-user owner.
    - Otherwise: return the sanitized name.

    Args:
        override: Explicit user name; takes precedence over every other source.

    Returns:
        The sanitized user name, or ``None`` when no usable per-user identity is
        available.

    Raises:
        ValueError: If the resolved name is malformed.
    """
    resolved = _resolve_raw_user(override)

    if resolved in GENERIC_USER_NAMES:
        return None

    if not USER_NAME_PATTERN.match(resolved):
        raise ValueError(
            f"Invalid Marin user name {resolved!r}: must match {USER_NAME_PATTERN.pattern} "
            "(no '/', '..', empty, or non-printable characters)."
        )

    return resolved


def _resolve_raw_user(override: str | None) -> str:
    """Resolve the raw, unsanitized user name following the precedence order."""
    if override is not None and override.strip():
        return override.strip()

    env_user = os.environ.get("MARIN_USER")
    if env_user is not None and env_user.strip():
        return env_user.strip()

    # iris is an optional dependency; guard the import locally (the one allowed
    # local import per AGENTS.md). resolve_job_user() may itself fall back to
    # "root", which the sanitizer/generic checks then handle.
    try:
        from iris.cluster.client.job_info import resolve_job_user  # noqa: PLC0415  # optional dep: iris
    except ImportError:
        pass
    else:
        return resolve_job_user()

    return getpass.getuser()
