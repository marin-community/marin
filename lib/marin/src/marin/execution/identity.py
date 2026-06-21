# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Resolution of the Marin user identity for scoping output paths."""

import getpass
import os
import re

USER_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")

# Generic / service identities that must not be used to scope a user's outputs.
GENERIC_USER_NAMES = frozenset({"root", "runner", "nobody", "user", ""})


def resolve_marin_user(override: str | None = None, *, for_user_scope: bool) -> str:
    """Resolve the Marin user used to scope output paths.

    Resolution order: ``override`` (if non-empty after strip) > ``MARIN_USER`` env
    (if set and non-empty) > iris ``resolve_job_user()`` (import-guarded) >
    ``getpass.getuser()``.

    The resolved name is sanitized against ``^[A-Za-z0-9][A-Za-z0-9_-]*$``; any
    value containing ``/``, ``..``, that is empty, or that is non-printable raises
    ``ValueError``.

    Args:
        override: Explicit user name; takes precedence over every other source.
        for_user_scope: When True, a generic/service identity (``root``,
            ``runner``, ``nobody``, ``user``, empty) is rejected with a
            ``ValueError`` instructing the caller to set ``MARIN_USER`` or use a
            shared output scope. When False, generic names pass the regex check
            and are returned unchanged.

    Returns:
        The sanitized user name.

    Raises:
        ValueError: If the resolved name fails the pattern, or if it is a generic
            identity while ``for_user_scope`` is True.
    """
    resolved = _resolve_raw_user(override)

    if not USER_NAME_PATTERN.match(resolved):
        raise ValueError(
            f"Invalid Marin user name {resolved!r}: must match {USER_NAME_PATTERN.pattern} "
            "(no '/', '..', empty, or non-printable characters)."
        )

    if for_user_scope and resolved in GENERIC_USER_NAMES:
        raise ValueError(
            f"Refusing to scope user outputs under generic identity {resolved!r}. "
            "Set MARIN_USER to a real user name or pass output_scope=SHARED."
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
