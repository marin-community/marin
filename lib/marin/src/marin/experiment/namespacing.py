# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Per-user namespacing for training artifacts.

A mutable (``dev``) checkpoint addresses identity on ``{name}/dev`` alone, so two people
iterating on the same experiment write to the same path and clobber each other — and a
resumption checkpointer would resume from whichever run last touched it.
:func:`user_owned_name` prefixes an artifact unconditionally when its output is
owned by the caller. :func:`user_namespaced_name` applies that prefix only to
mutable training steps. Fixed checkpoints and datasets otherwise keep their
shared names, so published runs stay citable and tokenized caches remain shared.
"""

from rigging.provenance import username_segment

from marin.execution.artifact import is_mutable_version


def user_owned_name(name: str) -> str:
    """Return ``users/{username}/{name}`` for a user-owned artifact."""
    return f"users/{username_segment()}/{name}"


def user_namespaced_name(name: str, version: str) -> str:
    """Return ``users/{username}/{name}`` for a mutable version, ``name`` unchanged otherwise.

    A fixed (calendar) ``version`` stays in the shared namespace; a mutable
    ``dev``/``<label>-dev`` version is isolated per user. Raises if no username resolves, so a
    dev run never silently lands in a shared ``users/unknown/`` bucket.
    """
    if not is_mutable_version(version):
        return name
    return user_owned_name(name)
