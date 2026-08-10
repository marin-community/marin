# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Per-user namespacing for mutable training artifacts.

A mutable (``dev``) checkpoint addresses identity on ``{name}/dev`` alone, so two people
iterating on the same experiment write to the same path and clobber each other — and a
resumption checkpointer would resume from whichever run last touched it.
:func:`user_namespaced_name` prefixes a *mutable* training step's name with
``users/{username}/`` so each author gets an isolated scratch namespace. Fixed
(calendar-versioned) checkpoints and all datasets keep their shared names, so published runs
stay citable and the expensive multi-TB tokenized caches still cache-hit across users.
"""

from rigging.provenance import username_segment

from marin.execution.artifact import is_mutable_version


def user_namespaced_name(name: str, version: str) -> str:
    """Return ``users/{username}/{name}`` for a mutable version, ``name`` unchanged otherwise.

    A fixed (calendar) ``version`` stays in the shared namespace; a mutable
    ``dev``/``<label>-dev`` version is isolated per user. Raises if no username resolves, so a
    dev run never silently lands in a shared ``users/unknown/`` bucket.
    """
    if not is_mutable_version(version):
        return name
    return f"users/{username_segment()}/{name}"
