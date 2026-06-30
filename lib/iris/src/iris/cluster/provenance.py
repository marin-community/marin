# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bridge between the build-time ``rigging.provenance.Provenance`` value object,
the image environment, and the ``job_pb2.Provenance`` wire/storage form.

``IRIS_GIT_HASH`` carries the dedup key (the tree hash) on its own; the full
context is carried in ``IRIS_PROVENANCE`` as JSON. Older images without
``IRIS_PROVENANCE`` degrade to just the tree hash.
"""

import os
from collections.abc import Mapping

from rigging.provenance import Provenance

from iris.rpc import job_pb2

PROVENANCE_ENV = "IRIS_PROVENANCE"
GIT_HASH_ENV = "IRIS_GIT_HASH"


def provenance_to_env(provenance: Provenance) -> dict[str, str]:
    """Environment to bake into an image so the runtime can report provenance."""
    return {GIT_HASH_ENV: provenance.tree_hash, PROVENANCE_ENV: provenance.to_json()}


def provenance_from_env(environ: Mapping[str, str] = os.environ) -> Provenance:
    # "{}" is the Dockerfile's default ARG: an image built without the CLI's
    # --build-arg carries it, so treat it (like an absent var) as no provenance
    # and fall back to the bare tree hash.
    raw = environ.get(PROVENANCE_ENV)
    if raw and raw != "{}":
        return Provenance.from_json(raw)
    git_hash = environ.get(GIT_HASH_ENV, "unknown")
    return Provenance(tree_hash=git_hash, base_commit=git_hash, dirty=False, branch=None, built_by=None)


def provenance_to_proto(provenance: Provenance) -> job_pb2.Provenance:
    return job_pb2.Provenance(
        tree_hash=provenance.tree_hash,
        base_commit=provenance.base_commit,
        dirty=provenance.dirty,
        branch=provenance.branch or "",
        built_by=provenance.built_by or "",
    )


def provenance_from_proto(msg: job_pb2.Provenance) -> Provenance:
    return Provenance(
        tree_hash=msg.tree_hash,
        base_commit=msg.base_commit or msg.tree_hash,
        dirty=msg.dirty,
        branch=msg.branch or None,
        built_by=msg.built_by or None,
    )
