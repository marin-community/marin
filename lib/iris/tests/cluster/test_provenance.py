# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from iris.cluster.provenance import (
    provenance_from_env,
    provenance_from_proto,
    provenance_to_env,
    provenance_to_proto,
)
from rigging.provenance import Provenance

_PROV = Provenance(tree_hash="abcd", base_commit="9d2edea", dirty=True, branch="feat", built_by="power")


def test_from_env_parses_baked_provenance():
    assert provenance_from_env(provenance_to_env(_PROV)) == _PROV


def test_from_env_falls_back_to_tree_hash_when_unset():
    # An image built without the CLI's --build-arg has no IRIS_PROVENANCE.
    got = provenance_from_env({"IRIS_GIT_HASH": "5e0490f6f4"})
    assert got == Provenance(tree_hash="5e0490f6f4", base_commit="5e0490f6f4", dirty=False, branch=None, built_by=None)


def test_from_env_treats_dockerfile_default_brace_as_unset():
    # "{}" is the Dockerfile's default ARG value; it must not crash.
    got = provenance_from_env({"IRIS_PROVENANCE": "{}", "IRIS_GIT_HASH": "5e0490f6f4"})
    assert got.tree_hash == "5e0490f6f4"


def test_proto_round_trip():
    assert provenance_from_proto(provenance_to_proto(_PROV)) == _PROV
