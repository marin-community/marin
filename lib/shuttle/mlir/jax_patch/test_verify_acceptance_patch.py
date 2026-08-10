# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavior tests for fail-closed Bazel cquery label evidence."""

import pytest
from verify_acceptance_patch import ADAPTER, BRIDGE, parse_cquery_labels, require_cquery_labels


def test_cquery_labels_normalize_configuration_suffixes_and_retain_extras():
    output = "\n".join(
        (
            "//jaxlib:_jax_pywrap_library (b86b285)",
            "//jaxlib:jax.cc (null)",
            BRIDGE,
            f"{ADAPTER} (0123456789abcdef)",
            "@llvm-project//mlir:LinalgNamedStructuredOpsYamlIncGen__o_impl_$@_genrule (b86b285)",
        )
    )

    labels = require_cquery_labels(output, (BRIDGE, ADAPTER))

    assert labels == {
        "//jaxlib:_jax_pywrap_library",
        "//jaxlib:jax.cc",
        BRIDGE,
        ADAPTER,
        "@llvm-project//mlir:LinalgNamedStructuredOpsYamlIncGen__o_impl_$@_genrule",
    }


@pytest.mark.parametrize(
    "output",
    (
        "",
        f"{ADAPTER} (not-a-hash)",
        f"{ADAPTER} (b86b285) trailing",
        f" {ADAPTER} (b86b285)",
        f"{ADAPTER} ",
        "@shuttle_mlir//ShuttleXlaRegistryAdapter (b86b285)",
        f"{ADAPTER}\n\n{BRIDGE}",
    ),
)
def test_cquery_labels_reject_malformed_output(output):
    with pytest.raises(ValueError, match="cquery"):
        parse_cquery_labels(output)


def test_cquery_labels_reject_duplicate_required_normalized_labels():
    output = f"{ADAPTER}\n{ADAPTER} (b86b285)"

    with pytest.raises(ValueError, match="repeated required labels"):
        require_cquery_labels(output, (ADAPTER,))


def test_cquery_labels_allow_extra_labels_in_multiple_configurations():
    output = f"{ADAPTER} (b86b285)\n@llvm-project//mlir:IR (b86b285)\n@llvm-project//mlir:IR (71ac42f)"

    labels = require_cquery_labels(output, (ADAPTER,))

    assert labels == {ADAPTER, "@llvm-project//mlir:IR"}


@pytest.mark.parametrize(
    "output",
    (
        f"{BRIDGE} (b86b285)\n@shuttle_mlir//:ShuttleXlaRegistryAdapterLookalike (b86b285)",
        f"{ADAPTER} (b86b285)\n@other_repository//:ShuttlePythonObserverTestBridge (b86b285)",
    ),
)
def test_cquery_labels_reject_absent_exact_dependency_among_lookalikes(output):
    with pytest.raises(ValueError, match="omitted required labels"):
        require_cquery_labels(output, (BRIDGE, ADAPTER))
