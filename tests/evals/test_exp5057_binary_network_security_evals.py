# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from experiments.evals.exp5057_binary_network_security_evals import (
    UWF_ZEEK_SLICE_KEY,
    binary_network_security_raw_validation_sets,
)


def test_binary_network_security_raw_validation_sets_uses_expected_slice_key_and_tags():
    datasets = binary_network_security_raw_validation_sets(raw_root="gs://example-bucket/raw")

    dataset = datasets[UWF_ZEEK_SLICE_KEY]

    assert dataset.input_path == "gs://example-bucket/raw/binary/uwf/zeek.jsonl.gz"
    assert dataset.tags == (
        "binary_network_security",
        "epic:5005",
        "issue:5057",
        UWF_ZEEK_SLICE_KEY,
    )
