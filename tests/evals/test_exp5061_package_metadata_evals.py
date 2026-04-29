# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from experiments.evals.exp5061_package_metadata_evals import NPM_REGISTRY_SLICE_KEY, package_metadata_raw_validation_sets


def test_package_metadata_raw_validation_sets_render_expected_path_and_tags() -> None:
    datasets = package_metadata_raw_validation_sets(raw_root="gs://example-bucket/raw/long_tail")

    dataset = datasets[NPM_REGISTRY_SLICE_KEY]
    assert dataset.input_path == "gs://example-bucket/raw/long_tail/packages/npm/registry.jsonl.gz"
    assert dataset.tags == ("package_metadata", "epic:5005", "issue:5061", NPM_REGISTRY_SLICE_KEY)
    assert dataset.text_key == "text"
