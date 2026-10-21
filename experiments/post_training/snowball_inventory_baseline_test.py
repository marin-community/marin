# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace

import pytest

from experiments.post_training.snowball_inventory_baseline import BaselineIdentity


def baseline_identity():
    return BaselineIdentity(
        model_uri="s3://example/model/checkpoint25",
        model_manifest_identity="a" * 64,
        tokenizer_uri="example/tokenizer",
        tokenizer_revision="b" * 40,
        tokenizer_template_sha256="0" * 64,
        panel_sha256="c" * 64,
        family_counts={"calendar": 20, "structural": 20},
        evaluation_recipe={"generator": {"temperature": 0, "max_new_tokens": 8192}},
        recovery_policy={"per_answer_timeout": 600, "total_timeout": 900},
        runtime_source_digest="d" * 64,
        scoring_source_sha256="e" * 64,
    )


@pytest.mark.parametrize(
    "change",
    [
        {"model_uri": "s3://example/model/checkpoint50"},
        {"model_manifest_identity": "f" * 64},
        {"tokenizer_uri": "example/other-tokenizer"},
        {"tokenizer_revision": "f" * 40},
        {"tokenizer_template_sha256": "f" * 64},
        {"panel_sha256": "f" * 64},
        {"family_counts": {"calendar": 19, "structural": 20}},
        {"evaluation_recipe": {"generator": {"temperature": 0, "max_new_tokens": 4096}}},
        {"recovery_policy": {"per_answer_timeout": 60, "total_timeout": 900}},
        {"runtime_source_digest": "f" * 64},
        {"scoring_source_sha256": "f" * 64},
    ],
)
def test_baseline_cache_never_aliases_changed_evaluation_contract(change):
    identity = baseline_identity()
    assert replace(identity, **change).digest != identity.digest


def test_baseline_cache_reuses_same_contract_despite_mapping_order():
    identity = baseline_identity()
    assert replace(identity, family_counts={"structural": 20, "calendar": 20}).digest == identity.digest
