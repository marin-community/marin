# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

import pytest
from marin.midtraining.data_manifest import (
    DataCacheComponent,
    DataCacheManifest,
    DataManifestPointer,
    approved_manifest_uri,
    dump_data_manifest,
)
from marin.midtraining.tokenizers import LLAMA3_TOKENIZER

from tests.midtraining._fixtures import make_data_manifest


def test_weights_must_sum_to_one():
    base = make_data_manifest()
    with pytest.raises(ValueError, match=r"must sum to ~1\.0"):
        DataCacheManifest(
            mix_name=base.mix_name,
            mix_spec_digest=base.mix_spec_digest,
            region=base.region,
            components=base.components,
            weights={"pretrain": 0.3, "nemotron_cc_math_v1/4plus": 0.5},
            seq_len=base.seq_len,
        )


def test_component_must_live_in_declared_region():
    mismatched_component = DataCacheComponent(
        logical_name="x",
        cache_path="gs://marin-us-central1/tokenized/x",
        cache_digest="sha256:test",
        tokenizer=LLAMA3_TOKENIZER,
        bos_sample=(128_000, 128_000),
    )
    with pytest.raises(ValueError, match="is not in declared region"):
        DataCacheManifest(
            mix_name="m",
            mix_spec_digest="sha256:m",
            region="us-east5",
            components=(mismatched_component,),
            weights={"x": 1.0},
            seq_len=4096,
        )


def test_mixed_tokenizers_rejected():
    from marin.midtraining.tokenizers import QWEN3_TOKENIZER

    a = DataCacheComponent(
        logical_name="a",
        cache_path="gs://marin-us-east5/a",
        cache_digest="sha256:a",
        tokenizer=LLAMA3_TOKENIZER,
        bos_sample=(128_000, 128_000),
    )
    b = DataCacheComponent(
        logical_name="b",
        cache_path="gs://marin-us-east5/b",
        cache_digest="sha256:b",
        tokenizer=QWEN3_TOKENIZER,
        bos_sample=(151_643, 151_643),
    )
    with pytest.raises(ValueError, match="Mixed tokenizers"):
        DataCacheManifest(
            mix_name="m",
            mix_spec_digest="sha256:m",
            region="us-east5",
            components=(a, b),
            weights={"a": 0.5, "b": 0.5},
            seq_len=4096,
        )


def test_fingerprint_changes_with_cache_digest():
    base = make_data_manifest()
    altered_components = (
        base.components[0],
        DataCacheComponent(
            logical_name=base.components[1].logical_name,
            cache_path=base.components[1].cache_path,
            cache_digest="sha256:CHANGED",
            tokenizer=base.components[1].tokenizer,
            total_sequences=base.components[1].total_sequences,
            total_tokens=base.components[1].total_tokens,
            bos_sample=base.components[1].bos_sample,
        ),
    )
    altered = DataCacheManifest(
        mix_name=base.mix_name,
        mix_spec_digest=base.mix_spec_digest,
        region=base.region,
        components=altered_components,
        weights=base.weights,
        seq_len=base.seq_len,
    )
    assert base.fingerprint() != altered.fingerprint()


def test_dump_data_manifest_roundtrips_json():
    base = make_data_manifest()
    text = dump_data_manifest(base)
    data = json.loads(text)
    assert data["mix_name"] == base.mix_name
    assert data["region"] == base.region
    assert len(data["components"]) == 2


def test_pointer_must_reference_approved_manifest_path():
    with pytest.raises(ValueError, match="must live under"):
        DataManifestPointer(
            mix_name="p33m67_test",
            approved_manifest_uri="gs://marin-us-east5/some/other/place.json",
            approved_at="2026-05-16T00:00:00Z",
        )


def test_approved_manifest_uri_uses_fingerprint():
    uri = approved_manifest_uri(mix_name="p33m67", region="us-east5", fingerprint="sha256:abc123")
    assert uri == "gs://marin-us-east5/midtrain-manifests/data/p33m67/abc123.json"
