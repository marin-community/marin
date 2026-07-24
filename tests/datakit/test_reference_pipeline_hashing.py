# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Cache-identity (``hash_attrs``) regression tests for the reference Datakit DAG.

These are pure StepSpec-construction tests -- no cluster, no data. They lock in the
cache-identity contract: every content-determining parameter enters the step hash, no
region-specific ``gs://`` path does, and external inputs are pinned by a caller
version tag rather than their absolute path.
"""

import dataclasses
import json

import pytest
from marin.execution.step_spec import StepSpec

from experiments.datakit import reference_pipeline
from experiments.datakit.reference_pipeline import SMOKE_SCALE, PoolConfig, reference_datakit_steps


@pytest.fixture(autouse=True)
def _marin_prefix(monkeypatch):
    # ``StepSpec.output_path`` resolves ``marin_prefix()``; pin it so the test never
    # depends on ambient GCS metadata. (``hash_id`` itself excludes the prefix.)
    monkeypatch.setenv("MARIN_PREFIX", "gs://marin-test-region")


def _sources() -> dict[str, StepSpec]:
    return {name: StepSpec(name=f"datakit/normalize/{name}", fn=lambda op: None) for name in ("a", "b")}


def _build(*, scale=SMOKE_SCALE, **kw):
    return reference_datakit_steps(
        _sources(),
        quality_model="gs://some-region/quality/pooled_junkgate2",
        quality_model_version="pooled-junkgate2",
        scale=scale,
        **kw,
    )


def _steps_by_name(result) -> dict[str, StepSpec]:
    return {s.name: s for s in result.all_steps}


def test_no_region_path_in_hash_attrs_except_known_bloom_gap():
    # A region-specific gs:// path in a hash means byte-identical data gets a
    # different output path per region. The only remaining leak is the decontam
    # bloom's EVAL_ROOT (tracked follow-up); everything else must be clean.
    for step in _build().all_steps:
        if step.name == "datakit/bloom/_combined_fixed":
            continue
        assert "gs://" not in json.dumps(step.hash_attrs, default=str), f"{step.name} leaks a gs:// path into its hash"


def test_store_hash_tracks_content_not_resources():
    base = _build().output_buckets.hash_id
    # cluster_view is read by the store fn and NOT captured by any dep -> must re-key.
    cv = dataclasses.replace(SMOKE_SCALE.cluster, cluster_view=16)
    changed = _build(scale=dataclasses.replace(SMOKE_SCALE, cluster=cv)).output_buckets.hash_id
    # The worker fleet is execution policy -> must NOT re-key.
    pool = dataclasses.replace(SMOKE_SCALE, pool=PoolConfig(n_workers=999))
    resourced = _build(scale=pool).output_buckets.hash_id
    assert changed != base
    assert resourced == base


def test_minhash_params_rekey_minhash_and_dedup():
    base = _steps_by_name(_build())
    mh = dataclasses.replace(SMOKE_SCALE.minhash, num_bands=13)
    changed = _steps_by_name(_build(scale=dataclasses.replace(SMOKE_SCALE, minhash=mh)))
    assert changed["datakit/minhash/a"].hash_id != base["datakit/minhash/a"].hash_id
    # dedup has no params of its own; it must re-key via its minhash deps.
    assert changed["datakit/dedup"].hash_id != base["datakit/dedup"].hash_id


def test_centroid_seed_rekeys_training():
    base = _steps_by_name(_build())["datakit/cluster/train_centroids"].hash_id
    seeded = dataclasses.replace(SMOKE_SCALE.cluster, train_seed=7)
    changed = _steps_by_name(_build(scale=dataclasses.replace(SMOKE_SCALE, cluster=seeded)))
    assert changed["datakit/cluster/train_centroids"].hash_id != base


@pytest.mark.parametrize(
    ("constant", "step"),
    [("LUXICAL_REVISION", "datakit/embed/a"), ("TOKENIZER_REVISION", "datakit/tokenize/a")],
)
def test_upstream_revision_bump_rekeys_its_step(monkeypatch, constant, step):
    # The pins exist so a retagged upstream artifact invalidates the cache rather than
    # silently serving bytes built from the old revision.
    base = _steps_by_name(_build())[step].hash_id
    monkeypatch.setattr(reference_pipeline, constant, "deadbeef")
    assert _steps_by_name(_build())[step].hash_id != base


def test_external_path_requires_version_tag():
    with pytest.raises(ValueError, match="quality_model_version is required"):
        reference_datakit_steps(_sources(), quality_model="gs://r/model", quality_model_version=None)
    with pytest.raises(ValueError, match="centroids_version is required"):
        reference_datakit_steps(
            _sources(),
            quality_model="gs://r/model",
            quality_model_version="v",
            domain_centroids="gs://r/centroids",
            centroids_version=None,
        )


def test_quality_model_version_not_path_drives_identity():
    # Same model bytes staged at two region paths, same version tag -> one output path.
    def quality_hash(model_dir: str) -> str:
        result = reference_datakit_steps(
            _sources(), quality_model=model_dir, quality_model_version="pooled-junkgate2", scale=SMOKE_SCALE
        )
        return _steps_by_name(result)["datakit/quality/a"].hash_id

    assert quality_hash("gs://region-a/quality/m") == quality_hash("gs://region-b/quality/m")


def test_centroids_version_not_path_drives_identity():
    def assign_hash(centroids_dir: str) -> str:
        result = _build(domain_centroids=centroids_dir, centroids_version="run-42")
        return _steps_by_name(result)["datakit/cluster_assign/a"].hash_id

    assert assign_hash("gs://region-a/centroids") == assign_hash("gs://region-b/centroids")
