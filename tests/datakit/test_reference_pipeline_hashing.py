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
from fray.types import ResourceConfig
from marin.execution.artifact import ArtifactRecord, write_artifact, write_record
from marin.execution.step_spec import StepSpec
from marin.processing.classification.deduplication.fuzzy_dups import (
    FuzzyDupsAttrData,
    FuzzyDupsPerSource,
    compute_fuzzy_dups_attrs_step,
)
from marin.processing.classification.deduplication.fuzzy_minhash import (
    MinHashAttrData,
    MinHashParams,
    compute_minhash_attrs_step,
)
from marin.processing.classification.deduplication.verify_fuzzy_dups import (
    FuzzyVerificationImplementation,
    FuzzyVerificationStoreConfig,
)

from experiments.datakit import reference_pipeline
from experiments.datakit.fuzzy_validation import (
    FOCUS_SOURCE_NAME,
    BenchmarkSource,
    _legacy_candidate_input_steps,
    _select_benchmark_sources,
    build_fuzzy_validation_step,
    build_repacked_fuzzy_validation_step,
)
from experiments.datakit.reference_pipeline import (
    SMOKE_SCALE,
    PoolConfig,
    reference_datakit_steps,
    zephyr_datakit_steps,
)
from experiments.datakit.zephyr_benchmark import _route_outputs


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


def _depends_on(step: StepSpec, dependency: StepSpec) -> bool:
    return any(parent is dependency or _depends_on(parent, dependency) for parent in step.deps)


def _ancestor_names(step: StepSpec) -> set[str]:
    return {parent.name for parent in step.deps} | {
        ancestor_name for parent in step.deps for ancestor_name in _ancestor_names(parent)
    }


def test_global_exact_dedup_filters_only_the_store():
    result = _build()
    steps = _steps_by_name(result)
    exact_dedup = steps["datakit/global_exact_dedup"]

    for stage in ("tokenize", "embed", "quality", "decontam", "minhash"):
        assert not _depends_on(steps[f"datakit/{stage}/a"], exact_dedup)
    assert _depends_on(steps["datakit/store"], exact_dedup)


def test_benchmark_routes_every_stage_under_one_prefix():
    routed = _route_outputs(reference_pipeline.zephyr_datakit_steps(_sources()), "gs://temp/benchmark")
    steps = [routed.exact_dedup, *routed.tokenize.values(), *routed.minhash.values(), routed.fuzzy_dedup]

    assert all(step.output_path.startswith("gs://temp/benchmark/") for step in steps)
    assert routed.fuzzy_dedup.deps == list(routed.minhash.values())


def test_benchmark_document_target_selects_a_stable_source_prefix():
    sizes = {
        "too-large": 8_100_000,
        "large": 7_807_075,
        "medium": 192_436,
        "small": 488,
        "tail": 22,
        "empty": 0,
    }
    sources = [
        BenchmarkSource(
            source_key=source_key,
            normalized_artifact_path=f"gs://normalized/{source_key}",
            minhash_artifact_path=f"gs://minhash/{source_key}",
            documents=documents,
        )
        for source_key, documents in reversed(sizes.items())
    ]

    selected = _select_benchmark_sources(sources, 8_000_000)

    assert [source.source_key for source in selected] == ["large", "medium", "small", "tail"]
    assert sum(source.documents for source in selected) == 8_000_021


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
    layout = dataclasses.replace(SMOKE_SCALE.store, task_count=2)
    relaid = _build(scale=dataclasses.replace(SMOKE_SCALE, store=layout)).output_buckets.hash_id
    # The worker fleet is execution policy -> must NOT re-key.
    pool = dataclasses.replace(SMOKE_SCALE, pool=PoolConfig(n_workers=999))
    resourced = _build(scale=pool).output_buckets.hash_id
    execution = dataclasses.replace(SMOKE_SCALE.store, max_parallel_bucket_writes=1)
    rescheduled = _build(scale=dataclasses.replace(SMOKE_SCALE, store=execution)).output_buckets.hash_id
    spill_execution = dataclasses.replace(SMOKE_SCALE.store, partition_processes=2)
    respilled = _build(scale=dataclasses.replace(SMOKE_SCALE, store=spill_execution)).output_buckets.hash_id
    store_worker = dataclasses.replace(SMOKE_SCALE.store, worker=PoolConfig().worker)
    resized = _build(scale=dataclasses.replace(SMOKE_SCALE, store=store_worker)).output_buckets.hash_id
    assert changed != base
    assert relaid != base
    assert resourced == base
    assert rescheduled == base
    assert respilled == base
    assert resized == base


def test_minhash_params_rekey_minhash_and_dedup():
    base = _steps_by_name(_build())
    mh = dataclasses.replace(SMOKE_SCALE.minhash, num_bands=13)
    changed = _steps_by_name(_build(scale=dataclasses.replace(SMOKE_SCALE, minhash=mh)))
    assert changed["datakit/minhash/a"].hash_id != base["datakit/minhash/a"].hash_id
    # dedup has no params of its own; it must re-key via its minhash deps.
    assert changed["datakit/dedup"].hash_id != base["datakit/dedup"].hash_id


def test_decon_drop_set_tracks_normalized_source_identity():
    base = _steps_by_name(_build())["datakit/decon_drop/_combined"].hash_id
    sources = _sources()
    sources["a"] = dataclasses.replace(sources["a"], hash_attrs={"revision": 1})
    changed = reference_datakit_steps(
        sources,
        quality_model="gs://some-region/quality/pooled_junkgate2",
        quality_model_version="pooled-junkgate2",
        scale=SMOKE_SCALE,
    )
    assert _steps_by_name(changed)["datakit/decon_drop/_combined"].hash_id != base


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


def test_dedup_step_builders_match_the_datakit_graph_identity():
    """A step built by the helpers must resolve to the artifacts the DAG produced.

    The two constructions hashed different key names, so a helper-built step
    pointed at a fresh output tree and would recompute every MinHash source.
    """
    sources = _sources()
    graph = zephyr_datakit_steps(sources, SMOKE_SCALE)
    minhash = {
        name: compute_minhash_attrs_step(
            name=f"datakit/minhash/{name}",
            normalize=step,
            num_perms=SMOKE_SCALE.minhash.num_perms,
            num_bands=SMOKE_SCALE.minhash.num_bands,
            ngram_size=SMOKE_SCALE.minhash.ngram_size,
            text_cap_chars=SMOKE_SCALE.minhash.text_cap_chars,
            seed=SMOKE_SCALE.minhash.seed,
        )
        for name, step in sources.items()
    }
    dedup = compute_fuzzy_dups_attrs_step(
        name="datakit/dedup",
        minhash_steps=list(minhash.values()),
        max_parallelism=SMOKE_SCALE.dedup_max_parallelism,
    )

    assert {name: step.hash_id for name, step in minhash.items()} == {
        name: step.hash_id for name, step in graph.minhash.items()
    }
    assert dedup.hash_id == graph.fuzzy_dedup.hash_id


def test_fuzzy_validation_entry_point_matches_reference_graph():
    sources = _sources()
    target = build_fuzzy_validation_step(
        sources,
        implementation=FuzzyVerificationImplementation.EXACT,
        scale=SMOKE_SCALE,
    )
    reference_target = _steps_by_name(
        reference_datakit_steps(
            sources,
            quality_model="gs://some-region/quality/pooled_junkgate2",
            quality_model_version="pooled-junkgate2",
            scale=SMOKE_SCALE,
        )
    )["datakit/verify_fuzzy_dups"]

    assert target.output_path == reference_target.output_path
    assert target.dep_paths == reference_target.dep_paths


def test_repacked_fuzzy_validation_uses_current_normalized_and_minhash_steps():
    sources = _sources()
    minhash_steps = zephyr_datakit_steps(sources, SMOKE_SCALE).minhash
    target = build_repacked_fuzzy_validation_step(
        sources,
        minhash_steps,
        candidate_artifact_path="s3://candidate-bucket/legacy-dedup",
        legacy_source_key="normalized/legacy-a/outputs/main",
        source_name="a",
        repack_output_path_prefix="s3://temp-bucket/ttl=7d/fuzzy-validation/run-1",
        validation_output_path_prefix="s3://production-bucket/marin",
        validation_step_name="datakit/verify_fuzzy_dups",
        validation_scale=SMOKE_SCALE,
        store_config=FuzzyVerificationStoreConfig(
            recovery_timeout=30, ready_timeout=30, lookup_batch_size=8, shards_per_worker=1
        ),
        implementation=FuzzyVerificationImplementation.EXACT,
        coordinator_resources=ResourceConfig(cpu=1, ram="1g"),
        task_resources=ResourceConfig(cpu=1, ram="1g"),
    )

    ancestor_names = _ancestor_names(target)
    repack_step = next(step for step in target.deps if step.name == "fuzzy-validation/repack/a")
    assert target.output_path.startswith("s3://production-bucket/marin/datakit/verify_fuzzy_dups_")
    assert repack_step.output_path.startswith("s3://temp-bucket/ttl=7d/fuzzy-validation/run-1/")
    assert "datakit/minhash/a" in ancestor_names
    assert "datakit/minhash/b" in ancestor_names
    assert "datakit/normalize/a" in ancestor_names
    assert "fuzzy-validation/repack/a" in ancestor_names
    assert "fuzzy-validation/legacy-candidates" in ancestor_names
    assert "datakit/dedup" not in ancestor_names
    assert not any(name.startswith("datakit/tokenize/") for name in ancestor_names)


def test_legacy_candidate_inputs_replace_only_the_focus_source(tmp_path, monkeypatch):
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path))
    legacy_focus_key = "normalized/legacy-focus/outputs/main"
    other_key = "normalized/other/outputs/main"
    params = MinHashParams(num_perms=8, num_bands=4, ngram_size=5, seed=0)
    minhash_paths = [str(tmp_path / "minhash-focus"), str(tmp_path / "minhash-other")]
    write_artifact(
        MinHashAttrData(params=params, source_key=legacy_focus_key, attr_dir="attrs/focus", counters={}),
        minhash_paths[0],
    )
    write_artifact(
        MinHashAttrData(params=params, source_key=other_key, attr_dir="attrs/other", counters={}),
        minhash_paths[1],
    )
    candidate_path = str(tmp_path / "candidates")
    candidates = FuzzyDupsAttrData(
        params=params,
        sources={
            legacy_focus_key: FuzzyDupsPerSource(attr_dir="attrs/focus"),
            other_key: FuzzyDupsPerSource(attr_dir="attrs/other"),
        },
        counters={},
    )
    write_record(
        ArtifactRecord(output_path=candidate_path, dep_paths=minhash_paths, result=candidates.model_dump(mode="json"))
    )
    focus_normalized = StepSpec(name="normalized/current-focus", override_output_path="normalized/current-focus")
    focus_minhash = StepSpec(name="minhash/current-focus", override_output_path="minhash/current-focus")

    normalized_steps, minhash_steps = _legacy_candidate_input_steps(
        candidate_artifact_path=candidate_path,
        legacy_focus_source_key=legacy_focus_key,
        focus_normalized_step=focus_normalized,
        focus_minhash_step=focus_minhash,
    )

    assert normalized_steps[FOCUS_SOURCE_NAME] is focus_normalized
    assert minhash_steps[FOCUS_SOURCE_NAME] is focus_minhash
    pinned_names = set(normalized_steps) - {FOCUS_SOURCE_NAME}
    assert len(pinned_names) == 1
    pinned_name = pinned_names.pop()
    assert normalized_steps[pinned_name].output_path == str(tmp_path / "normalized/other")
    assert minhash_steps[pinned_name].output_path == minhash_paths[1]
