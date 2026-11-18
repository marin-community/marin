# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The Exp3 V7 mix graph must resolve to the data config the V7 release declares.

These tests resolve the graph offline — seal records seeded exactly as the seal steps write
them — and pin the resolved ``LmDataConfig`` component by component: names, cache addresses,
format, packing, the three-phase weight schedule, and the shuffle/geometry settings. The
online test pins the committed seal manifest against the live sealed catalog.

The bounded-lookahead packing the Exp3 runtime used (cap 256, lookahead 128) is
recorded in every seal record as provenance; the resolved components do not carry it.
"""

import hashlib
import json
import os

import pytest
from levanter.data.text.datasets import DatasetComponent, UrlDatasetSourceConfig
from levanter.data.text.formats import PrebuiltLmDatasetFormat, TextLmDatasetFormat
from marin.execution.artifact import ArtifactRecord, write_record
from marin.execution.lazy import ArtifactStep, StepContext, materialized_config
from marin.processing.tokenize.tokenize import TokenizedCache
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath, prefix_join

from experiments.sft_new_mix import exp3_v7_mix as mix

_TOKENIZED_CACHE = f"{TokenizedCache.__module__}.{TokenizedCache.__qualname__}"
_HAS_S3_CREDENTIALS = bool(
    (os.environ.get("CW_KEY_ID") and os.environ.get("CW_KEY_SECRET"))
    or (os.environ.get("AWS_ACCESS_KEY_ID") and os.environ.get("AWS_SECRET_ACCESS_KEY"))
)
COMP_PROG = (
    "sft/source/nemotron_sft_v3/competitive_programming_v2/competitive_coding_cpp",
    "sft/source/nemotron_sft_v3/competitive_programming_v2/competitive_coding_python",
)


def _consumer() -> ArtifactStep[TokenizedCache]:
    """The stand-in consumer a training run would lower: the mix wired through ``mixture``."""
    return ArtifactStep(
        name="sft-mixes/exp3-v7/data-config",
        version=mix.EXP3_V7_VERSION,
        artifact_type=TokenizedCache,
        run=lambda _config: None,
        build_config=mix.exp3_v7_data_config,
        deps=mix.exp3_v7_mix_deps(),
    )


def _resolved_data_config(prefix: str):
    """The resolved ``LmDataConfig`` a consumer of the mix would assemble at run time."""
    for handle in mix._exp3_v7_sources().values():
        # The exact record a seal step leaves: config equals its literal build config.
        write_record(
            ArtifactRecord(
                name=handle.name,
                version=handle.version,
                output_path=handle.path(prefix),
                result_type=_TOKENIZED_CACHE,
                config=materialized_config(handle, prefix),
            )
        )
    return materialized_config(_consumer(), prefix)


def _component_dict(component: DatasetComponent) -> dict:
    """A component as the release serialized it (draccus field names, ``type`` tags aside).

    The release components carry ``source: null``; marin's mixture emits an empty-URL
    ``UrlDatasetSourceConfig`` for a built cache. They are the same thing to Levanter
    (both mean "load from ``cache_dir``"), and the empty-URL form keeps Levanter's
    skip-a-missing-split behavior, so the comparison normalizes it to ``None``.
    """
    source = component.source
    if isinstance(source, UrlDatasetSourceConfig):
        assert source.train_urls == [] and source.validation_urls == []
        assert source.cache_dir == component.cache_dir and source.format == component.format
        source_out = None
    else:
        assert source is None
        source_out = None
    fmt = component.format
    if isinstance(fmt, PrebuiltLmDatasetFormat):
        fmt_dict = {"input_ids_key": fmt.input_ids_key, "loss_weights_key": fmt.loss_weights_key, "type": "prebuilt"}
    elif isinstance(fmt, TextLmDatasetFormat):
        fmt_dict = {"text_key": fmt.text_key, "type": "text"}
    else:
        raise AssertionError(f"unexpected component format {fmt!r}")
    return {
        "source": source_out,
        "cache_dir": component.cache_dir,
        "format": fmt_dict,
        "pack": component.pack,
        "tags": component.tags,
        "split": component.split,
        "flat_cache": component.flat_cache,
    }


def _expected_component(release_component: dict) -> dict:
    """The release component with fields ``main`` does not express dropped and defaults filled.

    The release serializer omits null fields (``source``/``tags``), which the comparison
    normalizes the resolved components to (see :func:`_component_dict`).
    """
    expected = {key: value for key, value in release_component.items() if key not in ("type", "packing_lookahead")}
    expected.setdefault("source", None)
    expected.setdefault("tags", None)
    return expected


def test_seal_manifest_totals_match_the_release():
    """The derived manifest agrees with the members it cites and its own sums."""
    derived = mix._MANIFEST["derived_from"]
    assert derived["cache_catalog"]["sha256"] == mix._RELEASE["artifacts"]["cache_catalog.json"]["sha256"]
    assert derived["source_loss"]["sha256"] == mix._RELEASE["artifacts"]["source_loss.json"]["sha256"]
    seals = mix._SEALS
    totals = mix._MANIFEST["totals"]
    assert totals["sources"] == len(seals) == 434
    assert totals["rows"] == sum(seal["rows"] for seal in seals.values())
    assert totals["input_ids_tokens"] == sum(seal["input_ids_tokens"] for seal in seals.values())
    assert totals["input_ids_tokens"] == totals["loss_weights_tokens"]
    assert totals["shards"] == sum(seal["shards"] for seal in seals.values())
    assert set(seals) == set(mix._SOURCE_NAMES) == set(mix._SCHEDULE[0][1])
    # The V7 release is V6 with the option-G comp-prog rebuild; record the expected population.
    assert seals[COMP_PROG[0]]["rows"] + seals[COMP_PROG[1]]["rows"] == 474896
    assert seals[COMP_PROG[0]]["input_ids_tokens"] + seals[COMP_PROG[1]]["input_ids_tokens"] == 11789616476


def test_resolved_components_match_the_release_data_config(tmp_path):
    resolved = _resolved_data_config(str(tmp_path))
    expected_components = mix._DATA["components"]
    assert set(resolved.components) == set(expected_components)
    assert len(resolved.components) == 434
    for name, component in resolved.components.items():
        assert _component_dict(component) == _expected_component(expected_components[name]), name


def test_resolved_weights_and_schedule_match_the_release(tmp_path):
    resolved = _resolved_data_config(str(tmp_path))
    assert resolved.train_weights == [(phase, weights) for phase, weights in mix._SCHEDULE]
    assert [phase for phase, _ in resolved.train_weights] == [0, 250, 500]
    for _phase, weights in resolved.train_weights:
        assert sum(weights.values()) == pytest.approx(1.0, abs=1e-9)
    assert set(resolved.train_weights[0][1]) == set(resolved.components)
    # The mixture record pins the shared comp-prog population at the post-option-G weights.
    for entry in mix._SCHEDULE:
        for name in COMP_PROG:
            assert name in entry[1]


def test_resolved_mixture_settings_match_the_release(tmp_path):
    resolved = _resolved_data_config(str(tmp_path))
    data = mix._DATA
    assert resolved.tokenizer == data["tokenizer"]
    assert resolved.cache_catalog == data["cache_catalog"]
    assert resolved.cache_dir is None
    assert resolved.vocab_size is None
    assert resolved.chat_template is None
    assert resolved.enforce_eos == data["enforce_eos"]
    assert resolved.auto_build_caches == data["auto_build_caches"]
    assert resolved.shuffle == data["shuffle"]
    assert resolved.permutation_type == data["permutation_type"]
    assert resolved.block_cross_document_attention == data["block_cross_document_attention"]
    assert resolved.stop_strategy == data["stop_strategy"]
    assert resolved.target_budget is None and resolved.experiment_budget is None
    assert resolved.mixture_block_size == data["mixture_block_size"]
    assert resolved.max_train_batches is None
    assert resolved.num_validation_sequences == data["num_validation_sequences"]
    assert resolved.shuffle_before_trainval_split == data["shuffle_before_trainval_split"]
    assert resolved.prior_context_phases == []
    assert resolved.mixture_start_step is None
    assert resolved.cache_options.batch_size == 128


def test_seal_check_rejects_a_drifted_catalog(tmp_path, monkeypatch):
    """The seal check validates digests, counts, loss modes, and totals."""
    swept: list[tuple[str, int]] = []
    monkeypatch.setattr(mix, "_verify_shard_paths", lambda cache_dir, shards: swept.append((cache_dir, len(shards))))
    check = mix.exp3_v7_seal_check()
    config = materialized_config(check, str(tmp_path))
    entries = {}
    for name, seal in mix._SEALS.items():
        entries[name] = {
            "cache_dir": mix._CACHE_ROOT,
            "ledger": {
                "is_finished": True,
                "total_num_rows": seal["rows"],
                "finished_shards": [f"stage/components/{name}/shards/{i}/cache" for i in range(seal["shards"])],
                "field_counts": {"input_ids": seal["input_ids_tokens"]},
                "metadata": {"preprocessor_metadata": {"loss_mode": seal["loss_mode"]}},
            },
        }
    raw = json.dumps({"splits": {"train": entries}}).encode()
    path = tmp_path / "catalog.json"
    path.write_bytes(raw)
    digest = mix._entry_digest

    expected = {
        name: {
            "rows": seal["rows"],
            "shards": seal["shards"],
            "loss_mode": seal["loss_mode"],
            "entry_sha256": digest(entries[name]),
        }
        for name, seal in mix._SEALS.items()
    }
    good = dict(
        config, cache_catalog={"uri": str(path), "sha256": hashlib.sha256(raw).hexdigest()}, expected_sources=expected
    )
    check.run(good)
    assert swept == [(mix._CACHE_ROOT, mix._MANIFEST["totals"]["shards"])]

    bad_sha = dict(good, cache_catalog={"uri": str(path), "sha256": "0" * 64})
    with pytest.raises(ValueError, match="digest differs"):
        check.run(bad_sha)

    name = "pretrain/source/c00q0-train/gt_64k"
    wrong_rows = dict(
        good, expected_sources=dict(expected, **{name: dict(expected[name], rows=expected[name]["rows"] + 1)})
    )
    with pytest.raises(ValueError, match="rows"):
        check.run(wrong_rows)

    wrong_digest = dict(good, expected_sources=dict(expected, **{name: dict(expected[name], entry_sha256="1" * 64)}))
    with pytest.raises(ValueError, match="entry digest differs"):
        check.run(wrong_digest)

    wrong_loss = dict(
        good, expected_sources=dict(expected, **{name: dict(expected[name], loss_mode="chat_assistant_only")})
    )
    with pytest.raises(ValueError, match="loss mode"):
        check.run(wrong_loss)


@pytest.mark.skipif(not _HAS_S3_CREDENTIALS, reason="CoreWeave S3 credentials unavailable")
def test_sealed_catalog_matches_the_manifest_online():
    """The committed manifest still matches the live sealed catalog, and shards exist."""
    configure_coreweave_s3()
    raw = StoragePath(mix._CATALOG_URI).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == mix._CATALOG_SHA256
    catalog = json.loads(raw)["splits"]["train"]
    assert set(catalog) == set(mix._SEALS)
    for name, seal in mix._SEALS.items():
        entry = catalog[name]
        ledger = entry["ledger"]
        assert ledger["total_num_rows"] == seal["rows"]
        assert len(ledger["finished_shards"]) == seal["shards"]
        assert mix._entry_digest(entry) == seal["entry_sha256"]
    # A spot check that the content-addressed shard caches are still in place.
    shards = catalog["pretrain/source/c00q0-train/gt_64k"]["ledger"]["finished_shards"]
    shards += catalog[COMP_PROG[0]]["ledger"]["finished_shards"][:2]
    for shard in shards:
        assert StoragePath(prefix_join(mix._CACHE_ROOT, shard)).exists()


def test_data_config_is_fingerprint_safe():
    """Assembling the mix at fingerprint time reads no records and pulls no real paths."""
    cfg = mix.exp3_v7_data_config(StepContext.for_fingerprint(deps=mix.exp3_v7_mix_deps()))
    assert len(cfg.components) == 434
    assert all("@" in component.cache_dir for component in cfg.components.values())
    assert cfg.cache_catalog == "sft-mixes/exp3-v7/cache-catalog@2026.11.18"
    assert cfg.train_weights == [(phase, weights) for phase, weights in mix._SCHEDULE]


def test_consumer_lowering_carries_every_source():
    spec = _consumer().lower()
    dep_names = [dep.name for dep in spec.deps]
    assert len(dep_names) == 434 + 3
    assert "sft-mixes/exp3-v7/cache-catalog" in dep_names
    assert mix._SOURCE_NAMES[0] in dep_names
