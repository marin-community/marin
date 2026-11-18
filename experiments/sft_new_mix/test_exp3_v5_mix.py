# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The Exp3 V5 mix graph must resolve to the data config the Exp3 SFT run consumed.

The run's consumed data config equals the release's ``data.json`` byte for byte on every
release field (verified against ``production51-sft-consumed-config.json`` before this test
was written; the launcher added only default-valued fields). These tests resolve the graph
offline — seal records seeded exactly as the seal steps write them — and pin the resolved
``LmDataConfig`` component by component: names, cache addresses, format, packing, rows and
token counts, the three-phase weight schedule, and the shuffle/geometry settings.

The one component field the release carries that current ``main`` cannot express is the
fork-runtime bounded-lookahead packing (``packing_lookahead: 128``): it is recorded in every
seal record as provenance and asserted here, not silently dropped.
"""

import hashlib
import json
import os

import pytest
from levanter.data.text.datasets import DatasetComponent
from levanter.data.text.formats import PrebuiltLmDatasetFormat, TextLmDatasetFormat
from marin.execution.artifact import ArtifactRecord, read_record, write_record
from marin.execution.lazy import ArtifactStep, StepContext, materialized_config
from marin.processing.tokenize.tokenize import TokenizedCache

from experiments.sft_new_mix import exp3_v5_mix as mix

_TOKENIZED_CACHE = f"{TokenizedCache.__module__}.{TokenizedCache.__qualname__}"
_HAS_S3_CREDENTIALS = bool(
    (os.environ.get("CW_KEY_ID") and os.environ.get("CW_KEY_SECRET"))
    or (os.environ.get("AWS_ACCESS_KEY_ID") and os.environ.get("AWS_SECRET_ACCESS_KEY"))
)


def _consumer() -> ArtifactStep[TokenizedCache]:
    """The stand-in consumer a training run would lower: the mix wired through ``mixture``."""
    return ArtifactStep(
        name="sft-mixes/exp3-v5/data-config",
        version=mix.EXP3_V5_VERSION,
        artifact_type=TokenizedCache,
        run=lambda _config: None,
        build_config=mix.exp3_v5_data_config,
        deps=mix.exp3_v5_mix_deps(),
    )


def _resolved_data_config(prefix: str):
    """The resolved ``LmDataConfig`` a consumer of the mix would assemble at run time."""
    for handle in mix.exp3_v5_sources().values():
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
    """A component as the release serialized it (draccus field names, ``type`` tags aside)."""
    fmt = component.format
    if isinstance(fmt, PrebuiltLmDatasetFormat):
        fmt_dict = {"input_ids_key": fmt.input_ids_key, "loss_weights_key": fmt.loss_weights_key, "type": "prebuilt"}
    elif isinstance(fmt, TextLmDatasetFormat):
        fmt_dict = {"text_key": fmt.text_key, "type": "text"}
    else:
        raise AssertionError(f"unexpected component format {fmt!r}")
    return {
        "source": component.source,
        "cache_dir": component.cache_dir,
        "format": fmt_dict,
        "pack": component.pack,
        "tags": component.tags,
        "split": component.split,
        "flat_cache": component.flat_cache,
    }


def _expected_component(release_component: dict) -> dict:
    """The release component with fields ``main`` does not express dropped and defaults filled.

    ``packing_lookahead`` is the fork-runtime bounded-lookahead packing; it is asserted
    separately from the seal records. The release serializer omits null fields
    (``source``/``tags``), which the resolved components carry as ``None``.
    """
    expected = {key: value for key, value in release_component.items() if key not in ("type", "packing_lookahead")}
    expected.setdefault("source", None)
    expected.setdefault("tags", None)
    return expected


def test_release_fixtures_match_the_pinned_release():
    release_raw = (mix._FIXTURES / "exp3_v5_release.json").read_bytes()
    assert hashlib.sha256(release_raw).hexdigest() == mix._RELEASE_SHA256
    data_raw = (mix._FIXTURES / "exp3_v5_data.json").read_bytes()
    assert hashlib.sha256(data_raw).hexdigest() == mix._RELEASE["artifacts"]["data.json"]["sha256"]
    assert mix._CATALOG_SHA256 == mix._RELEASE["artifacts"]["cache_catalog.json"]["sha256"]
    # The seal manifest claims derivation from these exact release members.
    derived = mix._MANIFEST["derived_from"]
    assert derived["cache_catalog"]["sha256"] == mix._RELEASE["artifacts"]["cache_catalog.json"]["sha256"]
    assert derived["source_loss"]["sha256"] == mix._RELEASE["artifacts"]["source_loss.json"]["sha256"]


def test_seal_manifest_totals_match_the_release_population():
    population = mix._RELEASE["natural_population"]
    totals = mix._MANIFEST["totals"]
    assert totals["sources"] == 434
    docs = population["chat_assistant_only"]["output_documents"] + population["full_token"]["output_documents"]
    tokens = population["chat_assistant_only"]["output_tokens"] + population["full_token"]["output_tokens"]
    assert totals["rows"] == docs
    assert totals["input_ids_tokens"] == tokens
    assert totals["loss_weights_tokens"] == totals["input_ids_tokens"]
    assert totals["shards"] == sum(len(seal["shards"]) for seal in mix._SEALS.values())
    assert set(mix._SEALS) == set(mix._SOURCE_NAMES)


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
    assert set(resolved.train_weights[0][1]) == set(resolved.components)


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


def test_seal_records_pin_the_exact_sealed_caches(tmp_path):
    prefix = str(tmp_path)
    for name in ("pretrain/source/c00q0-train/gt_64k", "sft/source/nemotron_sft_v3/math_v3/train"):
        config = materialized_config(mix.exp3_v5_sources()[name], prefix)
        seal = mix._SEALS[name]
        assert config["cache_dir"] == mix._CACHE_ROOT
        assert config["shards"] == seal["shards"]
        for shard in config["shards"]:
            # Relative, content-addressed shard paths under the pinned root — never a bare prefix.
            assert not shard.startswith("s3://")
            assert len(shard.split("/components/")) == 2
        assert config["rows"] == seal["rows"]
        assert config["input_ids_tokens"] == seal["input_ids_tokens"]
        assert config["loss_weights_tokens"] == seal["loss_weights_tokens"]
        assert config["loss_mode"] == seal["loss_mode"]
        assert config["pack"] == 256
        assert config["packing_lookahead"] == 128
        assert config["split"] == "train" and config["flat_cache"] is True
        assert config["tags"] is None
        assert config["release"]["sha256"] == mix._RELEASE_SHA256


def test_seal_record_resolves_as_a_component(tmp_path):
    """A seeded seal record yields the component ``mixture`` reads: prebuilt, packed, flat."""
    prefix = str(tmp_path)
    handle = mix.exp3_v5_sources()["pretrain/source/c00q0-train/gt_64k"]
    write_record(
        ArtifactRecord(
            name=handle.name,
            version=handle.version,
            output_path=handle.path(prefix),
            result_type=_TOKENIZED_CACHE,
            config=materialized_config(handle, prefix),
        )
    )
    cache = TokenizedCache.raw_load(handle.path(prefix))
    component = cache.as_component()
    assert component.cache_dir == mix._CACHE_ROOT
    assert component.format == PrebuiltLmDatasetFormat(input_ids_key="input_ids", loss_weights_key="loss_weights")
    assert component.pack == 256
    assert component.split == "train"
    assert component.flat_cache is True
    assert component.tags is None
    assert cache.tokenizer == mix._TOKENIZER


def test_seal_check_rejects_a_drifted_catalog(tmp_path):
    """The seal check validates digests, rows, shard counts, and loss modes."""
    check = mix.exp3_v5_seal_check()
    config = materialized_config(check, str(tmp_path))
    name = "pretrain/source/c00q0-train/gt_64k"
    seal = mix._SEALS[name]
    catalog = {
        "splits": {
            "train": {
                source: {
                    "cache_dir": mix._CACHE_ROOT,
                    "ledger": {
                        "is_finished": True,
                        "total_num_rows": source_seal["rows"],
                        "finished_shards": source_seal["shards"],
                        "field_counts": {"input_ids": source_seal["input_ids_tokens"]},
                        "metadata": {"preprocessor_metadata": {"loss_mode": source_seal["loss_mode"]}},
                    },
                }
                for source, source_seal in mix._SEALS.items()
            }
        }
    }
    raw = json.dumps(catalog).encode()
    path = tmp_path / "catalog.json"
    path.write_bytes(raw)

    good = dict(config, cache_catalog={"uri": str(path), "sha256": hashlib.sha256(raw).hexdigest()})
    check.run(good)

    bad_sha = dict(config, cache_catalog={"uri": str(path), "sha256": "0" * 64})
    with pytest.raises(ValueError, match="digest differs"):
        check.run(bad_sha)

    drifted_rows = dict(good, expected_sources=dict(good["expected_sources"]))
    drifted_rows["expected_sources"][name] = dict(drifted_rows["expected_sources"][name], rows=seal["rows"] + 1)
    with pytest.raises(ValueError, match="rows"):
        check.run(drifted_rows)

    drifted_loss = dict(good, expected_sources=dict(good["expected_sources"]))
    drifted_loss["expected_sources"][name] = dict(drifted_loss["expected_sources"][name], loss_mode="chat_assistant_only")
    with pytest.raises(ValueError, match="loss mode"):
        check.run(drifted_loss)


@pytest.mark.skipif(not _HAS_S3_CREDENTIALS, reason="CoreWeave S3 credentials unavailable")
def test_sealed_catalog_matches_the_manifest_online():
    """The committed manifest still matches the live sealed catalog, and shards exist."""
    from rigging.filesystem.s3_compat import configure_coreweave_s3
    from rigging.filesystem.storage_path import StoragePath

    configure_coreweave_s3()
    raw = StoragePath(mix._CATALOG_URI).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == mix._CATALOG_SHA256
    catalog = json.loads(raw)["splits"]["train"]
    assert set(catalog) == set(mix._SEALS)
    for name, seal in mix._SEALS.items():
        ledger = catalog[name]["ledger"]
        assert ledger["total_num_rows"] == seal["rows"]
        assert ledger["finished_shards"] == seal["shards"]
    # A spot check that the content-addressed shard caches are still in place.
    for name in ("pretrain/source/c00q0-train/gt_64k", "sft/source/nemotron_sft_v3/math_v3/train"):
        for shard in mix._SEALS[name]["shards"][:2]:
            assert StoragePath(f"{mix._CACHE_ROOT}/{shard}").exists()


def test_data_config_is_fingerprint_safe():
    """Assembling the mix at fingerprint time reads no records and pulls no real paths."""
    cfg = mix.exp3_v5_data_config(StepContext.for_fingerprint(deps=mix.exp3_v5_mix_deps()))
    assert len(cfg.components) == 434
    assert all("@" in component.cache_dir for component in cfg.components.values())
    assert cfg.cache_catalog == "sft-mixes/exp3-v5/cache-catalog@2026.11.16"
    assert cfg.train_weights == [(phase, weights) for phase, weights in mix._SCHEDULE]


def test_consumer_lowering_carries_every_source():
    spec = _consumer().lower()
    dep_names = [dep.name for dep in spec.deps]
    assert len(dep_names) == 434 + 3
    assert "sft-mixes/exp3-v5/cache-catalog" in dep_names
    assert mix._SOURCE_NAMES[0] in dep_names
