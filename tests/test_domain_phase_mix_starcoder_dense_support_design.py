# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from types import SimpleNamespace
from typing import Any

import jax
import numpy as np
import pytest
from marin.execution.lazy import lower
from marin.execution.step_status import STATUS_SUCCESS

from experiments.domain_phase_mix import launch_starcoder_wsd80_dense_support_surfaces as launcher
from experiments.domain_phase_mix import launch_starcoder_wsd_80_20_surface as base
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    design_starcoder_wsd80_dense_support_surfaces_20260808 as design,
)


def test_dense_support_manifest_has_valid_frozen_hash_and_environment() -> None:
    manifest = json.loads(design.OUTPUT_PATH.read_text(encoding="utf-8"))
    claimed_hash = manifest.pop("design_sha256")

    assert design.canonical_sha256(manifest) == claimed_hash
    assert manifest["design_environment"] == {
        "jax_version": design.DESIGN_JAX_VERSION,
        "numpy_version": design.DESIGN_NUMPY_VERSION,
        "jax_default_prng_impl": design.DESIGN_JAX_DEFAULT_PRNG_IMPL,
        "jax_enable_x64": design.DESIGN_JAX_ENABLE_X64,
        "uv_lock_sha256": design.UV_LOCK_SHA256,
    }
    assert manifest["training_environment"] == launcher.EXPECTED_TRAINING_ENVIRONMENT


def test_dense_support_manifest_matches_generator_in_frozen_environment() -> None:
    if jax.__version__ != design.DESIGN_JAX_VERSION or np.__version__ != design.DESIGN_NUMPY_VERSION:
        pytest.skip(
            f"Frozen design requires jax={design.DESIGN_JAX_VERSION}, numpy={design.DESIGN_NUMPY_VERSION}; "
            f"current environment has jax={jax.__version__}, numpy={np.__version__}"
        )

    manifest = json.loads(design.OUTPUT_PATH.read_text(encoding="utf-8"))
    manifest.pop("design_sha256")
    assert design.build_payload() == manifest


def test_coverage_gate_accepts_complete_artifacts(monkeypatch: Any) -> None:
    class CompleteStep:
        def path(self) -> str:
            return "gs://marin-us-central1/complete"

    class CompleteStatusFile:
        def __init__(self, path: str, *, worker_id: str):
            assert path == "gs://marin-us-central1/complete"
            assert worker_id == "dense-support-coverage-gate"
            self.status = STATUS_SUCCESS

    monkeypatch.setattr(launcher, "build_training_steps", lambda **_: (CompleteStep(),))
    monkeypatch.setattr(launcher, "StatusFile", CompleteStatusFile)

    launcher._require_complete_coverage(
        name_prefix=launcher.NAME,
        tpu_type=base.DEFAULT_TPU_TYPE,
        tpu_region=base.DEFAULT_TPU_REGION,
        tpu_zone=base.DEFAULT_TPU_ZONE,
    )


def test_coverage_gate_path_matches_runner_output(monkeypatch: Any) -> None:
    manifest = json.loads(design.OUTPUT_PATH.read_text(encoding="utf-8"))
    run_name = next(row["run_name"] for row in manifest["runs"] if row["replicate_kind"] == "coverage")
    marin_prefix = "gs://marin-us-central1"
    monkeypatch.setenv("MARIN_PREFIX", marin_prefix)

    (step,) = launcher.build_training_steps(
        name_prefix=launcher.NAME,
        tpu_type=base.DEFAULT_TPU_TYPE,
        tpu_region=base.DEFAULT_TPU_REGION,
        tpu_zone=base.DEFAULT_TPU_ZONE,
        selected_runs=frozenset({run_name}),
        selected_replicate_kind="coverage",
    )

    assert step.path() == lower(step).output_path
    assert step.path().startswith(f"{marin_prefix}/")


def test_dense_support_sequence_counts_match_mixture_dataset() -> None:
    assert launcher.audit_runtime_sequence_counts() == launcher.EXPECTED_UNIQUE_SEQUENCE_IDENTITIES


def test_starcoder_source_validation_accepts_complete_legacy_cache(monkeypatch: Any) -> None:
    expected_shards = [f"{index:02d}_json_gz" for index in range(launcher.EXPECTED_STARCODER_CACHE_SHARDS)]
    ledger = SimpleNamespace(
        total_num_rows=launcher.EXPECTED_STARCODER_CACHE_DOCUMENTS,
        shard_rows={shard: 1 for shard in expected_shards},
        is_finished=True,
        finished_shards=expected_shards,
        field_counts={},
        layout=launcher.CACHE_LAYOUT_CONSOLIDATED,
        metadata=SimpleNamespace(preprocessor_metadata=launcher.EXPECTED_STARCODER_CACHE_TOKENIZER_METADATA),
    )
    monkeypatch.setattr(launcher.CacheLedger, "load", lambda path: ledger)
    _, requests = launcher.load_design(selected_replicate_kind="coverage")

    observed = launcher._validate_starcoder_source(base.DEFAULT_MARIN_PREFIX, requests)

    assert observed == "gs://marin-us-central1/tokenized/dolma/starcoder-8b6089"


def test_starcoder_source_validation_treats_registry_token_count_as_provenance(monkeypatch: Any) -> None:
    expected_shards = [f"{index:02d}_json_gz" for index in range(launcher.EXPECTED_STARCODER_CACHE_SHARDS)]
    ledger = SimpleNamespace(
        total_num_rows=launcher.EXPECTED_STARCODER_CACHE_DOCUMENTS,
        shard_rows={shard: 1 for shard in expected_shards},
        is_finished=True,
        finished_shards=expected_shards,
        field_counts={"legacy_tokens": 1},
        layout=launcher.CACHE_LAYOUT_CONSOLIDATED,
        metadata=SimpleNamespace(preprocessor_metadata=launcher.EXPECTED_STARCODER_CACHE_TOKENIZER_METADATA),
    )
    monkeypatch.setattr(launcher.CacheLedger, "load", lambda path: ledger)
    _, requests = launcher.load_design(selected_replicate_kind="coverage")

    observed = launcher._validate_starcoder_source(base.DEFAULT_MARIN_PREFIX, requests)

    assert observed == "gs://marin-us-central1/tokenized/dolma/starcoder-8b6089"


def test_materialized_runtime_configs_cover_every_cell_support_block(monkeypatch: Any) -> None:
    def load_cache(_cls: type[launcher.TokenizedCache], path: str) -> launcher.TokenizedCache:
        cache = launcher.TokenizedCache(path=path)
        cache.__dict__["record"] = SimpleNamespace(
            config={
                "format": {"text_key": "text"},
                "tags": [],
                "tokenizer": "meta-llama/Meta-Llama-3.1-8B",
            },
            source=None,
        )
        return cache

    monkeypatch.setattr(launcher.TokenizedCache, "raw_load", classmethod(load_cache))
    _, coverage = launcher.load_design(selected_replicate_kind="coverage")
    representative_names = {
        request.run_name
        for request in {(request.cell_id, request.support_id): request for request in reversed(coverage)}.values()
    }
    _, requests = launcher.load_design(
        selected_runs=frozenset(representative_names),
        selected_replicate_kind="coverage",
    )
    steps = launcher.build_training_steps(
        name_prefix=launcher.NAME,
        tpu_type=base.DEFAULT_TPU_TYPE,
        tpu_region=base.DEFAULT_TPU_REGION,
        tpu_zone=base.DEFAULT_TPU_ZONE,
        selected_runs=frozenset(representative_names),
        selected_replicate_kind="coverage",
    )

    assert launcher.audit_materialized_runtime_configs(requests, steps, marin_prefix=base.DEFAULT_MARIN_PREFIX) == 28
