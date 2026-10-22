# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, replace
from pathlib import Path

import pytest
from marin.execution.artifact import read_record
from marin.execution.lazy import ArtifactStep, run
from marin.training.training import LevanterCheckpoint
from rigging.provenance import Provenance

from experiments.post_training import snowball_inventory_loop as loop
from experiments.post_training.snowball_inventory_compare import Initializer
from experiments.post_training.snowball_inventory_evaluate import InventoryEvaluation
from experiments.post_training.snowball_inventory_panel import DevelopmentPanel


@dataclass(frozen=True)
class LocalTraining:
    output: str


@dataclass(frozen=True)
class LocalScore:
    output: str
    model: str
    baseline: bool


def loop_options(tmp_path, max_steps=25, version="2026.10.21.1"):
    return {
        "initializer": Initializer.SFT,
        "version": version,
        "max_steps": max_steps,
        "model_uri": str(tmp_path / "initializer"),
        "model_manifest_identity": "a" * 64,
        "tokenizer_template_sha256": "b" * 64,
        "pool_uri": str(tmp_path / "pool"),
        "pool_version": "2026.10.21",
        "pool_sha256": "c" * 64,
        "validation_uri": str(tmp_path / "full"),
        "validation_version": "2026.10.21",
        "validation_sha256": "d" * 64,
        "panel": DevelopmentPanel.FAST,
        "panel_uri": str(tmp_path / "fast"),
        "panel_version": "2026.10.21",
        "panel_sha256": "e" * 64,
        "judge_base_url": "https://judge.example/v1",
        "judge_model": "example-judge",
    }


def test_scoring_failure_preserves_export_and_warm_baseline_across_iterations(tmp_path, monkeypatch):
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path / "artifacts"))
    calls = {"baseline": 0, "training": 0, "final": 0}
    fail_scoring = True

    def train(config):
        output = config.output
        calls["training"] += 1
        Path(output).mkdir(parents=True, exist_ok=True)
        (Path(output) / "committed-export.json").write_text('{"global_step":25}')
        return LevanterCheckpoint(path=output)

    def training_step(initializer, *, version, **kwargs):
        return ArtifactStep(
            name=f"checkpoints/{initializer}",
            version=version,
            artifact_type=LevanterCheckpoint,
            run=train,
            build_config=lambda ctx: LocalTraining(ctx.output_path),
        )

    def score(config):
        kind = "baseline" if config.baseline else "final"
        calls[kind] += 1
        if not config.baseline:
            assert (Path(config.model) / "committed-export.json").exists()
            if fail_scoring:
                raise RuntimeError("original grader unavailable")
        return InventoryEvaluation(
            path=config.output,
            report_uri=f"{config.output}/report.json",
            pairing_identity="fixed-paired-panel",
            model_uri=config.model,
            complete=True,
        )

    def scoring_step(*, name, version, model, expected_model_manifest_identity=None, **kwargs):
        return ArtifactStep(
            name=name,
            version=version,
            artifact_type=InventoryEvaluation,
            run=score,
            build_config=lambda ctx: LocalScore(
                ctx.output_path, ctx.artifact_path(model), expected_model_manifest_identity is not None
            ),
            deps=(model,),
        )

    # Replace only the GPU/grader boundaries. The iteration DAG, artifact runner,
    # content cache, persisted records and retry scheduling remain real.
    monkeypatch.setattr(loop, "comparison_step", training_step)
    monkeypatch.setattr(loop, "recoverable_evaluation_step", scoring_step)
    monkeypatch.setattr(loop, "run_recoverable_evaluation", score)
    provenance = Provenance("frozen-runtime", "fixture", True, None, None)
    monkeypatch.setattr(loop, "launch_provenance", lambda: provenance)
    runtime_lock = tmp_path / "runtime.lock"
    runtime_lock.write_text("frozen-runtime-lock")
    monkeypatch.setattr(loop, "RUNTIME_LOCK", runtime_lock)
    options = loop_options(tmp_path)
    with pytest.raises(ValueError):
        loop.iteration_step(**options)
    provenance = replace(provenance, tree_hash="", dirty=False)
    with pytest.raises(ValueError):
        loop.iteration_step(**options)
    assert calls == {"baseline": 0, "training": 0, "final": 0}
    provenance = replace(provenance, tree_hash="frozen-runtime")
    Path(options["model_uri"]).mkdir()
    with pytest.raises(RuntimeError, match=r"1 step\(s\) failed"):
        run(loop.iteration_step(**options), max_concurrent=1)
    export = tmp_path / "artifacts" / "checkpoints" / "sft" / options["version"]
    assert read_record(str(export)) is not None
    assert (export / "committed-export.json").read_text() == '{"global_step":25}'
    assert calls == {"baseline": 1, "training": 1, "final": 1}

    fail_scoring = False
    [result] = run(loop.iteration_step(**options), max_concurrent=1)
    assert result.complete
    assert calls == {"baseline": 1, "training": 1, "final": 2}
    run(loop.iteration_step(**loop_options(tmp_path, max_steps=50, version="2026.10.21.2")), max_concurrent=1)
    assert calls == {"baseline": 1, "training": 2, "final": 3}

    provenance = replace(provenance, tree_hash="changed-runtime")
    run(loop.iteration_step(**loop_options(tmp_path, version="2026.10.21.3")), max_concurrent=1)
    assert calls == {"baseline": 2, "training": 3, "final": 4}
    runtime_lock.write_text("updated-runtime-lock")
    run(loop.iteration_step(**loop_options(tmp_path, version="2026.10.21.4")), max_concurrent=1)
    assert calls == {"baseline": 3, "training": 4, "final": 5}
