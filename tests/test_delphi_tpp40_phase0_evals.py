# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from dataclasses import asdict
from pathlib import Path

import fsspec
import pytest
from levanter.data.text.datasets import DatasetComponent

from experiments.domain_phase_mix import launch_delphi_tpp40_phase0_evals as boundary


@pytest.fixture
def checkpoint(tmp_path: Path) -> boundary.BoundaryCheckpoint:
    spec = boundary.base.DelphiSwarmRunSpec(
        run_order=89,
        run_id=7141089,
        run_name="fit_089_run_00089",
        source_run_name="run_00089",
        source_experiment="test",
        panel_source="test",
        target_flops=3.08e19,
        tpu_type="v5p-8",
        tpu_region="us-east5",
        tpu_zone="us-east5-a",
        batch_size=128,
        train_steps=27336,
        realized_train_tokens=14331936768,
        expected_checkpoint_step=27335,
        model_hidden_dim=896,
        model_layers=10,
        non_embedding_params=128469376,
        total_trainable_params=358304128,
        tensor_parallel_size=1,
        data_seed=7141089,
        trainer_seed=0,
        phase_boundary=0.8,
        phase_0_fraction=0.8,
        phase_1_fraction=0.2,
        simulated_epoch_target_budget=6325183647689,
        available_top_level_tokens=6986431605135,
        max_simulated_epoch=2.0,
        q95_simulated_epoch=1.0,
        mean_phase_tv_to_proportional=0.0,
        phase_weights={"phase_0": {"bucket": 1.0}, "phase_1": {"bucket": 1.0}},
    )
    root = tmp_path / "fit_089_run_00089-123456"
    path = root / "checkpoints" / "step-21855"
    path.mkdir(parents=True)
    (path / "metadata.json").write_text(json.dumps({"step": 21855, "is_temporary": False}))
    (path / "manifest.ocdbt").write_text("tensor manifest")
    training_path = fsspec.filesystem("file").unstrip_protocol(str(root))
    (root / ".executor_info").write_text(
        json.dumps(
            {
                "config": {
                    "output_path": training_path,
                    "analysis_output_path": "analysis",
                    "run_spec": asdict(spec),
                }
            }
        )
    )
    return boundary.ready_checkpoints(str(tmp_path), (89,))[0]


def test_ready_discovery_respects_assignment_and_requires_payload(tmp_path, checkpoint):
    assert boundary.ready_checkpoints(str(tmp_path), (90,)) == []
    discovered = boundary.ready_checkpoints(str(tmp_path), (89, 90))
    assert [item.run_spec.run_order for item in discovered] == [89]
    assert discovered[0].run_spec.data_seed == 7141089
    fs, path = fsspec.core.url_to_fs(checkpoint.checkpoint_path)
    fs.rm(f"{path}/manifest.ocdbt")
    with pytest.raises(ValueError, match="tensor payload"):
        boundary.ready_checkpoints(str(tmp_path), (89,))


def test_ready_discovery_rejects_wrong_training_identity(tmp_path, checkpoint):
    path = f"{checkpoint.training_output_path}/.executor_info"
    info = boundary.bridge._read_json(path)
    config = info["config"]
    assert isinstance(config, dict)
    config["run_spec"]["run_order"] = 90
    boundary.bridge._write_json(path, info)
    with pytest.raises(ValueError, match="identity"):
        boundary.ready_checkpoints(str(tmp_path), (89,))


def test_validation_requires_paloma_as_well_as_uncheatable(checkpoint):
    components = {name: DatasetComponent(cache_dir="unused") for name in ("paloma/ptb", "uncheatable_eval/bbc_news")}
    config = boundary.BoundaryEvalConfig(checkpoint, components, "output")
    metrics = {"eval/uncheatable_eval/bbc_news/bpb": 1.1}
    with pytest.raises(ValueError, match="paloma/ptb"):
        boundary.validation_result(metrics, config)
    metrics["eval/paloma/ptb/bpb"] = 1.2
    result = boundary.validation_result(metrics, config)
    assert result["checkpoint_step"] == 21855
    assert result["metrics"] == metrics
    assert result["data_seed"] == 7141089


def test_completed_validation_reuses_exact_saved_result(tmp_path, checkpoint):
    components = {"paloma/ptb": DatasetComponent(cache_dir="original-cache")}
    config = boundary.BoundaryEvalConfig(checkpoint, components, str(tmp_path / "eval"))
    assert boundary.completed_validation_result(config) is None
    result = boundary.validation_result({"eval/paloma/ptb/bpb": 1.2}, config)
    boundary.bridge._write_json(f"{config.output_path}/{boundary.RESULT_FILE}", result)
    assert boundary.completed_validation_result(config) == result


@pytest.mark.parametrize("field", ["checkpoint_metadata_sha256", "data_seed", "validation_caches", "metrics"])
def test_completed_validation_rejects_stale_or_incomplete_result(tmp_path, checkpoint, field):
    components = {"paloma/ptb": DatasetComponent(cache_dir="original-cache")}
    config = boundary.BoundaryEvalConfig(checkpoint, components, str(tmp_path / "eval"))
    result = boundary.validation_result({"eval/paloma/ptb/bpb": 1.2}, config)
    result[field] = {} if field in {"validation_caches", "metrics"} else "wrong"
    boundary.bridge._write_json(f"{config.output_path}/{boundary.RESULT_FILE}", result)
    with pytest.raises(ValueError):
        boundary.completed_validation_result(config)


def test_table9_completion_rejects_partial_result(tmp_path):
    path = str(tmp_path / "table9")
    result = {"table9_macro_bpb": 1.1, "table9_components": {"only_one": 1.1}}
    boundary.bridge._write_json(f"{path}/{boundary.RESULTS_FILENAME}", result)
    with pytest.raises(ValueError, match="complete Table-9"):
        boundary.require_complete_table9({89: path})


def test_eval_graph_does_not_recreate_training_and_reuses_paths(checkpoint, tmp_path):
    components = {"paloma/ptb": DatasetComponent(cache_dir="gs://marin-us-east5/tokenized/paloma/ptb")}
    first = boundary.build_steps([checkpoint], components, side="east5")
    second = boundary.build_steps([checkpoint], components, side="east5")
    prefix = str(tmp_path)
    assert boundary.result_paths(first, prefix) == boundary.result_paths(second, prefix)
    table9 = first[0].config.eval_config
    assert table9.provenance["source_checkpoint"].endswith("/checkpoints/step-21855")
    resolver = boundary.Executor(prefix=prefix, executor_info_base_path=str(tmp_path / "experiments"))
    with boundary.executor_context():
        resolver.compute_version(first[0], is_pseudo_dep=False)
    assert len(resolver.steps) == 2
    assert all("run_delphi_swarm_training" not in str(step.fn) for step in resolver.steps)
