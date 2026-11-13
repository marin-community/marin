# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner
from marin.execution.artifact import Artifact
from marin.execution.lazy import ArtifactStep
from marin.experiment.cli import graph_handles
from marin.rl.skyrl import ArtifactDataSource, ArtifactHfModel
from marin.training.training import LevanterCheckpoint

from experiments.post_training.mochi_rl.launch import build_run, main


@pytest.mark.parametrize("resume_step", [None, 13])
def test_cli_renders_direct_training_and_native_continuation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, resume_step: int | None
) -> None:
    monkeypatch.setenv("MARIN_PREFIX", "s3://marin-us-east-08a")
    config_path = tmp_path / "launch.yaml"
    model, tokenizer, first_tasks, second_tasks = (
        tmp_path / name for name in ("model", "tokenizer", "tasks-a", "tasks-b")
    )
    args = [
        "--version",
        "2026.11.13.1",
        "--input-version",
        "2026.11.13.0",
        "--model-uri",
        str(model),
        "--tokenizer-uri",
        str(tokenizer),
        "--tokenizer-revision",
        "pinned-tokenizer",
        "--data-uri",
        str(first_tasks),
        "--data-uri",
        str(second_tasks),
        "--max-steps",
        str((resume_step or 0) + 3),
        "--config-output",
        str(config_path),
    ]
    native = tmp_path / f"global_step_{resume_step}"
    if resume_step is not None:
        args.extend(("--resume-uri", str(native)))
    result = CliRunner().invoke(main, args)
    assert result.exit_code == 0, result.output
    launch = yaml.safe_load(config_path.read_text())
    trainer, generator = launch["skyrl"]["trainer"], launch["skyrl"]["generator"]

    placement = trainer["placement"]
    policy_gpus = placement["policy_num_nodes"] * placement["policy_num_gpus_per_node"]
    inference_gpus = generator["num_inference_engines"] * generator["inference_engine_data_parallel_size"]
    allocation = launch["iris"]["allocation"]
    assert (policy_gpus, inference_gpus) == (48, 12)
    assert allocation["num_nodes"] * allocation["gpus_per_node"] == policy_gpus + inference_gpus
    assert launch["skyrl"]["context_budget"]["request_window_tokens"] == 65536
    assert (trainer["algorithm"]["advantage_estimator"], generator["n_samples_per_prompt"]) == ("rloo_n", 8)
    assert trainer["policy"]["optimizer_config"]["optimizer_kwargs"]["expert_normalization"] == "all_experts"

    inputs = launch["inputs"]
    assert (inputs["model"]["uri"], inputs["model"]["tokenizer_uri"]) == (str(model), str(tokenizer))
    assert {source["uri"] for source in inputs["train_data"]} == {str(first_tasks), str(second_tasks)}
    assert len({source["local_path"] for source in inputs["train_data"]}) == 2
    assert trainer["max_steps"] == (resume_step or 0) + 3
    assert trainer["epochs"] >= trainer["max_steps"]
    assert trainer["restore_dataloader_state"] and not trainer["reset_global_step_on_resume"]
    assert launch["run"]["mode"] == "train" and not launch["run"]["export_hf"]
    assert trainer["callbacks"][0]["save_on_train_end"]
    assert trainer["callbacks"][0]["save_steps"] == trainer["ckpt_interval"]
    assert "/tmp/ttl=3d/" in launch["artifacts"]["checkpoint_root"]
    assert trainer["resume_mode"] == ("latest" if resume_step is None else "from_path")
    if resume_step is not None:
        assert trainer["resume_path"] == str(native)
        assert launch["artifacts"]["checkpoint_root"] != str(native.parent)


def test_native_checkpoint_is_an_input_to_one_training_stage(tmp_path: Path) -> None:
    model = ArtifactStep.adopt("inputs/model", "2026.11.13", str(tmp_path / "model"), kind=LevanterCheckpoint)
    tasks = ArtifactStep.adopt("inputs/tasks", "2026.11.13", str(tmp_path / "tasks"))
    native = ArtifactStep.adopt("inputs/native", "2026.11.13", str(tmp_path / "native"), kind=Artifact)
    step = build_run(
        name="checkpoints/direct-mochi",
        version="2026.11.13",
        model=ArtifactHfModel(model, str(tmp_path / "tokenizer"), "pinned", relative_path=""),
        train_data=(ArtifactDataSource(tasks, "train.parquet"),),
        max_steps=16,
        resume=native,
    )
    graph = graph_handles([step])
    assert [handle for handle in graph if handle.adopt_source is None] == [step]
    assert set(graph) == {model, tasks, native, step}
