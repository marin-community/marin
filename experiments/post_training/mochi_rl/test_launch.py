# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from inspect import unwrap
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner
from marin.execution.build_context import BuildContext, VersionCodex, build_context
from marin.experiment.cli import graph_handles
from marin.rl.cli import _coordinator_request
from rigging.timing import Duration

from experiments.post_training.mochi_rl.launch import main


@pytest.mark.parametrize("resume_step", [None, 13])
def test_cli_renders_direct_training_and_native_continuation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, resume_step: int | None
) -> None:
    monkeypatch.setenv("MARIN_PREFIX", "s3://marin-us-east-08a")
    config_path = tmp_path / "launch.yaml"
    model, tokenizer, first_tasks, second_tasks = (
        tmp_path / name for name in ("model", "tokenizer", "tasks-a", "tasks-b")
    )
    args = f"""--version 2026.11.13.1 --input-version 2026.11.13.0
        --model-uri "{model}" --tokenizer-uri "{tokenizer}" --tokenizer-revision pinned-tokenizer
        --data-uri "{first_tasks}" --data-uri "{second_tasks}"
        --max-steps {(resume_step or 0) + 3} --job-timeout-hours 48
        --config-output "{config_path}"
    """
    native = tmp_path / f"global_step_{resume_step}"
    if resume_step is not None:
        args += f' --resume-uri "{native}"'
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
    assert launch["iris"]["timeout"] == 48 * 3600

    inputs = launch["inputs"]
    assert (inputs["model"]["uri"], inputs["model"]["tokenizer_uri"]) == (str(model), str(tokenizer))
    assert {source["uri"] for source in inputs["train_data"]} == {str(first_tasks), str(second_tasks)}
    assert len({source["local_path"] for source in inputs["train_data"]}) == 2
    assert trainer["max_steps"] == (resume_step or 0) + 3
    assert trainer["epochs"] >= trainer["max_steps"]
    assert launch["run"]["mode"] == "train" and not launch["run"]["export_hf"]
    assert trainer["resume_mode"] == "latest"
    if resume_step is not None:
        assert inputs["starting_checkpoint"]["uri"] == str(native)


def test_continuations_keep_distinct_inputs_and_coordinator_queue_time(tmp_path: Path) -> None:
    checkpoints = []
    for version in ("2026.11.13.1", "2026.11.13.2"):
        with build_context(BuildContext(versions=VersionCodex(default=version))):
            step = unwrap(main.callback)(
                model_uri=str(tmp_path / "model"),
                tokenizer_uri=None,
                tokenizer_revision="pinned",
                data_uri=(str(tmp_path / "tasks"),),
                data_file="train.parquet",
                input_version="2026.11.13.0",
                name="checkpoints/continuation",
                max_steps=25,
                job_timeout_hours=72,
                resume_uri=str(tmp_path / version / "checkpoint"),
                checkpoint_count=2,
                ttl_days=3,
                config_output=None,
            )
        graph = graph_handles([step])
        assert [handle for handle in graph if handle.adopt_source is None] == [step]
        checkpoints.append(next(dep for dep in graph if dep.name.endswith("/inputs/resume")))
    assert len({(checkpoint.name, checkpoint.version) for checkpoint in checkpoints}) == 2
    env = {"DAYTONA_RL_API_KEY": "test", "DAYTONA_EVAL_API_KEY": "test", "MARIN_PREFIX": "s3://marin-us-east-08a"}
    _, request = _coordinator_request([step], "experiments.post_training.mochi_rl.launch", (), 1, tmp_path, env)
    assert request.timeout >= Duration.from_hours(120)
