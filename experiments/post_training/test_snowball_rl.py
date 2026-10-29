# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import subprocess
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import click
import marin.rl.skyrl as adapter
import pytest
import yaml
from iris.rpc import controller_pb2, job_pb2
from marin.execution.fingerprint import fingerprint_hash
from marin.execution.lazy import ArtifactStep, StepContext
from marin.rl.skyrl import SkyRLRun, skyrl_cpu_command
from marin.training.training import LevanterCheckpoint
from rigging.filesystem.storage_path import StoragePath
from rigging.provenance import Provenance

from experiments.post_training import snowball_rl as snowball
from experiments.post_training import snowball_rl_presets as presets


def _render(step, path):
    config = step.build_config(StepContext.for_fingerprint(step.runtime_args, step.deps))
    path.write_text(config.launch_config_yaml)
    return yaml.safe_load(config.launch_config_yaml)


def _native(script, paths):
    result = subprocess.run(
        skyrl_cpu_command("python", "-c", script, *paths), capture_output=True, text=True, timeout=170
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.splitlines()[-1])


pytestmark = pytest.mark.usefixtures("source_provenance")


@pytest.mark.timeout(180)  # The real bootstrap installs its independent, locked CPU environment.
def test_presets_compose_and_preserve_measurement_contracts(selection, tmp_path):
    paths = []
    launches = {}
    for name, preset in presets.PRESETS.items():
        layouts = (presets.L112, presets.P40) if name == "train-long" else (None,)
        for layout in layouts:
            run = snowball.build_run(
                preset, tasks=selection if preset.focus == presets.Focus.DATA else None, layout=layout
            )
            for index, step in enumerate((*run.runs, *run.post_run_evaluations)):
                path = tmp_path / f"{name}-{layout.name if layout else 'default'}-{index}.yaml"
                document = _render(step, path)
                paths.append(str(path))
                if index == 0:
                    launches[name] = document
            if name == "train-long":
                assert run.runs[0] in run.post_run_evaluations[0].deps
    off_boundary = snowball.build_run(
        presets.PRESETS["train-long"], layout=presets.P40, settings=(presets.parse_setting("trainer.max_steps=2"),)
    ).runs[0]
    path = tmp_path / "off-boundary.yaml"
    _render(off_boundary, path)
    paths.append(str(path))
    changed = snowball.build_run(
        presets.PRESETS["learning-medium"],
        settings=tuple(
            map(
                presets.parse_setting,
                (
                    "trainer={train_batch_size: 64, policy_mini_batch_size: 64}",
                    "generator.n_samples_per_prompt=8",
                    "trainer.micro_train_batch_size_per_gpu=2",
                    "+trainer.policy.megatron_config.logprob_chunk_size=512",
                ),
            )
        ),
    ).runs[0]
    path = tmp_path / "changed-batch.yaml"
    _render(changed, path)
    paths.append(str(path))
    rows = _native(
        """
import json, sys
from cloud.iris.launch_config import load_launch_config
configs = [load_launch_config(path) for path in sys.argv[1:]]
print(json.dumps([{
    "name": config.run.id, "entrypoint": config.runtime.entrypoint,
    "nodes": config.iris.allocation.num_nodes, "eval_interval": config.skyrl.trainer.eval_interval,
    "validation": bool(config.inputs.validation_data),
    "admission": config.skyrl.trajectory_runner.max_concurrent_tasks,
    "window": config.skyrl.generator.engine_init_kwargs.max_model_len,
    "batch": config.skyrl.trainer.train_batch_size, "samples": config.skyrl.generator.n_samples_per_prompt,
    "tools": config.skyrl.generator.engine_init_kwargs.get("tool_parser"),
    "fixed": config.skyrl.generator.eval_sampling_params.ignore_eos or config.skyrl.generator.sampling_params.ignore_eos,
    "overlong_filtering": config.skyrl.generator.apply_overlong_filtering,
    "training": config.runtime.training_type is not None,
    "callbacks": [entry.type for entry in config.skyrl.trainer.callbacks],
    "steps": config.skyrl.trainer.max_steps,
    "micro": config.skyrl.trainer.micro_train_batch_size_per_gpu,
    "forward_micro": config.skyrl.trainer.micro_forward_batch_size_per_gpu,
    "terminal_eval": [entry.eval_on_train_end for entry in config.skyrl.trainer.callbacks if entry.type == "evaluation"],
} for config in configs]))
""",
        paths,
    )
    assert len(rows) == len(paths)
    for row in rows:
        name = row["name"]
        if "speed-short-" in name or "data-short-" in name or "/eval/panel-910" in name:
            assert row["entrypoint"] == "skyrl_train.entrypoints.main_generate", row
            assert row["nodes"] == 1, row
            assert row["eval_interval"] > 0 and row["validation"], row
            expected = 2048 if "speed-short-" in name else 1024 if "data-short-" in name else 910
            assert row["admission"] == expected, row
        if "/eval/panel-910" in name:
            assert row["window"] == (65536 if row["tools"] == "hermes" else 24576), row
        if "train-long-" in name and "/eval/" not in name:
            assert row["terminal_eval"] == [row["steps"] % 10 != 0], row
        if row["fixed"]:
            assert not row["overlong_filtering"], row
        if row["training"] and row["validation"]:
            assert "evaluation" in row["callbacks"], row
            if "best_checkpoint" in row["callbacks"]:
                assert row["callbacks"].index("evaluation") < row["callbacks"].index("best_checkpoint"), row
    assert (rows[-1]["batch"], rows[-1]["samples"], rows[-1]["micro"], rows[-1]["forward_micro"]) == (64, 8, 2, 2)
    # Candidate selection is the only difference between the two learning experiments.
    learning, data = launches["learning-medium"], launches["data-medium"]
    assert learning["skyrl"] == data["skyrl"]
    assert learning["inputs"]["model"] == data["inputs"]["model"]
    assert learning["inputs"]["validation_data"] == data["inputs"]["validation_data"]
    assert learning["inputs"]["train_data"] != data["inputs"]["train_data"]
    with pytest.raises(click.BadParameter, match="owned by"):
        snowball.build_run(
            presets.PRESETS["learning-medium"],
            settings=(presets.parse_setting("generator.sampling_params={ignore_eos: true}"),),
        )
    execution = next(value for value in changed.runtime_args.values() if isinstance(value, adapter.IrisSkyRLExecution))
    assert presets.JUDGE_KEY in execution.required_env_vars


@pytest.mark.timeout(180)
def test_gated_presets_match_calibrated_configs(selection, tmp_path, monkeypatch):
    summary = (Path(selection.path()) / "summary.json").read_text()
    monkeypatch.setattr(presets, "SELECTION_ROOT", "memory://snowball-calibration")
    (
        StoragePath(presets.SELECTION_ROOT) / "snowball-100k" / presets.DEFAULT_SELECTION_VERSION / "summary.json"
    ).write_text(summary)
    expected = {
        "speed-short": "cfef0c60",
        "data-short": "54709ff4",
        "speed-medium": "d230e889",
        "learning-medium": "670e44c0",
    }
    paths = []
    for name in expected:
        step = snowball.build_run(presets.PRESETS[name], control=True).runs[0]
        path = tmp_path / f"{name}.yaml"
        _render(step, path)
        paths.append(str(path))
    configs = _native(
        """
import json, os, sys
from pathlib import Path
from omegaconf import OmegaConf
from cloud.iris.launch_config import load_launch_config
os.environ["HOME"] = "/snowball-pin-home"
configs = [load_launch_config(Path(path)) for path in sys.argv[1:]]
print(json.dumps([
    {"skyrl": OmegaConf.to_container(config.skyrl, resolve=True),
     "allocation": OmegaConf.to_container(config.iris.allocation, resolve=True)}
    for config in configs
], sort_keys=True))
""",
        paths,
    )
    actual = {
        name: fingerprint_hash(json.dumps(config, sort_keys=True))
        for name, config in zip(expected, configs, strict=True)
    }
    assert actual == expected, "Gated config changed. Top-level key hashes: " + json.dumps(
        {
            name: {
                section: {key: fingerprint_hash(json.dumps(value, sort_keys=True)) for key, value in values.items()}
                for section, values in config.items()
            }
            for name, config in zip(expected, configs, strict=True)
        },
        sort_keys=True,
    )


def test_run_identity_tracks_code_inputs_and_replicates(selection, monkeypatch):
    learning = presets.PRESETS["learning-medium"]
    data = presets.PRESETS["data-short"]
    long = presets.PRESETS["train-long"]

    def run(preset=learning, *settings, **kwargs):
        return snowball.build_run(preset, settings=tuple(map(presets.parse_setting, settings)), **kwargs).runs[0]

    def name(preset=learning, *settings, **kwargs):
        return run(preset, *settings, **kwargs).name

    repeated = snowball.build_run(learning, replicates=3)
    assert [step.name for step in snowball.build_run(learning, replicates=2).runs] == [
        step.name for step in repeated.runs[:2]
    ]
    assert len({step.name for step in repeated.runs}) == 3
    initializer = ArtifactStep.adopt("tests/initializer", "2026.10.26", selection.path(), kind=LevanterCheckpoint)
    pairs = [
        (name(), name(learning, "trainer.policy.optimizer_config.lr=2e-6")),
        (name(data, tasks=selection), name(data, tasks=replace(selection, version="2026.10.27"))),
        (name(), name(init=initializer)),
        (name(long), name(long, layout=presets.P40)),
    ]
    assert all(left != right for left, right in pairs), pairs
    assert name(
        learning,
        "trainer.policy.optimizer_config={lr: 0.000002}",
        "trainer.policy.optimizer_config={weight_decay: 0.1}",
    ) == name(
        learning, "trainer.policy.optimizer_config.lr=0.000002", "trainer.policy.optimizer_config.weight_decay=0.1"
    )
    original = run(long)
    run(long, "trainer.policy.megatron_config.ddp_config.overlap_grad_reduce=true")
    assert run(long).fingerprint() == original.fingerprint()
    original_name = name()
    monkeypatch.setattr(
        snowball, "_provenance", lambda **kwargs: Provenance("tree-two", "commit-one", False, "branch", "test")
    )
    assert name() != original_name


def test_resume_reads_failed_run_without_rebuilding_it(selection, tmp_path, monkeypatch):
    original = snowball.build_run(presets.PRESETS["train-long"], layout=presets.P40).runs[0]
    document = yaml.safe_load(
        original.build_config(StepContext.for_fingerprint(original.runtime_args, original.deps)).launch_config_yaml
    )
    source = tmp_path / original.name / original.version
    source.mkdir(parents=True)
    checkpoints = source / "checkpoints"
    (checkpoints / "global_step_10").mkdir(parents=True)
    marker = checkpoints / "latest_ckpt_global_step.txt"
    marker.write_text("10")
    attempts = source / "attempts"
    attempts.mkdir()
    document["run"]["attempt_id"] = "attempt-fixture"
    document["iris"]["job_name"] = "train-long-fixture-attempt-fixture"
    document["artifacts"].update(checkpoint_root=str(checkpoints), attempts_root=str(attempts))
    (source / "resolved-launch.yaml").write_text(
        json.dumps({"config": document, "train_data_sources": [], "val_data_sources": []})
    )
    state = {"value": job_pb2.JOB_STATE_KILLED, "known": True}
    job_id = "/tests/parent/" + document["iris"]["job_name"]

    @contextmanager
    def endpoint(**kwargs):
        yield SimpleNamespace(url="http://iris.test", credentials=None)

    @contextmanager
    def rpc(*args):
        yield SimpleNamespace(
            list_jobs=lambda request: controller_pb2.Controller.ListJobsResponse(
                jobs=[job_pb2.JobStatus(job_id=job_id, state=state["value"])] if state["known"] else []
            )
        )

    monkeypatch.setattr(adapter, "connect_controller", endpoint)
    monkeypatch.setattr(adapter, "rpc_client", rpc)
    earlier = ArtifactStep.adopt(original.name, original.version, str(source), kind=SkyRLRun)
    resumed = snowball.build_run(presets.PRESETS["train-long"], layout=presets.P40, resume_from=earlier).runs[0]
    assert resumed.name != original.name
    assert all(dep.artifact_type is not SkyRLRun for dep in resumed.deps)
    config = resumed.build_config(
        StepContext.for_run(
            str(tmp_path / "resumed"),
            str(tmp_path / "artifacts"),
            deps=resumed.deps,
            runtime_args=resumed.runtime_args,
        )
    )
    assert config.output.checkpoint_root == str(checkpoints)
    assert yaml.safe_load(config.launch_config_yaml)["artifacts"]["checkpoint_root"] == str(checkpoints)
    state["known"] = False
    with pytest.raises(click.BadParameter, match="cannot identify the Iris job"):
        snowball.build_run(presets.PRESETS["train-long"], layout=presets.P40, resume_from=earlier)
    state["known"] = True
    state["value"] = job_pb2.JOB_STATE_RUNNING
    with pytest.raises(click.BadParameter, match=job_id):
        snowball.build_run(presets.PRESETS["train-long"], layout=presets.P40, resume_from=earlier)
    state["value"] = job_pb2.JOB_STATE_KILLED
    marker.unlink()
    with pytest.raises(click.BadParameter, match="14 days"):
        snowball.build_run(presets.PRESETS["train-long"], layout=presets.P40, resume_from=earlier)
