# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from omegaconf import OmegaConf, open_dict

from cloud.iris.artifacts import CHECKPOINT_MARKER_FILENAME
from cloud.iris.export_hf_checkpoint import ExportJobSpec, checkpoint_export_launch_config
from cloud.iris.launch import LaunchOutcome, execute_launch
from cloud.iris.launch_config import load_launch_config
from cloud.iris.skyrl_entrypoint import run_config
from marinskyrl.checkpoint_paths import policy_export_path
from skyrl_train.hf_export_schema import HFExportRequest
from skyrl_train.utils.tracking import Tracking
from finestore.rl.run_records import AttemptRecord, CheckpointReference, read_attempt, run_records_root, update_attempt
from finestore.store import DataStore


def main(path: Path) -> None:
    root = path.parent
    invalid = OmegaConf.load(path)
    invalid.iris.max_retries = 1
    invalid_path = root / "retry-enabled.yaml"
    invalid_path.write_text(OmegaConf.to_yaml(invalid))
    try:
        load_launch_config(invalid_path)
    except ValueError as error:
        assert "records backend requires iris.max_retries=0" in str(error), error
    else:
        raise AssertionError("retry-enabled records launch was accepted")
    import cloud.iris.launch, skyrl_train, taskcompendium, finestore.rl.run_records
    for module in (cloud.iris.launch, skyrl_train, taskcompendium, finestore.rl.run_records):
        assert Path(module.__file__).resolve().is_relative_to(Path.cwd()), module.__file__

    def history_config(attempt_id, export_root):
        return OmegaConf.create({"trainer": {"export_path": str(export_root)},
            "runtime": {"run_id": "history", "attempt_id": attempt_id}})

    ancestor_root = root / "ancestor-exports"
    with Tracking("tests-project", "history", ("records",), history_config("original", ancestor_root)) as original:
        original.start(resumed_step=None)
        original.log({"loss": 0.3}, step=3)
    parent = CheckpointReference(export_root=str(ancestor_root), attempt_id="original", step=3)
    with Tracking("tests-project", "history", ("records",), history_config("middle", root / "exports")) as middle:
        middle.start(resumed_step=3, checkpoint=parent)
        middle.log({"loss": 0.5, "eval/all/avg_score": 0.4}, step=5)
    try:
        with Tracking("tests-project", "history", ("records",), history_config("failed-direct", root / "exports")) as failed:
            failed.start(resumed_step=None)
            failed.log({"loss": 0.9}, step=1)
            raise RuntimeError("fixture workload failed")
    except RuntimeError as error:
        assert str(error) == "fixture workload failed", error

    class LocalBackend:
        def launch(self, config_path):
            config = load_launch_config(config_path)
            if config.run.mode == "train":
                prepared = read_attempt(str(config.artifacts.export_root), str(config.run.attempt_id))
                assert prepared is not None and prepared.result is None and prepared.index is not None
                assert prepared.index.run_id == "indexed-native-writers"
                assert prepared.native.config["run"]["coordinator_iris_job_id"] == "/local/coordinator"
            resolved = Path(config.artifacts.resolved_config_uri)
            resolved.parent.mkdir(parents=True, exist_ok=True)
            resolved.write_text(OmegaConf.to_yaml(config))
            if config.run.mode == "checkpoint_export":
                policy = Path(policy_export_path(str(config.artifacts.export_root), 8))
                policy.mkdir(parents=True, exist_ok=True)
                for name in ("config.json", "model.safetensors", "tokenizer.json"):
                    (policy / name).write_text("{}")
            elif config.run.attempt_id != "earlier-failed-attempt":
                with open_dict(config.skyrl):
                    config.skyrl.runtime = {"run_id": config.run.id, "attempt_id": config.run.attempt_id,
                        "iris_job_id": "/local/training", "coordinator_iris_job_id": None}
                with patch("cloud.iris.skyrl_entrypoint.version", return_value="fixture-vllm"):
                    interrupted = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
                    interrupted.run.attempt_id = "interrupted-attempt"
                    interrupted.run.coordinator_iris_job_id = "/local/interrupted-coordinator"
                    for launch_config, job_id in [(config, "/local/training"), (interrupted, "/local/interrupted")]:
                        startup_path = root / f"startup-{launch_config.run.attempt_id}.yaml"
                        startup_path.write_text(OmegaConf.to_yaml(launch_config))
                        main = SimpleNamespace(run=lambda _: None)
                        with (
                            patch("cloud.iris.skyrl_entrypoint.get_job_info", return_value=SimpleNamespace(job_id=job_id)),
                            patch.dict(sys.modules, {"skyrl_train.entrypoints.main_base": main}),
                        ):
                            run_config(startup_path)
                    foreign = OmegaConf.to_container(interrupted, resolve=True)
                    foreign["run"]["id"] = "unrelated-native-run"
                    foreign["run"]["attempt_id"] = "foreign-native"
                    with DataStore.open(run_records_root(str(config.artifacts.export_root))) as store:
                        update_attempt(store, AttemptRecord(run_id="unrelated-native-run", attempt_id="foreign-native",
                            config=foreign, started_at="2026-11-11T01:00:00+00:00"))
                startup = read_attempt(str(config.artifacts.export_root), str(config.run.attempt_id))
                assert startup is not None and startup.vllm_version == "fixture-vllm"
                (root / "startup-observation.json").write_text(startup.model_dump_json())
                with Tracking("tests-project", "record-boundary", ("records",), config.skyrl) as tracker:
                    tracker.start(resumed_step=5,
                        checkpoint=CheckpointReference(export_root=str(root / "exports"), attempt_id="middle", step=5))
                    immediate = read_attempt(str(config.artifacts.export_root), str(config.run.attempt_id))
                    assert immediate.metrics_started_at is not None and immediate.parent.attempt_id == "middle"
                    tracker.log({"startup/resume": 0.05, "eval/all/avg_score": 0.6}, step=5)
                    tracker.log({"loss": 1.25}, step=6)
                    tracker.log({"loss": 0.25, "entropy": 0.75}, step=6)
                checkpoints = Path(config.artifacts.checkpoint_root)
                checkpoints.mkdir(parents=True, exist_ok=True)
                (checkpoints / CHECKPOINT_MARKER_FILENAME).write_text("8\n")
            failed = config.run.attempt_id == "earlier-failed-attempt"
            return LaunchOutcome("/local/training", "failed" if failed else "succeeded", int(failed))

        def export_terminal_policy(self, config_path):
            config = load_launch_config(config_path)
            request = HFExportRequest(step=8, checkpoint_base_path=str(config.artifacts.checkpoint_root),
                checkpoint_path=str(Path(config.artifacts.checkpoint_root) / "global_step_8"),
                export_path=str(config.artifacts.export_root), model_path=str(config.skyrl.trainer.policy.model.path),
                model_source_uri=str(config.inputs.model.uri), model_source_identity=str(config.inputs.model.identity),
                num_nodes=1, gpus_per_node=4)
            spec = ExportJobSpec(request=request, cluster="local", priority="batch", gpu_variant="H100",
                job_name=None, timeout=60, no_wait=False)
            export = checkpoint_export_launch_config(config, request, spec)
            export_path = root / "export.yaml"
            export_path.write_text(OmegaConf.to_yaml(export))
            result = execute_launch(export_path, backend=self)
            assert result.state == "succeeded", result

        def validate(self, config_path):
            raise AssertionError("the test exercises wait-mode execution")

    config = OmegaConf.load(path)
    current = config.run.attempt_id
    config.run.attempt_id = "earlier-failed-attempt"
    config.run.export_hf = False
    path.write_text(OmegaConf.to_yaml(config))
    assert execute_launch(path, backend=LocalBackend()).state == "failed"
    config.run.attempt_id = current
    config.run.export_hf = True
    path.write_text(OmegaConf.to_yaml(config))
    result = execute_launch(path, backend=LocalBackend())
    assert result.state == "succeeded", result
    receipt = json.loads(Path(config.artifacts.terminal_manifest_uri).read_text())
    (root / "resolved-launch.yaml").write_text(OmegaConf.to_yaml({"config": receipt["config"]}))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
