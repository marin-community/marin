import gzip
import json
from dataclasses import replace
import zipfile

from skyrl_train.task_inputs import TrajectoryMetadata
from skyrl_train.trajectory_runners.trajectory_retention import TrajectorySink, execute_publication
from skyrl_train.trajectory_runners.trajectory_retention_config import parse_trajectory_retention_config
from skyrl_train.trajectory_runners.trajectory_retention_publisher import InlineTrajectoryPublisher
from skyrl_train.trajectory_runners.types import BatchMetadata, PreferenceRequestBatch, TrajectoryID


def test_eval_only_retains_every_record_without_an_optimizer_step_or_byte_cap(tmp_path):
    config = parse_trajectory_retention_config({
        "enabled": True, "required": True, "output_path": str(tmp_path), "run_id": "micro210-test",
        "phases": ["train", "eval"], "sample_fraction": 0, "eval_sample_fraction": 1,
        "sample_count_per_step": 0, "always_retain_failures": False,
        "always_retain_non_terminating": False, "always_retain_loops": False,
        "max_bytes_per_step": None, "max_bytes_per_run": None, "model_checkpoint_step": 20,
    })
    request = PreferenceRequestBatch(
        prompts=([{"role": "user", "content": "question"}],), preferences=({},),
        metadata=(TrajectoryMetadata("task", "test"),), trajectory_ids=(TrajectoryID("task", 0),),
        batch_metadata=BatchMetadata(None, "eval"),
    )
    output = {"prompt_token_ids": [[1]], "response_ids": [[2]], "loss_masks": [[1]],
              "rewards": [1.0], "stop_reasons": ["stop"]}
    sink = TrajectorySink(config, None, InlineTrajectoryPublisher(execute_publication))
    sink.bind_runner("test")
    sink.retain(request, output)
    sink.retain(replace(request, batch_metadata=BatchMetadata(1, "train")), output)
    sink.close()
    archives = list(tmp_path.rglob("*.zip"))
    assert len(archives) == 1
    assert "step=eval_only" in str(archives[0])
    with zipfile.ZipFile(archives[0]) as archive:
        records = [json.loads(gzip.decompress(archive.read(name))) for name in archive.namelist() if name.endswith(".json.gz")]
    assert len(records) == 1
    assert records[0]["global_step"] is None
    assert records[0]["provenance"]["model_version_step"] == 20
    assert records[0]["response"]["token_ids"] == [2]
    (tmp_path / "_retention_ledger.json").unlink()
    recovered = TrajectorySink(config, None, InlineTrajectoryPublisher(execute_publication))
    recovered.bind_runner("test")
    recovered.retain(replace(request, trajectory_ids=(TrajectoryID("task", 1),)), output)
    recovered.close()
    assert len(json.loads((tmp_path / "_retention_ledger.json").read_text())["records"]) == 2
