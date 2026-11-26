# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Opt-in regional proof of an unchanged Atlas verifier on LOCAL quota storage."""

import hashlib
import json
import math
import os
import subprocess
from pathlib import Path

import pyarrow.parquet as pq
import pytest
import yaml
from cloud.iris.runtime_environment import local_disk_setup_script
from huggingface_hub import get_hf_file_metadata, hf_hub_download, hf_hub_url
from iris.cluster.client.job_info import get_job_info
from marinskyrl.task_machine_factories import task_machine_factories
from omegaconf import OmegaConf
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.secrets import require_secret
from rolloutengine.engine import RolloutEngine
from rolloutengine.harness.assistant_prefix import ASSISTANT_PREFIX_HARNESS, AssistantPrefixHarness
from rolloutengine.models import TaskRolloutSpec
from rolloutengine.test_support import ReplayModel
from skyrl_gym.rollout_specs import shell_harness
from taskcompendium.grading_result import Outcome
from taskcompendium.models import TaskSpec, grades_with_judge

from experiments.post_training.long_loop.campaign_runtime import bind_runtime

pytestmark = [pytest.mark.manual, pytest.mark.asyncio]
SOURCE = "TaskTrove:laion__nemotron-gym-math-v5"
DISK_ROOT = Path("/var/lib/shellbox-disk-pool")
REPOSITORY = Path(__file__).resolve().parents[4]


@pytest.mark.timeout(600)
async def test_original_math_verifier_runs_with_declared_local_disk():
    assert get_job_info() is not None, "Run this proof inside the regional privileged CPU job"
    configure_coreweave_s3()
    subprocess.run(["bash", "-c", local_disk_setup_script(str(DISK_ROOT))], check=True)
    pins = REPOSITORY / "experiments/post_training/long_loop/campaign/source_pins.json"
    pin = json.loads(pins.read_text())["sources"][SOURCE]
    release = pin["current_release"]
    token = require_secret("HF_TOKEN")
    metadata = get_hf_file_metadata(
        hf_hub_url(release["repo"], "tasks.parquet", revision=release["revision"], repo_type="dataset"), token=token
    )
    local = Path(
        hf_hub_download(release["repo"], "tasks.parquet", revision=release["revision"], repo_type="dataset", token=token)
    )
    with local.open("rb") as stream:
        cohort_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    assert cohort_sha256 == metadata.etag
    row = next(pq.ParquetFile(local).iter_batches(batch_size=1)).to_pylist()[0]
    assert hashlib.sha256(row["task_spec"].encode()).hexdigest() == row["task_sha256"]
    task = TaskSpec.model_validate_json(row["task_spec"])
    assert not grades_with_judge(task.grader), "This proof requires a programmatic verifier"
    settings = yaml.safe_load(
        (REPOSITORY / "vendor/skyrl/skyrl-train/skyrl_train/config/task_session_config/default.yaml").read_text()
    )["task_spec"]
    spec = TaskRolloutSpec.model_validate_json(bind_runtime(row, settings)["task_rollout_spec"])
    assert spec.task_machine is None, "Choose a source without an actor machine for the serving-free proof"
    assert spec.verifier_machine is not None and spec.verifier_machine.backend == "local"
    requirements = task.verifier_execution_requirements
    assert requirements is not None and spec.verifier_machine.storage_mb is not None
    assert spec.verifier_machine.storage_mb >= requirements.storage_mb > 0
    factories = task_machine_factories(OmegaConf.create({"local": {"disk_root": str(DISK_ROOT)}}))
    harness = AssistantPrefixHarness() if spec.harness == ASSISTANT_PREFIX_HARNESS else shell_harness(spec.harness)
    model = ReplayModel([{"role": "assistant", "content": "0"}])
    result = await RolloutEngine(factories, {spec.harness: harness}).run_task(spec, model.complete)
    assert result.grade.status is Outcome.GRADED, result.grade
    assert result.grade.reward is not None and math.isfinite(result.grade.reward)
    assert not list(DISK_ROOT.glob("sandbox-*"))
    propagation = subprocess.check_output(
        ["findmnt", "-n", "-o", "PROPAGATION", "--target", str(DISK_ROOT)], text=True
    ).strip()
    assert "shared" not in propagation
    root_stat = DISK_ROOT.stat()
    assert root_stat.st_uid == 0 and root_stat.st_mode & 0o077 == 0
    output = Path(os.environ["IRIS_OUTPUT_DIR"])
    output.mkdir(parents=True, exist_ok=True)
    (output / "affected-source-local-disk.json").write_text(
        json.dumps(
            {
                "source": SOURCE,
                "release": release,
                "cohort_sha256": cohort_sha256,
                "task_id": task.id,
                "task_sha256": row["task_sha256"],
                "storage_mb": spec.verifier_machine.storage_mb,
                "grader_status": result.grade.status.value,
                "control_reward": result.grade.reward,
                "model": "fixed answer 0; no inference or training",
                "disk_root": str(DISK_ROOT),
                "mount_propagation": propagation,
                "root_uid": root_stat.st_uid,
                "root_mode": oct(root_stat.st_mode & 0o777),
                "pool_probe": "loop/ext4/overlay/ENOSPC/cleanup passed at construction",
                "remaining_owned_allocations": [],
                "task_spec_bytes_unchanged": True,
            },
            indent=2,
        )
        + "\n"
    )
