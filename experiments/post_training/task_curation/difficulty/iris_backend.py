# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Submit child jobs through the reconciler's inherited Iris context."""

import json
import re
from dataclasses import dataclass

from finestore.rl.difficulty import Solver
from iris.client.client import IrisClient
from iris.cluster.types import Entrypoint, EnvironmentSpec, JobName, ResourceSpec
from iris.resources.state import JobState
from iris.rpc import job_pb2
from rigging.filesystem.storage_path import StoragePath
from rigging.timing import Duration

from experiments.post_training.task_curation.difficulty.campaign import campaign_record
from experiments.post_training.task_curation.difficulty.executor import Job, action_job_slots
from experiments.post_training.task_curation.difficulty.launch import DAYTONA_API_URL, DAYTONA_TARGET, JOB_TIMEOUT
from experiments.post_training.task_curation.difficulty.ledger import ACTIONS_DIRECTORY, LEDGER_VERSION
from experiments.post_training.task_curation.difficulty.planner import (
    MAX_GPU_HOURS_PER_JOB,
    Action,
    AttemptBatch,
    QualifyJudgeRoute,
    RecheckExecutabilityAdmission,
    action_from_payload,
    action_id,
)
from experiments.post_training.task_curation.difficulty_protocol import PINS

JOB_COMPONENT = "difficulty-reconcile-"
MAX_JUDGE_GPUS = 8
JOB_NAMESPACE = "atlas-difficulty"
ADMISSION_RECHECK_TIMEOUT = 3600


@dataclass(frozen=True)
class IrisBackend:
    client: IrisClient
    controller_job: JobName
    ledger: str
    serving_pool: str | None = None

    def jobs(self) -> list[Job]:
        prefix = f"/{self.controller_job.user}/{JOB_NAMESPACE}"
        raw = self.client.list_jobs(prefix=prefix)
        jobs = []
        grouped: dict[str, list[Job]] = {}
        for status in raw:
            wire = str(status.job_id)
            if "atlas-difficulty-engines-" in wire or (
                "atlas-difficulty-reconciler-controller-" in wire and "/" + JOB_COMPONENT not in wire
            ):
                continue
            match = re.search(r"atlas-difficulty-(small|large|hosted|snowball)-", wire)
            solver = Solver(match.group(1)) if match else None
            job = Job(wire, status.state, solver)
            if "/" + JOB_COMPONENT in wire and job.active:
                base, component = wire.split("/" + JOB_COMPONENT, 1)
                group = base + "/" + JOB_COMPONENT + component.split("/", 1)[0]
                grouped.setdefault(group, []).append(job)
            else:
                jobs.append(job)
        for root in grouped:
            prefix = root.rsplit("/", 1)[-1].removeprefix(JOB_COMPONENT)
            matches = (StoragePath(self.ledger) / LEDGER_VERSION / ACTIONS_DIRECTORY / (prefix + "*.json")).glob()
            if len(matches) != 1:
                raise ValueError(f"Active job has no unique action manifest: {root}")
            action = action_from_payload(json.loads(matches[0].read_text()))
            jobs.append(
                Job(
                    root,
                    JobState.RUNNING,
                    action.solver if isinstance(action, AttemptBatch) else None,
                    action_job_slots(action),
                )
            )
        return jobs

    def submit(self, action: Action, manifest_uri: str) -> str:
        key = action_id(action)
        name = JOB_COMPONENT + key[:24]
        existing = self.client.list_jobs(prefix=f"/{self.controller_job.user}/{JOB_NAMESPACE}")
        for job in existing:
            if str(job.job_id).endswith("/" + name):
                return str(job.job_id)
        timeout = int(JOB_TIMEOUT.to_seconds())
        if isinstance(action, AttemptBatch) and self.serving_pool is None:
            # Include the largest locally served judge in the reservation bound.
            gpu_count = PINS[action.solver].gpus + MAX_JUDGE_GPUS
            timeout = int(action.max_gpu_hours * 3600 / gpu_count)
        elif isinstance(action, QualifyJudgeRoute):
            timeout = int(MAX_GPU_HOURS_PER_JOB * 3600 / MAX_JUDGE_GPUS)
        elif isinstance(action, RecheckExecutabilityAdmission):
            timeout = ADMISSION_RECHECK_TIMEOUT
        command = [
            "python",
            "-m",
            "experiments.post_training.task_curation.difficulty.run_action",
            "--manifest",
            manifest_uri,
            "--ledger",
            self.ledger,
        ]
        if self.serving_pool is not None:
            command.extend(["--serving-pool", self.serving_pool])
        job = self.client.submit(
            entrypoint=Entrypoint.from_command(*command),
            name=name,
            resources=(
                ResourceSpec(cpu=4, memory="12GB", disk="64GB")
                if campaign_record(self.ledger) is not None
                else (
                    ResourceSpec(cpu=16, memory="64GB", disk="256GB")
                    if isinstance(action, AttemptBatch)
                    else ResourceSpec(cpu=8, memory="32GB", disk="128GB")
                )
            ),
            environment=EnvironmentSpec(env_vars={"DAYTONA_API_URL": DAYTONA_API_URL, "DAYTONA_TARGET": DAYTONA_TARGET}),
            priority_band=job_pb2.PRIORITY_BAND_INTERACTIVE,
            container_profile=job_pb2.CONTAINER_PROFILE_PRIVILEGED,
            egress_policy=job_pb2.EGRESS_POLICY_CLUSTER,
            timeout=Duration.from_seconds(timeout),
            max_retries_failure=0,
            max_retries_preemption=0,
        )
        return str(job.job_id)
