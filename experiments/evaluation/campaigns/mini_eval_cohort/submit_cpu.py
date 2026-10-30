# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path

from iris.cli.connect import connect_controller
from iris.client.client import IrisClient
from iris.cluster.constraints import CLUSTER_CONSTRAINT_KEY, Constraint, ConstraintOp
from iris.cluster.types import Entrypoint, EnvironmentSpec, ResourceSpec
from iris.rpc import job_pb2
from rigging.timing import Duration

endpoint = connect_controller(cluster_name="marin")
client = IrisClient.remote(endpoint.url, workspace=Path.cwd(), credentials=endpoint.credentials)
job = client.submit(
    entrypoint=Entrypoint.from_command(
        "uv",
        "run",
        "--no-project",
        "--with",
        "numpy",
        "--with",
        "scipy",
        "--with",
        "scikit-learn",
        "--with",
        "matplotlib",
        "python",
        "experiments/evaluation/campaigns/mini_eval_cohort/cohort_automl.py",
        "analyze",
        "experiments/evaluation/campaigns/mini_eval_cohort/scores.txt",
        "@iris-output",
    ),
    name="mini-eval-cohort-20261030-cpu",
    resources=ResourceSpec(cpu=4, memory="8GB", disk="5GB"),
    environment=EnvironmentSpec(
        setup_scripts=[], secrets={key: os.environ[key] for key in ("CW_KEY_ID", "CW_KEY_SECRET")}
    ),
    constraints=[
        Constraint.create(key=CLUSTER_CONSTRAINT_KEY, op=ConstraintOp.EQ, value="cw-rno2a"),
        Constraint.create(key="device-type", op=ConstraintOp.EQ, value="cpu"),
    ],
    priority_band=job_pb2.PRIORITY_BAND_BATCH,
    timeout=Duration.from_seconds(3600),
)
print(job.job_id)
