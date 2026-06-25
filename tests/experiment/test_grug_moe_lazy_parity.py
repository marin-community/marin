# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The lazy-artifact grug-moe baseline must materialize to the same config as the
``ExecutorStep`` version (``baseline_moe``).

This pins the redesign on a real, active experiment: the new authoring model
(a ``Checkpoint`` handle with ``output_path=ctx.out`` and inline decisions) must
resolve to the identical ``GrugMoeLaunchConfig`` the executor produces today —
only the step's own output path is allowed to change (content hash → ``name@version``).
"""

import json

from marin.execution.executor import Executor, executor_context
from marin.execution.lazy import to_executor_step
from marin.execution.types import ExecutorStep
from marin.utilities.json_encoder import CustomJsonEncoder

from experiments.grug.moe.launch import baseline_moe
from experiments.grug.moe.launch_lazy import grug_moe_baseline


def _materialized(step: ExecutorStep, prefix: str) -> str:
    """Resolve a step's config through the executor and return it as JSON with the
    step's own output path normalized to ``<SELF>`` (so the content-hash scheme and
    the explicit name@version scheme compare equal)."""
    executor = Executor(prefix=prefix, executor_info_base_path=f"{prefix}/experiments")
    executor.compute_version(step, is_pseudo_dep=False)
    resolved = json.dumps(executor.configs[step], sort_keys=True, cls=CustomJsonEncoder)
    return resolved.replace(executor.output_paths[step], "<SELF>")


def test_grug_moe_lazy_matches_baseline_moe(monkeypatch):
    # run_id resolution reads these; clear so both sides resolve identically.
    monkeypatch.delenv("GRUG_RUN_ID", raising=False)
    monkeypatch.delenv("FERRY_DATE", raising=False)

    prefix = "gs://marin-golden"
    with executor_context():
        lazy_step = to_executor_step(grug_moe_baseline(), prefix)

    old = _materialized(baseline_moe, prefix)
    new = _materialized(lazy_step, prefix)
    assert old == new, f"\n old: {old}\n new: {new}"
