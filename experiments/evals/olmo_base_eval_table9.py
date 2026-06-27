# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Launch the Marin-native OLMoBaseEval Easy Table 9 BPB evaluator on Iris.

Run modes (set via the ``OLMO_EVAL_RUN`` env var):
  - ``parity`` (default): the ``baseline_proportional`` 300m checkpoint, which has
    an SC oracle, for the parity canary.
  - ``canary``: the two delphi 3e18 checkpoints from the task (no SC oracle).
  - ``all``: parity + both canaries.

All checkpoints and the request set live in ``gs://marin-us-east5`` so the TPU job
reads region-locally. Submit from an east5-targeted parent, e.g.:

    uv run python experiments/evals/olmo_base_eval_table9.py \
        --executor-... (or via the cluster launcher with --region us-east5 --zone us-east5-a)
"""

from __future__ import annotations

import os

from fray.cluster import ResourceConfig
from marin.evaluation.olmo_base_eval.run import olmo_base_eval_step
from marin.execution.executor import executor_main

# Frozen Table 9 request set (51 components / 104 scored tasks), region-local to us-east5.
REQUEST_SET_DIR = "gs://marin-us-east5/raw/eval-datasets/olmo_base_eval_table9/v1"

# Parity checkpoint: has an SC oracle (per-task BPB in the 300m wide results).
PARITY_CHECKPOINT = (
    "gs://marin-us-east5/checkpoints/pinlin_calvin_xu/data_mixture/"
    "ngd3dm2_qsplit240_300m_6b/baseline_proportional-982696/hf/step-22887"
)

# Canary checkpoints from the task (no SC oracle on disk; validate the pipeline).
CANARY_CHECKPOINTS = {
    "proportional_3e18": (
        "gs://marin-us-east5/pinlin_calvin_xu/data_mixture/"
        "delphi_baseline_mixtures_issue6607_20260623/proportional_3e18-ebc4aa/hf/step-3006"
    ),
    "dsp_effexp_table9_kl0025_3e18": (
        "gs://marin-us-east5/pinlin_calvin_xu/data_mixture/"
        "delphi_table9_optimized_mixtures_20260626/dsp_effexp_table9_kl0025_3e18-bf65d5/hf/step-3006"
    ),
}

# v6e-8 single slice in us-east5; region/zone pin the nested TPU job region-locally.
RESOURCES = ResourceConfig.with_tpu("v6e-8", regions=["us-east5"], zone="us-east5-a")


def _step(name: str, checkpoint: str, request_set_dir: str = REQUEST_SET_DIR):
    return olmo_base_eval_step(
        name=name,
        checkpoint=checkpoint,
        request_set_dir=request_set_dir,
        resource_config=RESOURCES,
        wandb_group="olmo_base_eval_table9",
        provenance={
            "checkpoint": checkpoint,
            "request_set_dir": request_set_dir,
            "evaluator": "marin-native-table9-bpb",
        },
    )


def _build_steps():
    mode = os.environ.get("OLMO_EVAL_RUN", "parity")
    if mode == "parity":
        return [_step("baseline_proportional_300m_parity", PARITY_CHECKPOINT)]
    if mode == "canary":
        return [_step(name, ckpt) for name, ckpt in CANARY_CHECKPOINTS.items()]
    if mode == "all":
        return [
            _step("baseline_proportional_300m_parity", PARITY_CHECKPOINT),
            *[_step(name, ckpt) for name, ckpt in CANARY_CHECKPOINTS.items()],
        ]
    raise ValueError(f"unknown OLMO_EVAL_RUN={mode!r}; expected parity|canary|all")


if __name__ == "__main__":
    executor_main(steps=_build_steps())
