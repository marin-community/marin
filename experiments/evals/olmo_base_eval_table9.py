# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Launch the Marin-native OLMoBaseEval Easy Table 9 BPB evaluator on Iris.

Run modes (set via the ``OLMO_EVAL_RUN`` env var):
  - ``parity`` (default): the ``baseline_proportional`` 300m checkpoint, which has
    an SC oracle, for the parity canary.
  - ``canary``: the two delphi 3e18 checkpoints from the task (no SC oracle).
  - ``all``: parity + both canaries.

All artifacts live under ``gs://marin-us-east5`` and are referenced as
``InputName.hardcoded`` prefix-relative paths so reads stay region-aware; the TPU
job is pinned to us-east5 so the resolved reads are region-local. Submit via the
executor from an east5-targeted parent.
"""

from __future__ import annotations

import os

from fray.cluster import ResourceConfig
from marin.evaluation.olmo_base_eval.run import olmo_base_eval_step
from marin.execution import InputName
from marin.execution.executor import executor_main

# Frozen Table 9 request set (51 components / 104 scored tasks), prefix-relative.
REQUEST_SET_DIR = InputName.hardcoded("raw/eval-datasets/olmo_base_eval_table9/v2")

# Parity checkpoint: has an SC oracle (per-task BPB in the 300m wide results).
PARITY_CHECKPOINT = InputName.hardcoded(
    "checkpoints/pinlin_calvin_xu/data_mixture/ngd3dm2_qsplit240_300m_6b/baseline_proportional-982696/hf/step-22887"
)

# Canary checkpoints from the task (no SC oracle on disk; validate the pipeline).
CANARY_CHECKPOINTS = {
    "proportional_3e18": InputName.hardcoded(
        "pinlin_calvin_xu/data_mixture/delphi_baseline_mixtures_issue6607_20260623/proportional_3e18-ebc4aa/hf/step-3006"
    ),
    "dsp_effexp_table9_kl0025_3e18": InputName.hardcoded(
        "pinlin_calvin_xu/data_mixture/delphi_table9_optimized_mixtures_20260626/"
        "dsp_effexp_table9_kl0025_3e18-bf65d5/hf/step-3006"
    ),
}

# v6e-8 single slice; pin the region for data locality and let Iris pick the zone.
RESOURCES = ResourceConfig.with_tpu("v6e-8", regions=["us-east5"])


def _step(name: str, checkpoint: InputName):
    return olmo_base_eval_step(
        name=name,
        checkpoint=checkpoint,
        request_set_dir=REQUEST_SET_DIR,
        resource_config=RESOURCES,
        wandb_group="olmo_base_eval_table9",
        provenance={"evaluator": "marin-native-table9-bpb"},
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
