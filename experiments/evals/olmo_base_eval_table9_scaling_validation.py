# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Launch Marin-native OLMoBaseEval Easy Table 9 BPB on scaling checkpoints.

The default mode evaluates the currently completed Delphi scaling validation
checkpoints. It includes the two checkpoints already evaluated on Stanford SC
(``proportional_3e18`` and ``dsp_effexp_table9_kl0025_3e18``), which are the
sanity rows used to compare the Marin-native evaluator against the SC oracle.

Submit from an east5-pinned Iris parent. Example:

    OLMO_EVAL_SCALING_MODE=sanity_pair \\
    MARIN_PREFIX=gs://marin-us-east5/ \\
    uv run --no-project --with-editable lib/iris --with-editable lib/rigging \\
      --with-editable lib/finelog iris --cluster=marin job run --no-wait \\
      --cpu=1 --memory=16G --disk=32G --enable-extra-resources \\
      --region us-east5 --zone us-east5-a --priority interactive \\
      --job-name dm-olmo-table9-scaling-native-20260627 \\
      -e WANDB_API_KEY $WANDB_API_KEY \\
      -e MARIN_PREFIX gs://marin-us-east5/ \\
      -e OLMO_EVAL_SCALING_MODE sanity_pair \\
      -- uv run python experiments/evals/olmo_base_eval_table9_scaling_validation.py
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from fray.cluster import ResourceConfig
from marin.evaluation.olmo_base_eval.run import olmo_base_eval_step
from marin.execution import InputName
from marin.execution.executor import executor_main

REQUEST_SET_DIR = InputName.hardcoded("raw/eval-datasets/olmo_base_eval_table9/v2")

# v6e-8 is the parity-tested production path. Pin children to the east5
# region, but not a zone: v6e capacity is not guaranteed in the v5p training
# zone. The request set and checkpoints live in the regional us-east5 bucket.
RESOURCES = ResourceConfig.with_tpu("v6e-8", regions=["us-east5"], disk="80g")


@dataclass(frozen=True)
class ScalingEvalTarget:
    name: str
    checkpoint: str
    panel: str
    scale: str
    run_name: str


def _target(name: str, checkpoint: str, *, panel: str, scale: str, run_name: str) -> ScalingEvalTarget:
    return ScalingEvalTarget(
        name=name,
        checkpoint=checkpoint,
        panel=panel,
        scale=scale,
        run_name=run_name,
    )


COMPLETED_20260627_TARGETS: tuple[ScalingEvalTarget, ...] = (
    _target(
        "t9_3e18_proportional",
        "pinlin_calvin_xu/data_mixture/delphi_baseline_mixtures_issue6607_20260623/"
        "proportional_3e18-ebc4aa/hf/step-3006",
        panel="delphi_baseline",
        scale="3e18",
        run_name="proportional_3e18",
    ),
    _target(
        "t9_2e19_proportional",
        "pinlin_calvin_xu/data_mixture/delphi_baseline_mixtures_issue6607_20260623/"
        "proportional_2e19-c5dbac/hf/step-9901",
        panel="delphi_baseline",
        scale="2e19",
        run_name="proportional_2e19",
    ),
    _target(
        "t9_3e20_proportional",
        "pinlin_calvin_xu/data_mixture/delphi_baseline_mixtures_issue6607_20260623/"
        "proportional_3e20-e5fca6/hf/step-23531",
        panel="delphi_baseline",
        scale="3e20",
        run_name="proportional_3e20",
    ),
    _target(
        "t9_3e18_unimax8",
        "pinlin_calvin_xu/data_mixture/delphi_baseline_mixtures_issue6607_20260623/"
        "unimax8_3e18-cb3b49/hf/step-3006",
        panel="delphi_baseline",
        scale="3e18",
        run_name="unimax8_3e18",
    ),
    _target(
        "t9_2e19_unimax8",
        "pinlin_calvin_xu/data_mixture/delphi_baseline_mixtures_issue6607_20260623/"
        "unimax8_2e19-b4704b/hf/step-9901",
        panel="delphi_baseline",
        scale="2e19",
        run_name="unimax8_2e19",
    ),
    _target(
        "t9_3e20_unimax8",
        "pinlin_calvin_xu/data_mixture/delphi_baseline_mixtures_issue6607_20260623/"
        "unimax8_3e20-c21bce/hf/step-23531",
        panel="delphi_baseline",
        scale="3e20",
        run_name="unimax8_3e20",
    ),
    _target(
        "t9_3e18_olmix_uncheatable",
        "pinlin_calvin_xu/data_mixture/delphi_uncheatable_optimized_mixtures_20260625/"
        "olmix_d001_kl005_cap4_3e18-c394e5/hf/step-3006",
        panel="delphi_uncheatable_optimized",
        scale="3e18",
        run_name="olmix_d001_kl005_cap4_3e18",
    ),
    _target(
        "t9_2e19_olmix_uncheatable",
        "pinlin_calvin_xu/data_mixture/delphi_uncheatable_optimized_mixtures_20260625/"
        "olmix_d001_kl005_cap4_2e19-1930db/hf/step-9901",
        panel="delphi_uncheatable_optimized",
        scale="2e19",
        run_name="olmix_d001_kl005_cap4_2e19",
    ),
    _target(
        "t9_3e20_olmix_uncheatable",
        "pinlin_calvin_xu/data_mixture/delphi_uncheatable_optimized_mixtures_20260625/"
        "olmix_d001_kl005_cap4_3e20-87fb82/hf/step-23531",
        panel="delphi_uncheatable_optimized",
        scale="3e20",
        run_name="olmix_d001_kl005_cap4_3e20",
    ),
    _target(
        "t9_3e18_dsp_uncheatable",
        "pinlin_calvin_xu/data_mixture/delphi_uncheatable_optimized_mixtures_20260625/"
        "dsp_effexp_kl01_3e18-bf29e9/hf/step-3006",
        panel="delphi_uncheatable_optimized",
        scale="3e18",
        run_name="dsp_effexp_kl01_3e18",
    ),
    _target(
        "t9_2e19_dsp_uncheatable",
        "pinlin_calvin_xu/data_mixture/delphi_uncheatable_optimized_mixtures_20260625/"
        "dsp_effexp_kl01_2e19-0e0cd6/hf/step-9901",
        panel="delphi_uncheatable_optimized",
        scale="2e19",
        run_name="dsp_effexp_kl01_2e19",
    ),
    _target(
        "t9_3e20_dsp_uncheatable",
        "pinlin_calvin_xu/data_mixture/delphi_uncheatable_optimized_mixtures_20260625/"
        "dsp_effexp_kl01_3e20-2e60f9/hf/step-23531",
        panel="delphi_uncheatable_optimized",
        scale="3e20",
        run_name="dsp_effexp_kl01_3e20",
    ),
    _target(
        "t9_3e18_olmix_table9",
        "pinlin_calvin_xu/data_mixture/delphi_table9_optimized_mixtures_20260626/"
        "olmix_table9_d001_kl005_cap4_3e18-8fb6cb/hf/step-3006",
        panel="delphi_table9_optimized",
        scale="3e18",
        run_name="olmix_table9_d001_kl005_cap4_3e18",
    ),
    _target(
        "t9_2e19_olmix_table9",
        "pinlin_calvin_xu/data_mixture/delphi_table9_optimized_mixtures_20260626/"
        "olmix_table9_d001_kl005_cap4_2e19-f3ed29/hf/step-9901",
        panel="delphi_table9_optimized",
        scale="2e19",
        run_name="olmix_table9_d001_kl005_cap4_2e19",
    ),
    _target(
        "t9_3e18_dsp_table9",
        "pinlin_calvin_xu/data_mixture/delphi_table9_optimized_mixtures_20260626/"
        "dsp_effexp_table9_kl0025_3e18-bf65d5/hf/step-3006",
        panel="delphi_table9_optimized",
        scale="3e18",
        run_name="dsp_effexp_table9_kl0025_3e18",
    ),
    _target(
        "t9_2e19_dsp_table9",
        "pinlin_calvin_xu/data_mixture/delphi_table9_optimized_mixtures_20260626/"
        "dsp_effexp_table9_kl0025_2e19-961992/hf/step-9901",
        panel="delphi_table9_optimized",
        scale="2e19",
        run_name="dsp_effexp_table9_kl0025_2e19",
    ),
)

SANITY_TARGETS = (
    target
    for target in COMPLETED_20260627_TARGETS
    if target.run_name in {"proportional_3e18", "dsp_effexp_table9_kl0025_3e18"}
)

TARGETS_BY_MODE = {
    "completed_20260627": COMPLETED_20260627_TARGETS,
    "sanity_pair": tuple(SANITY_TARGETS),
}


def _step(target: ScalingEvalTarget):
    return olmo_base_eval_step(
        name=target.name,
        checkpoint=InputName.hardcoded(target.checkpoint),
        request_set_dir=REQUEST_SET_DIR,
        resource_config=RESOURCES,
        wandb_group="olmo_base_eval_table9_scaling_validation",
        provenance={
            "evaluator": "marin-native-table9-bpb",
            "panel": target.panel,
            "scale": target.scale,
            "source_run_name": target.run_name,
        },
    )


def _build_steps():
    mode = os.environ.get("OLMO_EVAL_SCALING_MODE", "sanity_pair")
    if mode not in TARGETS_BY_MODE:
        raise ValueError(f"unknown OLMO_EVAL_SCALING_MODE={mode!r}; expected one of {sorted(TARGETS_BY_MODE)}")
    return [_step(target) for target in TARGETS_BY_MODE[mode]]


if __name__ == "__main__":
    executor_main(steps=_build_steps())
