# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Paths to the 10 Delphi scaling-ladder checkpoints (the parent ``/hf`` dirs).

Same MARIN_PREFIX-relative paths Will hardcodes in
``origin/will/delphi-evals:experiments/exp1337_eval_suite.py`` (``DELPHI_SWEEP_WINNERS``,
``DELPHI_OPTIMAL_RUNS``). The actual ``step-N`` snapshot under each ``/hf`` is
resolved at rollout/grade time via ``marin.evaluation.utils.discover_hf_checkpoints``
— same call lm-eval-harness's evaluator makes when ``discover_latest_checkpoint=True``
(see ``marin/evaluation/run.py:120-121``).

Sources:
- 7 IsoFLOP sweep winners come from
  ``MARIN_SCALING_SUITES["nemotron-completed-adamh"]`` in
  ``experiments/isoflop_sweep.py`` (built by
  ``experiments/scaling_law_sweeps/completed_adamh.py``).
- 3 compute-optimal target-budget runs come from
  ``experiments/exp1337_delphi_suite.py``. Their 6-char path suffixes
  (``019021/025b0e/27f2fb``) are content hashes of the training step config.
"""

DELPHI_CHECKPOINTS: dict[str, str] = {
    # IsoFLOP sweep winners (label = compute budget)
    "3e18": "checkpoints/isoflop/isoflop-3e+18-d1024-L11-B8-adamh_scaling_v6/hf",
    "9e18": "checkpoints/isoflop/isoflop-9e+18-d1152-L12-B16-adamh_scaling_v6/hf",
    "2e19": "checkpoints/isoflop/isoflop-2e+19-d1408-L15-B16-adamh_scaling_v6/hf",
    "3e19": "checkpoints/isoflop/isoflop-3e+19-d1536-L16-B32-adamh_scaling_v6/hf",
    "9e19": "checkpoints/isoflop/isoflop-9e+19-d1792-L18-B64-adamh_scaling_v6/hf",
    "2e20": "checkpoints/isoflop/isoflop-2e+20-d2048-L21-B64-adamh_scaling_v6/hf",
    "3e20": "checkpoints/isoflop/isoflop-3e+20-d2304-L23-B128-adamh_scaling_v6/hf",
    # Compute-optimal target-budget runs
    "1e21": "adamh-scaling-ladder-nemotron-optimal-1e+21-v5-019021/hf",
    "1e22": "adamh-scaling-ladder-nemotron-optimal-1e+22-v5-025b0e/hf",
    "1e23": "adamh-scaling-ladder-nemotron-optimal-1e+23-v5-27f2fb/hf",
}
