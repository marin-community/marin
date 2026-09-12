# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""A real loss-input counterexample for the missing frozen-runtime M2 mechanism."""

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf
from skyrl_train.utils.offpolicy_masks import apply_offpolicy_masks

LEGACY = Path(
    "/home/ahmad/oa/worktrees/MarinSkyRL/async-non-agentic-rl-research-k12-integration/skyrl-train/skyrl_train/utils/offpolicy_masks.py"
)
spec = importlib.util.spec_from_file_location("qualified_m2_operand", LEGACY)
legacy = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = legacy
spec.loader.exec_module(legacy)

config = OmegaConf.create(
    dict(
        offpolicy_mask=dict(enabled=False),
        m2_mask=dict(enabled=True, ratio="stale", tau=0.04, mode="mask", renormalize=False),
    )
)


def gradient(implementation, values):
    action = torch.tensor([values], dtype=torch.float64, requires_grad=True)
    result = implementation(
        action_log_probs=action,
        old_action_log_probs=torch.zeros_like(action),
        rollout_logprobs=torch.zeros_like(action),
        advantages=torch.ones_like(action),
        loss_mask=torch.ones_like(action),
        token_entropy=torch.ones_like(action),
        config=config,
    )
    (action * result.advantages).sum().backward()
    return action.grad.tolist()[0]


expected = [0.0, 1.0, 1.0]  # Remove delta=1; retained squared-delta mean is 0.005 < 0.04.
qualified = gradient(legacy.apply_offpolicy_masks, [1.0, 0.1, 0.0])
frozen = gradient(apply_offpolicy_masks, [1.0, 0.1, 0.0])
assert qualified == expected and frozen == [1.0, 1.0, 1.0]
assert gradient(legacy.apply_offpolicy_masks, [0.0, 0.0, 0.0]) == [1.0, 1.0, 1.0]
original = legacy.minimal_m2_mask


def mutated(delta, advantages, selected, tau):
    removed, candidates, before, after, unsatisfied = original(delta, advantages, selected, tau)
    return torch.zeros_like(removed), candidates, before, after, unsatisfied


legacy.minimal_m2_mask = mutated
assert gradient(legacy.apply_offpolicy_masks, [1.0, 0.1, 0.0]) != expected
legacy.minimal_m2_mask = original
report = dict(
    status="E43_M2_MISSING_RUNTIME_COUNTEREXAMPLE_PASS",
    expected_gradient=expected,
    qualified_gradient=qualified,
    frozen_gradient=frozen,
    zero_delta_control=True,
    no_op_mutation_rejected=True,
    qualified_source_sha256=hashlib.sha256(LEGACY.read_bytes()).hexdigest(),
    scope="CPU loss-input transform and its differentiable effect; no native or quality gate",
)
Path("/home/ahmad/.cache/oa/async-v2-e43-m2-counterexample-v1.json").write_text(json.dumps(report, indent=2) + "\n")
print(report["status"] + " frozen_no_op=true qualified_transform=true mutation_rejected=true")
