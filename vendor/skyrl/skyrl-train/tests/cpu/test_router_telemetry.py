"""Router telemetry: load statistics, collision mass, and the frozen-bias guard."""

import pytest
import torch
from omegaconf import OmegaConf
from torch import nn

from skyrl_train.config.utils import get_default_config
from skyrl_train.models.router_instrumentation import RouterObservation
from skyrl_train.workers.megatron.router_telemetry import (
    RouterLoadStats,
    assert_bias_unchanged,
    collision_metrics,
    layer_index,
    router_policy_manifest,
    router_telemetry_enabled,
    routed_expert_gradient_norms,
    snapshot_expert_bias,
)


class _FakeRouter(nn.Module):
    def __init__(self, num_experts: int, layer_number: int | None):
        super().__init__()
        self.layer_number = layer_number
        self.num_experts = num_experts

    def observation(self, selections) -> RouterObservation:
        selections = torch.tensor(selections)
        logits = torch.zeros(selections.shape[0], self.num_experts)
        return RouterObservation(
            router=self,
            router_inputs=logits,
            selection_logits=logits,
            selection_log_probs=logits,
            natural_selected_experts=selections,
            selected_experts=selections,
            combine_weights=torch.ones_like(selections, dtype=torch.float),
        )


def _stats(selections_by_layer) -> RouterLoadStats:
    stats = RouterLoadStats()
    for (layer_number, num_experts), selections in selections_by_layer:
        stats.observe(_FakeRouter(num_experts, layer_number).observation(selections))
    return stats


def test_load_stats_recompute_load_entropy_and_max_over_mean():
    # Two tokens route to experts 0 and 1 in a 4-expert layer: half the experts idle.
    stats = _stats([((3, 4), [[0], [1]])])
    metrics = stats.metrics()
    assert metrics["router/load/layer2/entropy"] == pytest.approx(0.693147, abs=1e-5)
    assert metrics["router/load/layer2/max_over_mean"] == pytest.approx(2.0)
    assert metrics["router/load/layer2/experts_below_10pct_of_mean"] == 2.0


def test_load_stats_accumulate_across_forwards_and_merge_across_ranks():
    first = _stats([((1, 2), [[0], [0]])])
    second = _stats([((1, 2), [[1]])])
    merged = RouterLoadStats()
    merged.merge_counts([first.counts(), second.counts()])
    # Three tokens: experts 0 and 1 carry 2/3 and 1/3 of the load.
    assert merged.metrics()["router/load/layer0/experts_below_10pct_of_mean"] == 0.0
    assert merged.metrics()["router/load/layer0/max_over_mean"] == pytest.approx(4.0 / 3.0)


def test_layer_index_is_one_based_mcore_layer_numbers():
    assert layer_index(_FakeRouter(4, 7)) == 6
    assert layer_index(_FakeRouter(4, None)) is None


def test_collision_mass_attributes_a_single_tag_and_mixes_unknown_ones():
    norms = {0: 2.0, 3: 1.0}
    single = collision_metrics(norms, ["task_x", "task_x"])
    assert single["router/collision/task_x/expert0"] == 2.0
    assert single["router/collision/task_x/expert3"] == 1.0
    mixed = collision_metrics(norms, ["task_x", "task_m"])
    assert mixed["router/collision/mixed/expert0"] == 2.0
    assert mixed["router/collision/task_share/task_x"] == 0.5
    assert collision_metrics({}, ["task_x"]) == {}


def test_routed_expert_gradient_norms_recompute_from_parameter_gradients():
    class Experts(nn.Module):
        def __init__(self):
            super().__init__()
            self.expert0 = nn.Linear(4, 4, bias=False)
            self.expert1 = nn.Linear(4, 4, bias=False)

    experts = Experts()
    other = nn.Linear(4, 4, bias=False)
    model = nn.ModuleDict({"mlp": nn.ModuleDict({"experts": experts}), "other": other})
    with torch.no_grad():
        experts.expert0.weight.grad = torch.full((4, 4), 0.5)
        experts.expert1.weight.grad = torch.full((4, 4), 1.0)
        other.weight.grad = torch.full((4, 4), 9.0)
    norms = routed_expert_gradient_norms(model)
    assert norms[0] == pytest.approx(2.0)  # sqrt(16 * 0.5^2)
    assert norms[1] == pytest.approx(4.0)
    assert len(norms) == 2


def test_routed_expert_gradients_require_an_expert_index():
    class Experts(nn.Module):
        def __init__(self):
            super().__init__()
            self.grouped = nn.Linear(4, 4, bias=False)

    model = nn.ModuleDict({"mlp": nn.ModuleDict({"experts": Experts()})})
    with pytest.raises(ValueError, match="no expert index"):
        routed_expert_gradient_norms(model)


def test_bias_guard_accepts_identical_buffers_and_rejects_drift():
    class Router(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("expert_bias", torch.zeros(4))

    model = Router()
    snapshot = snapshot_expert_bias(model)
    assert_bias_unchanged(model, snapshot)
    with torch.no_grad():
        model.expert_bias += 1
    with pytest.raises(ValueError, match="must stay frozen"):
        assert_bias_unchanged(model, snapshot)


def test_telemetry_flag_defaults_off_and_reads_the_config():
    cfg = get_default_config()
    assert router_telemetry_enabled(cfg) is False
    enabled = OmegaConf.merge(cfg, OmegaConf.create({"trainer": {"router_telemetry": {"enabled": True}}}))
    assert router_telemetry_enabled(enabled) is True


def test_manifest_records_the_r1_policy():
    assert router_policy_manifest("grugmoe") == {
        "model_type": "grugmoe",
        "router_weights": "trained",
        "selection_bias": "frozen",
        "aux_loss_coeff": 0.0,
    }
