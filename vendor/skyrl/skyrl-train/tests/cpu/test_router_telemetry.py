# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Router telemetry: load statistics, collision mass, and the frozen-bias guard.

The fixtures use Megatron's real parameter-name shapes — grouped-GEMM
``linear_fc1.weight<i>`` for Grug's TEGroupedMLP experts — so a rename in mcore
breaks these tests instead of silently emptying the telemetry.
"""

import json

import pytest
import torch
from omegaconf import OmegaConf
from torch import nn

from skyrl_train.config.utils import get_default_config
from skyrl_train.models.router_instrumentation import RouterObservation
from skyrl_train.workers.megatron.router_telemetry import (
    RouterCollisionTracker,
    RouterLoadStats,
    assert_bias_unchanged,
    collision_norms,
    collision_summary_metrics,
    layer_index,
    local_expert_index,
    router_telemetry_enabled,
    router_telemetry_record,
    routed_expert_parameters,
    should_count_router_forward,
    snapshot_expert_bias,
    write_router_telemetry_record,
)


class _FakeRouter(nn.Module):
    def __init__(self, num_experts: int, layer_number: int | None):
        super().__init__()
        self.layer_number = layer_number
        self.num_experts = num_experts

    def observation(self, selections, row_mask=None) -> RouterObservation:
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


def test_local_expert_index_parses_grouped_gemm_and_sequential_names():
    assert local_expert_index("decoder.layers.mlp.experts.linear_fc1.weight3") == 3
    assert local_expert_index("decoder.layers.mlp.experts.linear_fc2.weight11") == 11
    assert local_expert_index("mlp.experts.expert2.gate_proj.weight") == 2
    assert local_expert_index("mlp.experts.local_experts.1.gate_proj.weight") == 1
    assert local_expert_index("decoder.layers.mlp.router.weight") is None
    with pytest.raises(ValueError, match="no expert index"):
        # A parameter under mlp.experts without an index must raise at the
        # parameter-walk layer, not silently skip.
        routed_expert_parameters(_UnnamedExpertsModel())


class _Layer(nn.Module):
    """Shaped like mcore's TransformerLayer: layer_number plus an mlp.experts module.

    Grouped-GEMM experts nest a ``linear_fc1`` module whose fused weights carry
    the index suffix, so ``named_parameters()`` yields ``linear_fc1.weight<i>``
    without a dot inside a single name.
    """

    def __init__(self, layer_number: int, num_experts: int):
        super().__init__()
        self.layer_number = layer_number
        self.mlp = nn.Module()
        self.mlp.experts = nn.Module()
        self.mlp.experts.linear_fc1 = nn.Module()
        for expert in range(num_experts):
            self.mlp.experts.linear_fc1.register_parameter(f"weight{expert}", nn.Parameter(torch.zeros(4, 4)))


class _UnnamedExpertsModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_number = 1
        self.mlp = nn.Module()
        self.mlp.experts = nn.Module()
        self.mlp.experts.grouped = nn.Parameter(torch.zeros(4, 4))


def test_routed_expert_parameters_walk_layers_and_reject_unknown_names():
    model = nn.ModuleList([_Layer(1, 2), _Layer(2, 2)])
    rows = routed_expert_parameters(model)
    assert [(row.layer, row.local_expert) for row in rows] == [(0, 0), (0, 1), (1, 0), (1, 1)]
    with pytest.raises(ValueError, match="no routed-expert parameters"):
        routed_expert_parameters(nn.Linear(4, 4))


def test_layer_index_uses_the_layer_map_and_rejects_unknown_routers():
    router = _FakeRouter(4, 7)
    layer_map = {id(router): 6}
    assert layer_index(router, layer_map) == 6
    with pytest.raises(ValueError, match="absent from the layer map"):
        layer_index(_FakeRouter(4, 9), layer_map)


def test_only_original_forwards_are_counted():
    # Without checkpointing the original forward runs with grad; with
    # checkpointing the original runs under no_grad and the recompute has grad.
    assert should_count_router_forward(grad_enabled=True, gradient_checkpointing=False) is True
    assert should_count_router_forward(grad_enabled=False, gradient_checkpointing=True) is True
    assert should_count_router_forward(grad_enabled=True, gradient_checkpointing=True) is False
    assert should_count_router_forward(grad_enabled=False, gradient_checkpointing=False) is False


def test_load_stats_mask_to_response_tokens_and_summarize():
    router = _FakeRouter(4, 3)
    layer_map = {id(router): 2}
    stats = RouterLoadStats(layer_map)
    # Four rows; the response mask keeps rows 0 and 2 (prompt and pad rows drop).
    stats.observe(router.observation([[0], [1], [1], [0]]), row_mask=torch.tensor([True, False, True, False]))
    counts = stats.dense_counts()
    assert counts[0].tolist() == [1.0, 1.0, 0.0, 0.0]
    metrics = stats.metrics(counts)
    assert metrics["router/load/layer2/entropy"] == pytest.approx(0.693147, abs=1e-5)
    assert metrics["router/load/layer2/max_over_mean"] == pytest.approx(2.0)
    assert metrics["router/load/layer2/experts_below_10pct_of_mean"] == 2.0


def test_load_stats_reject_mismatched_masks_and_disagreeing_expert_counts():
    router = _FakeRouter(4, 1)
    stats = RouterLoadStats({id(router): 0})
    with pytest.raises(ValueError, match="response mask"):
        stats.observe(router.observation([[0], [1]]), row_mask=torch.tensor([True]))
    other = _FakeRouter(6, 1)
    stats.layer_map[id(other)] = 0
    stats.observe(other.observation([[0]]))
    with pytest.raises(ValueError, match="expert count"):
        stats.observe(_FakeRouter(4, 1).observation([[0]]))


class _ExpertsModel(nn.Module):
    def __init__(self, num_experts: int = 2):
        super().__init__()
        self.layer_number = 1
        self.mlp = nn.Module()
        self.mlp.experts = nn.Module()
        self.mlp.experts.linear_fc1 = nn.Module()
        self.parameters_by_expert = []
        for expert in range(num_experts):
            parameter = nn.Parameter(torch.zeros(4, 4))
            self.mlp.experts.linear_fc1.register_parameter(f"weight{expert}", parameter)
            self.parameters_by_expert.append(parameter)


def _tracker_with_main_grads(model: _ExpertsModel, grads: dict[int, float]) -> RouterCollisionTracker:
    tracker = RouterCollisionTracker(model, expert_offset=4, num_global_experts=8)
    for expert, norm_squared in grads.items():
        model.parameters_by_expert[expert].main_grad = torch.full((4, 4), norm_squared**0.5 / 2.0)
    return tracker


def test_collision_tracker_splits_waves_by_task_and_validates_the_schedule():
    model = _ExpertsModel()
    tracker = RouterCollisionTracker(model, expert_offset=4, num_global_experts=8)
    # Micro-batch 1: pure task_X. Micro-batch 2: a mixed θ_XR batch.
    tracker.begin_micro_batch({"task_X": 8})
    for expert, gradient_sum in ((0, 0.5), (1, 0.25)):
        model.parameters_by_expert[expert].main_grad = torch.full((4, 4), gradient_sum**0.5 / 4.0)
        tracker._on_gradient_accumulated(model.parameters_by_expert[expert])
    tracker.begin_micro_batch({"task_X": 4, "task_R": 4})
    # Wave 2 deltas: expert 0's accumulated ||grad||^2 grows 0.5 -> 1.0 (delta
    # 0.5); expert 1's stays at 0.25 (delta 0).
    model.parameters_by_expert[0].main_grad = torch.full((4, 4), 0.25)
    tracker._on_gradient_accumulated(model.parameters_by_expert[0])
    model.parameters_by_expert[1].main_grad = torch.full((4, 4), 0.125)
    tracker._on_gradient_accumulated(model.parameters_by_expert[1])
    tracker.finish(num_micro_batches=2)

    task_sos = tracker.scaled_task_sos(1.0)
    # Wave 1 attributed 0.5 to expert 0 and 0.25 to expert 1, all task_X; wave
    # 2's 0.5/0.0 split evenly across the 4/4 token mix on top.
    assert task_sos[("task_X", 0, 4)] == pytest.approx(0.5 + 0.25)
    assert task_sos[("task_X", 0, 5)] == pytest.approx(0.25)
    assert task_sos[("task_R", 0, 4)] == pytest.approx(0.25)
    assert ("task_R", 0, 5) not in task_sos

    with pytest.raises(ValueError, match="disagree"):
        tracker.finish(num_micro_batches=3)


def test_collision_tracker_requires_queued_mixes_and_response_tokens():
    model = _ExpertsModel()
    tracker = RouterCollisionTracker(model, expert_offset=0, num_global_experts=2)
    model.parameters_by_expert[0].main_grad = torch.ones(4, 4)
    model.parameters_by_expert[1].main_grad = torch.ones(4, 4)
    # The first wave completes on the second expert's hook and finds no mix.
    tracker._on_gradient_accumulated(model.parameters_by_expert[0])
    with pytest.raises(ValueError, match="no queued micro-batch"):
        tracker._on_gradient_accumulated(model.parameters_by_expert[1])
    tracker2 = RouterCollisionTracker(model, expert_offset=0, num_global_experts=2)
    tracker2.begin_micro_batch({"task_X": 0})
    model.parameters_by_expert[0].main_grad = torch.ones(4, 4)
    model.parameters_by_expert[1].main_grad = torch.ones(4, 4)
    # The wave completes only when every hooked expert fired; the flush then
    # finds a task mix with no response tokens and refuses to split by it.
    tracker2._on_gradient_accumulated(model.parameters_by_expert[0])
    with pytest.raises(ValueError, match="no response tokens"):
        tracker2._on_gradient_accumulated(model.parameters_by_expert[1])


def test_collision_tracker_rejects_an_offset_outside_the_global_experts():
    with pytest.raises(ValueError, match="outside"):
        RouterCollisionTracker(_ExpertsModel(), expert_offset=7, num_global_experts=8)


def test_collision_summaries_and_sidecar_vectors():
    task_sos = {
        ("task_X", 0, 0): 4.0,
        ("task_X", 0, 1): 1.0,
        ("task_R", 1, 0): 9.0,
    }
    metrics = collision_summary_metrics(task_sos)
    assert metrics["router/collision/task_X/layer0/mass"] == pytest.approx(5.0**0.5)
    assert metrics["router/collision/task_X/layer0/max_expert_share"] == pytest.approx(2.0 / 5.0**0.5)
    assert metrics["router/collision/task_R/layer1/mass"] == pytest.approx(3.0)
    norms = collision_norms(task_sos, num_experts=4)
    assert norms["task_X"]["layer0"] == [2.0, 1.0, 0.0, 0.0]
    assert norms["task_R"]["layer1"] == [3.0, 0.0, 0.0, 0.0]


def test_sidecar_record_writes_per_layer_expert_vectors(tmp_path):
    record = router_telemetry_record(
        step=17,
        layers=[0, 1],
        load_counts=torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64),
        norms={"task_X": {"layer0": [2.0, 1.0]}},
    )
    assert record["load_counts"]["layer0"] == [1.0, 2.0]
    write_router_telemetry_record(str(tmp_path), record)
    lines = (tmp_path / "router_telemetry.jsonl").read_text().splitlines()
    assert json.loads(lines[0])["step"] == 17
    assert json.loads(lines[0])["collision_norms"]["task_X"]["layer0"] == [2.0, 1.0]


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
