"""Grug log scaling: per-document positions under packing and CP, and the checkpoint contract."""

import math

import pytest
import torch

from skyrl_train.models.grug_log_scaling import grug_log_scaling_factor, thd_document_positions
from skyrl_train.models.grug_moe import GrugMoeConfig
from tests.cpu.util import stub_megatron_modules

HERO_LOG_SCALING = dict(log_scaling_n_floor=65536, log_scaling_alpha=0.1, log_scaling_power=2.0)


@pytest.mark.parametrize("cp_size", [1, 2, 4])
def test_positions_restart_per_document_on_every_cp_rank(monkeypatch, cp_size):
    stub_megatron_modules()
    from skyrl_train.distributed.megatron import megatron_utils

    # Rows are documents with left and right padding; each valid token holds its position in the document.
    attention_mask = torch.tensor(
        [[0, 0, 1, 1, 1, 1, 1, 0], [1, 1, 1, 0, 0, 0, 0, 0], [0, 1, 1, 1, 1, 1, 1, 1]], dtype=torch.bool
    )
    positions = (attention_mask.long().cumsum(-1) - 1).clamp(min=0)
    monkeypatch.setattr(megatron_utils.mpu, "get_tensor_model_parallel_world_size", lambda: 1, raising=False)
    monkeypatch.setattr(megatron_utils.mpu, "get_context_parallel_world_size", lambda: cp_size, raising=False)
    for rank in range(cp_size):
        monkeypatch.setattr(megatron_utils.mpu, "get_context_parallel_rank", lambda: rank, raising=False)
        packed_positions, params = megatron_utils.preprocess_packed_seqs(positions, attention_mask)
        packed_valid, _ = megatron_utils.preprocess_packed_seqs(attention_mask.long(), attention_mask)
        valid = packed_valid[0].bool()

        actual = thd_document_positions(params.cu_seqlens_q_padded, packed_positions.shape[1], cp_size, rank)

        assert valid.any()
        assert actual[valid].tolist() == packed_positions[0, valid].tolist()


def test_factor_is_one_through_the_floor_and_matches_the_hero_at_262k():
    factor = grug_log_scaling_factor(torch.tensor([1, 65536, 65537, 262144]), 65536, 0.1, 2.0)

    assert factor[:2].tolist() == [1.0, 1.0]
    assert factor[2].item() > 1.0
    assert factor[3].item() == pytest.approx((1 + 0.1 * math.log(4)) ** 2, rel=1e-6)


@pytest.mark.parametrize(
    ("log_scaling", "expected"),
    [
        (HERO_LOG_SCALING, (65536, 0.1, 2.0)),
        (dict.fromkeys(HERO_LOG_SCALING), (None, None, None)),
    ],
    ids=["on", "null"],
)
def test_schema_3_config_json_keeps_log_scaling_and_split_experts(log_scaling, expected):
    config = GrugMoeConfig(grugmoe_artifact_schema_version=3, **log_scaling)

    reloaded = GrugMoeConfig.from_dict(config.to_dict())

    assert (reloaded.log_scaling_n_floor, reloaded.log_scaling_alpha, reloaded.log_scaling_power) == expected
    assert reloaded.uses_hero_architecture


@pytest.mark.parametrize(
    ("schema", "log_scaling"),
    [
        (2, HERO_LOG_SCALING),
        (3, {"log_scaling_n_floor": 65536, "log_scaling_alpha": 0.1, "log_scaling_power": None}),
        (3, {**HERO_LOG_SCALING, "log_scaling_alpha": 0.0}),
    ],
    ids=["fields_before_schema_3", "partial_fields", "zero_slope"],
)
def test_config_rejects_inconsistent_log_scaling(schema, log_scaling):
    with pytest.raises(ValueError, match="log_scaling"):
        GrugMoeConfig(grugmoe_artifact_schema_version=schema, **log_scaling)
