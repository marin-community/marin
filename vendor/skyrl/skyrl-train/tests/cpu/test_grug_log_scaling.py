"""Grug log scaling: per-document positions under packing and CP, and the checkpoint contract."""

import pytest
import torch

from skyrl_train.models.grug_log_scaling import thd_document_positions
from skyrl_train.models.grug_moe import GrugMoeConfig
from tests.cpu.util import stub_megatron_modules

HERO_LOG_SCALING = dict(log_scaling_n_floor=65536, log_scaling_alpha=0.1, log_scaling_power=2.0)


@pytest.mark.parametrize("cp_size", [1, 2, 4])
def test_positions_restart_per_document_on_every_cp_rank(monkeypatch, cp_size):
    stub_megatron_modules()
    # Megatron is optional in CPU CI; import its packing utilities after installing the stubs.
    from skyrl_train.distributed.megatron import megatron_utils  # noqa: PLC0415

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
