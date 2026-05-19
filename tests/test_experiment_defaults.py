# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import logging

from experiments.defaults import _truncate_wandb_name


def test_truncate_wandb_name_preserves_scientific_notation_lr_suffix():
    name = "dpo/new_dpo_v2_bloom_speceval_v2_marin_instruct_beta0.1_lr7.5e-7_seed2"

    truncated = _truncate_wandb_name(name)

    assert len(truncated) <= 64
    assert truncated.endswith("lr7.5e-7_seed2")
    assert "_-7_seed2" not in truncated


def test_truncate_wandb_name_logs_warning(caplog):
    name = "dpo/new_dpo_v2_bloom_speceval_v2_marin_instruct_beta0.1_lr7.5e-7_seed2"

    with caplog.at_level(logging.WARNING):
        _truncate_wandb_name(name)

    assert any(record.levelno >= logging.WARNING for record in caplog.records)
