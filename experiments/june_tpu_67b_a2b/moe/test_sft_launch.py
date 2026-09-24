# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace

from fray.cluster import ResourceConfig
from levanter.data.text.datasets import LmDataConfig
from levanter.optim.config import AdamConfig
from marin.execution.lazy import StepContext

from experiments.june_tpu_67b_a2b.moe.model import GrugModelConfig
from experiments.june_tpu_67b_a2b.moe.sft_launch import GrugModel
from experiments.sft.launcher import SFTSpec


def test_grug_sft_run_id_tracks_full_name_and_version():
    model = GrugModel(
        model=GrugModelConfig(vocab_size=128),
        tokenizer_path="test-tokenizer",
        init_from="/tmp/base-checkpoint",
        expert_parallel=1,
    )
    spec = SFTSpec(
        name="snowball-final/qk157/chat",
        version="2026.09.24",
        model=model,
        chat_template="",
        datasets=[],
        optimizer=AdamConfig(),
        num_train_steps=1,
    )
    ctx = StepContext.for_run("/tmp/sft-output", "/tmp")
    resources = ResourceConfig.with_gpu("H100")

    def run_id(candidate):
        return model.build_train_config(ctx, candidate, LmDataConfig(), resources, 1).run_id

    first = run_id(spec)
    assert first == run_id(spec)
    assert first != run_id(replace(spec, name="snowball-final/qk175/chat"))
    assert first != run_id(replace(spec, version="2026.09.25"))
