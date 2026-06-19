# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from marin.profiling.semantics import classify_semantic_family, estimate_flop_proxy


def test_estimate_flop_proxy_attention_handles_nonpositive_dims() -> None:
    assert estimate_flop_proxy("attention_splash", "0,8,2048,64|0,8,2048,64") is None


def test_estimate_flop_proxy_loss_handles_nonpositive_dims() -> None:
    assert estimate_flop_proxy("loss_xent", "0,512|0,128256") is None


def test_classify_semantic_family_uses_attributed_framework_path() -> None:
    assert (
        classify_semantic_family(
            "nvjet_sm90_tst_128x160_64x5_2x1_v_bz_NNT",
            "jit(train_step)/optimizer_update/muonh/newton_schulz_grouped_4d/dot_general",
        )
        == "optimizer_muon"
    )


def test_classify_semantic_family_splits_non_kernel_framework_paths() -> None:
    assert classify_semantic_family("add", "jit(train_step)/apply_updates/add") == "optimizer_apply"
    assert (
        classify_semantic_family(
            "dot_general",
            "jit(train_step)/forward_backward/jvp(Transformer)/Block/Block._attention_update/CausalSelfAttention/bsd,dn->bsn/dot_general",
        )
        == "attention_dense"
    )
    assert (
        classify_semantic_family(
            "mul",
            "jit(train_step)/forward_backward/jvp(Transformer)/Block/Block._mlp_update/GatedNorm/mul",
        )
        == "norm_gating"
    )
