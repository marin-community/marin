# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from copy import deepcopy
from dataclasses import replace

import pytest
from levanter.models.snowball import SnowballConfig

from experiments.post_training.identity_context_distill.runtime_recipe import (
    POSITIONS_SHA256,
    check_precision_runtime_receipt,
    check_runtime_config,
    configure_precision_runtime,
)
from experiments.post_training.identity_context_distill.sft import identity_sft_config
from experiments.post_training.identity_context_distill.test_sft import source_arguments


def inputs():
    source = source_arguments()
    config_bytes = json.dumps(source["source_config"]).encode()
    generation_bytes = json.dumps(source["source_generation_config"]).encode()
    config = identity_sft_config(
        moe_capacity_factor=4.0,
        dataset_uri="synthetic",
        cache_uri="cache",
        output_uri="s3://marin-rno2a/fixture",
        run_id="test",
        seed=310,
        hosts=1,
        **source,
    )
    # Synthetic evidence metadata only; no model outputs or real training targets.
    rounding = {
        "positions_sha256": POSITIONS_SHA256,
        "sequences": 465,
        "top_k": 4,
        "totals": {
            "bf16_bias": {"positions": 805974, "positions_changed": 0},
            "bf16_logits": {"positions": 805974, "positions_changed": 1},
            "bf16_logits_bias": {"positions": 805974, "positions_changed": 1},
        },
    }
    rounding_bytes = json.dumps(rounding).encode()
    args = dict(
        config_bytes=config_bytes,
        generation_bytes=generation_bytes,
        rounding_bytes=rounding_bytes,
        rounding_path="s3://marin-rno2a/fixture/routing.json",
        rounding_sha256=hashlib.sha256(rounding_bytes).hexdigest(),
    )
    return config, args


def test_measured_dot_policy_accepts_runtime_flag_without_changing_native_metadata():
    config, args = inputs()
    source = json.loads(args["config_bytes"])
    before = deepcopy(source)
    runtime, receipt = configure_precision_runtime(config, **args)
    assert runtime.model.router_dot_fp32 is True
    assert runtime.trainer.compute_dtype_overrides == {"router_bias": "float32"}
    assert runtime.model.qk_mult == 1.75 and runtime.model.max_seq_len == 262144
    assert receipt["source_metadata"]["schema_version"] == 2
    assert receipt["source_metadata"]["model_config"]["router_dot_fp32"] is False
    assert receipt["runtime_model_config"]["router_dot_fp32"] is True
    assert receipt["schema_version"] == 3
    assert receipt["moe_capacity"] == {"factor": 4.0, "expert_axis": 4}
    assert receipt["runtime_model_config"]["moe_capacity_factor"] == 4.0
    assert receipt["source_metadata"]["model_config"]["moe_capacity_factor"] == 1.0
    assert receipt["native_export_config_overrides"] == config.adapter.hf_config_overrides
    assert source == before and json.loads(args["config_bytes"]) == before
    check_runtime_config(runtime, source)
    check_precision_runtime_receipt(
        runtime, receipt, **{k: args[k] for k in ("config_bytes", "generation_bytes", "rounding_bytes")}
    )


def test_runtime_receipt_fails_changed_evidence_or_unbound_runtime():
    config, args = inputs()
    runtime, receipt = configure_precision_runtime(config, **args)
    binding = {k: args[k] for k in ("config_bytes", "generation_bytes", "rounding_bytes")}
    with pytest.raises(ValueError, match="Runtime recipe"):
        check_precision_runtime_receipt(config, receipt, **binding)
    changed = deepcopy(receipt)
    changed["runtime_model_config"]["router_dot_fp32"] = False
    with pytest.raises(ValueError, match="Runtime recipe"):
        check_precision_runtime_receipt(runtime, changed, **binding)
    with pytest.raises(ValueError, match="approved digest"):
        check_precision_runtime_receipt(runtime, receipt, **dict(binding, rounding_bytes=b"{}"))


def test_runtime_flag_does_not_waive_scaling_or_master_precision_policy():
    config, args = inputs()
    runtime, _ = configure_precision_runtime(config, **args)
    source = json.loads(args["config_bytes"])
    with pytest.raises(ValueError, match="architecture differs"):
        assert isinstance(runtime.model, SnowballConfig)
        check_runtime_config(replace(runtime, model=replace(runtime.model, qk_mult=1.5)), source)
    with pytest.raises(ValueError, match="requires native FP32"):
        check_runtime_config(replace(runtime, trainer=replace(runtime.trainer, compute_dtype_overrides=None)), source)
    with pytest.raises(ValueError, match="exact router_bias"):
        check_runtime_config(
            replace(runtime, trainer=replace(runtime.trainer, compute_dtype_overrides={"router": "float32"})), source
        )


def test_runtime_recipe_requires_a_dropless_moe():
    config, args = inputs()
    runtime, receipt = configure_precision_runtime(config, **args)
    source = json.loads(args["config_bytes"])
    assert isinstance(runtime.model, SnowballConfig)
    with pytest.raises(ValueError, match="dropless"):
        check_runtime_config(replace(runtime, model=replace(runtime.model, moe_capacity_factor=1.0)), source)
    with pytest.raises(ValueError, match="dropless"):
        configure_precision_runtime(replace(config, model=replace(config.model, moe_capacity_factor=1.0)), **args)
    changed = deepcopy(receipt)
    changed["moe_capacity"]["factor"] = 1.0
    binding = {k: args[k] for k in ("config_bytes", "generation_bytes", "rounding_bytes")}
    with pytest.raises(ValueError, match="Runtime recipe"):
        check_precision_runtime_receipt(runtime, changed, **binding)
