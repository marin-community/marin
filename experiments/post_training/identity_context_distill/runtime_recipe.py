# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bind runtime-only router precision to immutable reference evidence, not HF defaults."""

import hashlib
import json
import os
from dataclasses import asdict, replace

from jax._src import xla_bridge
from levanter.main.train_lm import TrainLmConfig
from levanter.models.snowball import SnowballConfig

from experiments.interp.snowball_identity.routing import bias_exception_required, dot_exception_required
from experiments.post_training.identity_context_distill.sft import (
    ATTENTION_IMPLEMENTATIONS,
    check_runtime_model,
    source_export_overrides,
    source_metadata_receipt,
)

POSITIONS_SHA256 = "81fdfb85ae40e092c4b99c29440a7fccbc075b2d08bda2999321b250b82e201d"
XLA_PRECISION_FLAGS = {"default": None, "no_excess_precision": "--xla_allow_excess_precision=false"}
"""The recipe's XLA precision setting -> the process's ``XLA_FLAGS``, which XLA reads when its backend starts."""


XLA_CACHE_FLAGS = ("--xla_gpu_per_fusion_autotune_cache_dir",)
"""Flags Iris adds to ``XLA_FLAGS`` at JAX init (iris.jax.compile_cache): a cache location, not a numeric setting."""


def xla_precision_in_process() -> str:
    """This process's XLA precision setting, read from ``XLA_FLAGS``; any XLA flag other than the setting's own and
    Iris's cache location is refused."""
    kept = [flag for flag in os.environ.get("XLA_FLAGS", "").split() if flag.partition("=")[0] not in XLA_CACHE_FLAGS]
    flags = " ".join(kept) or None
    for name, value in XLA_PRECISION_FLAGS.items():
        if flags == value:
            return name
    raise ValueError(f"XLA_FLAGS {flags!r} is not a recipe XLA precision setting")


def check_xla_precision(expected: str) -> None:
    """Verify the recorded flag selection, including after the backend starts."""
    if expected not in XLA_PRECISION_FLAGS:
        raise ValueError(f"Unknown XLA precision setting {expected!r}")
    found = xla_precision_in_process()
    if found != expected:
        raise ValueError(f"XLA precision {found!r} in this process, the receipt binds {expected!r}")


def require_xla_precision(expected: str) -> None:
    """Check the process's selection before XLA reads it at backend startup."""
    check_xla_precision(expected)
    if xla_bridge.backends_are_initialized():
        raise RuntimeError("XLA's backend started before the XLA precision setting was checked")


def xla_precision_env(setting: str) -> dict[str, str]:
    """The job environment that applies ``setting`` before device init (empty for XLA's defaults)."""
    if setting not in XLA_PRECISION_FLAGS:
        raise ValueError(f"Unknown XLA precision setting {setting!r}")
    flags = XLA_PRECISION_FLAGS[setting]
    return {} if flags is None else {"XLA_FLAGS": flags}


def check_runtime_config(config: TrainLmConfig, source: dict) -> None:
    """Check the native architecture and the only permitted field-level cast."""
    if not isinstance(config.model, SnowballConfig):
        raise ValueError("Runtime recipe requires SnowballConfig")
    check_runtime_model(config.model, source)
    overrides = config.trainer.compute_dtype_overrides
    if overrides not in (None, {"router_bias": "float32"}):
        raise ValueError("Only the exact router_bias FP32 compute exception is permitted")
    if config.model.router_dot_fp32 and overrides != {"router_bias": "float32"}:
        raise ValueError("FP32 router dot requires native FP32 router biases")
    if config.model.attention_implementation not in ATTENTION_IMPLEMENTATIONS:
        raise ValueError(f"Attention implementation {config.model.attention_implementation!r} is not a recipe option")
    expert = config.trainer.mesh.axes["expert"]
    if config.model.moe_capacity_factor != expert:
        raise ValueError(
            f"The MoE must be dropless: capacity factor {config.model.moe_capacity_factor}, expert axis {expert}"
        )


def configure_precision_runtime(
    config: TrainLmConfig,
    *,
    config_bytes: bytes,
    generation_bytes: bytes,
    rounding_bytes: bytes,
    rounding_path: str,
    rounding_sha256: str,
    xla_precision: str,
) -> tuple[TrainLmConfig, dict]:
    """Apply only the registered router exceptions and produce the derived receipt.

    The source receipt describes the unmodified regional HF metadata (runtime flag
    false). The separate runtime model records the measured exception. The caller
    must pin ``rounding_sha256`` in its reviewed worker inputs, not calculate a new
    expected digest from an unreviewed replacement artifact. ``xla_precision`` is the
    XLA precision setting (``XLA_PRECISION_FLAGS``): given explicitly when a receipt is
    derived, read from the process (``xla_precision_in_process``) when a worker checks it.
    """
    if xla_precision not in XLA_PRECISION_FLAGS:
        raise ValueError(f"Unknown XLA precision setting {xla_precision!r}")
    if hashlib.sha256(rounding_bytes).hexdigest() != rounding_sha256:
        raise ValueError("Router reference bytes differ from the approved digest")
    if not rounding_path.startswith("s3://marin-rno2a/"):
        raise ValueError("Router reference must be the approved regional artifact")
    rounding = json.loads(rounding_bytes)
    if (
        rounding.get("positions_sha256") != POSITIONS_SHA256
        or rounding.get("sequences") != 465
        or rounding.get("top_k") != 4
        or any(
            rounding["totals"][variant]["positions"] != 805974
            for variant in ("bf16_bias", "bf16_logits", "bf16_logits_bias")
        )
    ):
        raise ValueError("Router reference does not cover the frozen inputs")
    dot = dot_exception_required(rounding)
    bias = bias_exception_required(rounding)
    if not dot and not bias and rounding["totals"]["bf16_logits_bias"]["positions_changed"]:
        raise ValueError("Combined-only routing discrepancy requires diagnosis")
    overrides = {"router_bias": "float32"} if dot or bias else None
    source = json.loads(config_bytes)
    check_runtime_config(config, source)
    assert isinstance(config.model, SnowballConfig)
    runtime = replace(
        config,
        model=replace(config.model, router_dot_fp32=dot),
        trainer=replace(config.trainer, compute_dtype_overrides=overrides),
    )
    check_runtime_config(runtime, source)
    receipt = {
        "schema_version": 6,
        "source_metadata": source_metadata_receipt(config_bytes, generation_bytes),
        "runtime_model_config": asdict(runtime.model),
        "compute_dtype_overrides": overrides,
        "moe_capacity": {
            "factor": runtime.model.moe_capacity_factor,
            "expert_axis": runtime.trainer.mesh.axes["expert"],
        },
        "router_combine_fp32": runtime.model.router_combine_fp32,
        "fp32_activations": runtime.model.fp32_activations,
        "attention_implementation": runtime.model.attention_implementation,
        "xla_precision": xla_precision,
        "native_export_config_overrides": source_export_overrides(source),
        "routing_reference": {"path": rounding_path, "sha256": rounding_sha256},
        "trigger": {"bias_only": bias, "dot_only": dot},
        "load_binding": "native HF reload, shared runtime-setting restoration, then explicit loader config",
        "stock_train_lm_hf_reload_supported": True,
    }
    return runtime, receipt


def check_precision_runtime_receipt(
    config: TrainLmConfig,
    expected: dict,
    *,
    config_bytes: bytes,
    generation_bytes: bytes,
    rounding_bytes: bytes,
) -> None:
    """Validate the actual worker recipe before loading; never silently repair it."""
    runtime, observed = configure_precision_runtime(
        config,
        config_bytes=config_bytes,
        generation_bytes=generation_bytes,
        rounding_bytes=rounding_bytes,
        rounding_path=expected["routing_reference"]["path"],
        rounding_sha256=expected["routing_reference"]["sha256"],
        xla_precision=xla_precision_in_process(),
    )
    if (
        observed != expected
        or runtime.model != config.model
        or runtime.trainer.compute_dtype_overrides != config.trainer.compute_dtype_overrides
    ):
        raise ValueError("Runtime recipe differs from the approved derived receipt")
