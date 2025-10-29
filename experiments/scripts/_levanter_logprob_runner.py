"""Levanter-backed Gemma log-prob computation built for Marin executor steps."""

from __future__ import annotations

import logging
import posixpath
from dataclasses import dataclass, field, replace
from typing import Any

import haliax as hax
import jax
import jax.numpy as jnp
import ray
from levanter.compat.hf_checkpoints import RepoRef
from levanter.models.gemma import Gemma2Config as LevGemma2Config
from levanter.models.gemma import Gemma3Config as LevGemma3Config
from levanter.models.gemma import GemmaConfig as LevGemmaConfig
from levanter.models.lm_model import LmExample, compute_next_token_loss
from levanter.utils.tree_utils import inference_mode
from transformers import AutoConfig, AutoTokenizer

from marin.execution.executor import this_output_path

from gemma_logprob_utils import (
    DEFAULT_PROMPT,
    LogProbResult,
    add_eos_if_missing,
    compare_results,
    load_result,
    save_result,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GemmaLevanterLogProbConfig:
    """Configuration for a single Levanter log-prob computation."""

    backend: str
    model_id: str = "google/gemma-2-27b"
    revision: str | None = "main"
    prompt: str = DEFAULT_PROMPT
    dtype: str = "bfloat16"
    output_filename: str = "logprob.json"
    reference_path: str | None = None
    tolerance: float = 5e-5
    output_path: str = field(default_factory=this_output_path)
    resource_config: Any | None = None


def run_gemma_levanter_logprob(config: GemmaLevanterLogProbConfig) -> None:
    """Compute Gemma log probabilities with Levanter and optionally compare to a reference."""
    remote_kwargs = _ray_remote_options_for_logprob(config.resource_config)
    worker_config = replace(config, resource_config=None)
    summary = ray.get(_remote_logprob.options(**remote_kwargs).remote(worker_config))

    logger.info(
        "[%s] total log-prob %.6f (%d tokens) -> %s",
        config.backend,
        summary["total_logprob"],
        summary["token_count"],
        summary["output_file"],
    )

    diffs = summary.get("diffs")
    if diffs is not None:
        logger.info(
            "Matched reference %s within tolerance (total diff %.2e, per-token diff %.2e)",
            config.reference_path,
            diffs["total_diff"],
            diffs["per_token_diff"],
        )


@ray.remote
def _remote_logprob(config: GemmaLevanterLogProbConfig) -> dict[str, Any]:
    logger.info(
        "[%s] Starting Gemma log-prob evaluation for %s@%s",
        config.backend,
        config.model_id,
        config.revision or "latest",
    )
    revision = config.revision or None
    result = _compute_result(
        backend=config.backend,
        model_id=config.model_id,
        revision=revision,
        prompt=config.prompt,
        dtype_name=config.dtype,
    )

    output_file = posixpath.join(config.output_path, config.output_filename)
    save_result(result, output_file)
    logger.info(
        "[%s] Completed forward pass (%d target tokens). Results written to %s",
        config.backend,
        len(result.per_token_logprobs),
        output_file,
    )

    diffs = None
    if config.reference_path:
        reference = load_result(config.reference_path)
        diffs = compare_results(result, reference, tolerance=config.tolerance)
        logger.info(
            "[%s] Compared against reference %s (total diff %.2e, per-token diff %.2e)",
            config.backend,
            config.reference_path,
            diffs["total_diff"],
            diffs["per_token_diff"],
        )

    return {
        "total_logprob": result.total_logprob,
        "token_count": len(result.per_token_logprobs),
        "output_file": output_file,
        "diffs": diffs,
    }


def _compute_result(
    *,
    backend: str,
    model_id: str,
    revision: str | None,
    prompt: str,
    dtype_name: str,
) -> LogProbResult:
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, trust_remote_code=True)
    token_ids = tokenizer.encode(prompt, add_special_tokens=False)
    token_ids = add_eos_if_missing(token_ids, tokenizer.eos_token_id)
    logger.info(
        "[%s] Tokenized prompt into %d ids (model vocab size approx %s)",
        backend,
        len(token_ids),
        getattr(tokenizer, "vocab_size", "unknown"),
    )
    repo_ref = RepoRef.from_string(f"{model_id}@{revision}" if revision else model_id)

    lev_config = _levanter_config_for(model_id, revision)
    converter = lev_config.hf_checkpoint_converter().replaced(reference_checkpoint=repo_ref, tokenizer=tokenizer)

    dtype = _dtype_from_name(dtype_name)
    model = converter.load_pretrained(
        lev_config.model_type,
        ref=repo_ref,
        config=lev_config,
        dtype=dtype,
    )
    model = inference_mode(model, True)
    logger.info("[%s] Loaded HF weights into Levanter model with dtype %s", backend, dtype.name)

    Pos = hax.Axis("position", len(token_ids))
    tokens_named = hax.named(jnp.asarray(token_ids, dtype=jnp.int32), axis=Pos)
    example = LmExample.causal(tokens_named, eos_id=tokenizer.eos_token_id)
    loss_named = compute_next_token_loss(model, example, reduction=None)

    loss_values = jax.device_get(loss_named.array)
    mask_values = jax.device_get(example.loss_mask.array)

    if len(mask_values) != len(token_ids):
        raise ValueError(
            f"Internal error: mask length {len(mask_values)} does not match token count {len(token_ids)}"
        )

    per_token_logprobs: list[float] = []
    predicted_token_ids: list[int] = []
    for idx, mask in enumerate(mask_values):
        if mask:
            # Mask position idx predicts token idx + 1.
            target_idx = idx + 1
            if target_idx >= len(token_ids):
                continue
            predicted_token_ids.append(int(token_ids[target_idx]))
            per_token_logprobs.append(-float(loss_values[idx]))

    predicted_tokens = tokenizer.convert_ids_to_tokens(predicted_token_ids)

    return LogProbResult(
        backend=backend,
        model_id=model_id,
        revision=revision,
        prompt=prompt,
        token_ids=list(map(int, token_ids)),
        predicted_token_ids=predicted_token_ids,
        predicted_tokens=predicted_tokens,
        per_token_logprobs=per_token_logprobs,
    )


def _levanter_config_for(model_id: str, revision: str | None) -> LevGemmaConfig:
    hf_config = AutoConfig.from_pretrained(model_id, revision=revision, trust_remote_code=True)
    cls_name = hf_config.__class__.__name__.lower()

    if "gemma3" in cls_name:
        return LevGemma3Config.from_hf_config(hf_config)
    if "gemma2" in cls_name:
        return LevGemma2Config.from_hf_config(hf_config)
    return LevGemmaConfig.from_hf_config(hf_config)


def _dtype_from_name(name: str):
    if name == "bfloat16":
        return jnp.bfloat16
    if name == "float32":
        return jnp.float32
    raise ValueError(f"Unsupported dtype {name}")


def _ray_remote_options_for_logprob(resource_config: Any | None) -> dict[str, Any]:
    """
    Construct Ray remote options based on the supplied resource configuration.

    Mirrors the behaviour used by Marin's log-prob evaluator so we can request TPU/GPU slices.
    """
    if resource_config is None:
        return {}

    remote_kwargs: dict[str, Any] = {}

    if hasattr(resource_config, "as_remote_kwargs"):
        remote_kwargs.update(resource_config.as_remote_kwargs())
    elif hasattr(resource_config, "as_ray_resources"):
        remote_kwargs.update(resource_config.as_ray_resources().to_kwargs())

    resources = dict(remote_kwargs.pop("resources", {}) or {})

    # Legacy TPU configs from experiments/evals/resource_configs.
    if hasattr(resource_config, "tpu_type") and hasattr(resource_config, "num_tpu"):
        resources.setdefault("TPU", getattr(resource_config, "num_tpu", 1))
        resources.setdefault(f"{resource_config.tpu_type}-head", 1)
    # Modern TPU pod configs.
    elif hasattr(resource_config, "tpu_type"):
        slice_count = getattr(resource_config, "slice_count", 1)
        resources.setdefault("TPU", slice_count if isinstance(slice_count, int) else max(slice_count))
        resources.setdefault(f"{resource_config.tpu_type}-head", 1)
    # GPU configs.
    elif hasattr(resource_config, "gpu_count"):
        remote_kwargs.setdefault("num_gpus", getattr(resource_config, "gpu_count", 1))
        if getattr(resource_config, "accelerator_type", None):
            remote_kwargs["accelerator_type"] = resource_config.accelerator_type

    if resources:
        remote_kwargs["resources"] = resources

    return remote_kwargs
