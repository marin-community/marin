# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared DPO runtime helpers."""

import hashlib
import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Literal

import equinox as eqx
import haliax as hax
import jax
import jax.numpy as jnp
import jmp
import numpy as np
from haliax.partitioning import ResourceMapping, named_jit

from levanter.data.dataset import AsyncDataset
from levanter.data.loader import DataLoader
from levanter.data.text import DpoExample
from levanter.metrics import Metric, ReductionType
from levanter.models.lm_model import LmHeadModel
from levanter.store.cache import CacheLedger, CacheMetadata, SerialCacheWriter, TreeCache
from levanter.utils.py_utils import FailSafeJSONEncoder
from levanter.utils.tree_utils import inference_mode


logger = logging.getLogger(__name__)

REFERENCE_LOGPROBS_CACHE_KIND = "dpo_reference_logprobs_v1"
_REFERENCE_LOGPROBS_EXEMPLAR = {
    "logp_ref_chosen": np.zeros((), dtype=np.float32),
    "logp_ref_rejected": np.zeros((), dtype=np.float32),
}


class DpoModel(eqx.Module):
    policy: LmHeadModel
    reference: LmHeadModel


class CachedDpoExample(eqx.Module):
    """DPO example augmented with precomputed reference log-probs."""

    chosen: object
    rejected: object
    logp_ref_chosen: float
    logp_ref_rejected: float


@dataclass(frozen=True)
class ReferenceEvalCacheConfig:
    mode: Literal["disabled", "build_or_load"] = "disabled"
    cache_dir: str | None = None


@dataclass(frozen=True)
class ValidationDatasetSpec:
    name: str
    dataset: AsyncDataset[DpoExample] | AsyncDataset[CachedDpoExample]
    source_cache_path: str
    source_split: str
    slice_start: int | None = None
    slice_end: int | None = None


class CachedReferenceDataset(AsyncDataset[CachedDpoExample]):
    """AsyncDataset wrapper that attaches cached reference log-probs by index."""

    def __init__(self, base_dataset: AsyncDataset[DpoExample], ref_chosen: np.ndarray, ref_rejected: np.ndarray):
        self._base = base_dataset
        self._ref_chosen = np.asarray(ref_chosen, dtype=np.float32).reshape(-1)
        self._ref_rejected = np.asarray(ref_rejected, dtype=np.float32).reshape(-1)

        expected_len = len(base_dataset.as_sync_dataset())
        if len(self._ref_chosen) != expected_len or len(self._ref_rejected) != expected_len:
            raise ValueError(
                "Cached reference log-probs must match the validation dataset length. "
                f"Got chosen={len(self._ref_chosen)}, rejected={len(self._ref_rejected)}, dataset={expected_len}."
            )

    async def async_len(self) -> int:
        return await self._base.async_len()

    def is_finite(self) -> bool:
        return self._base.is_finite()

    async def getitem_async(self, index: int) -> CachedDpoExample:
        example = await self._base.getitem_async(index)
        return CachedDpoExample(
            chosen=example.chosen,
            rejected=example.rejected,
            logp_ref_chosen=self._ref_chosen[index],
            logp_ref_rejected=self._ref_rejected[index],
        )

    async def get_batch(self, indices) -> list[CachedDpoExample]:
        examples = await self._base.get_batch(indices)
        return [
            CachedDpoExample(
                chosen=example.chosen,
                rejected=example.rejected,
                logp_ref_chosen=self._ref_chosen[index],
                logp_ref_rejected=self._ref_rejected[index],
            )
            for index, example in zip(indices, examples, strict=True)
        ]


def dpo_loss_from_logps(
    delta_pi: hax.NamedArray,
    delta_ref: hax.NamedArray,
    *,
    beta: float,
) -> tuple[jnp.ndarray, dict[str, Metric]]:
    logits = (delta_pi - delta_ref) * beta
    loss = hax.mean(hax.nn.softplus(-logits)).scalar()
    metrics = {
        "dpo_loss": Metric.from_value(loss, ReductionType.MEAN),
        "dpo_margin_policy": Metric.from_value(hax.mean(delta_pi).scalar(), ReductionType.MEAN),
        "dpo_margin_ref": Metric.from_value(hax.mean(delta_ref).scalar(), ReductionType.MEAN),
        "dpo_accuracy": Metric.from_value(hax.mean(logits > 0).scalar(), ReductionType.MEAN),
    }
    return loss, metrics


def logp_sum(model: LmHeadModel, example, *, key=None) -> hax.NamedArray:
    nll = model.compute_next_token_loss(example, reduction=hax.sum, reduction_axis="position", key=key)
    return -nll


def _logp_sum(model: LmHeadModel, example, *, key=None) -> hax.NamedArray:
    return logp_sum(model, example, key=key)


def _cached_logp_named(logps: Any, axes: tuple[hax.Axis, ...]) -> hax.NamedArray:
    return hax.named(jnp.reshape(jnp.asarray(logps), tuple(axis.size for axis in axes)), axes)


def _reference_logps_for_example(
    example: DpoExample | CachedDpoExample,
    *,
    reference_model: LmHeadModel | None,
    logp_axes: tuple[hax.Axis, ...],
    key_chosen=None,
    key_rejected=None,
) -> tuple[hax.NamedArray, hax.NamedArray]:
    if isinstance(example, CachedDpoExample):
        return _cached_logp_named(example.logp_ref_chosen, logp_axes), _cached_logp_named(
            example.logp_ref_rejected, logp_axes
        )

    if reference_model is None:
        raise ValueError("reference_model is required when reference log-probs are not cached.")

    with jax.named_scope("reference_chosen"):
        logp_ref_chosen = jax.lax.stop_gradient(logp_sum(reference_model, example.chosen, key=key_chosen))
    with jax.named_scope("reference_rejected"):
        logp_ref_rejected = jax.lax.stop_gradient(logp_sum(reference_model, example.rejected, key=key_rejected))
    return logp_ref_chosen, logp_ref_rejected


def dpo_loss(
    policy_model: LmHeadModel,
    reference_model: LmHeadModel | None,
    example: DpoExample | CachedDpoExample,
    *,
    beta: float,
    key_chosen=None,
    key_rejected=None,
) -> tuple[jnp.ndarray, dict[str, Metric]]:
    with jax.named_scope("policy_chosen"):
        logp_pi_chosen = logp_sum(policy_model, example.chosen, key=key_chosen)
    with jax.named_scope("policy_rejected"):
        logp_pi_rejected = logp_sum(policy_model, example.rejected, key=key_rejected)

    logp_ref_chosen, logp_ref_rejected = _reference_logps_for_example(
        example,
        reference_model=reference_model,
        logp_axes=logp_pi_chosen.axes,
        key_chosen=key_chosen,
        key_rejected=key_rejected,
    )

    delta_pi = logp_pi_chosen - logp_pi_rejected
    delta_ref = logp_ref_chosen - logp_ref_rejected

    loss, metrics = dpo_loss_from_logps(delta_pi, delta_ref, beta=beta)
    chosen_reward = (logp_pi_chosen - logp_ref_chosen) * beta
    rejected_reward = (logp_pi_rejected - logp_ref_rejected) * beta
    metrics["dpo_chosen_reward"] = Metric.from_value(hax.mean(chosen_reward).scalar(), ReductionType.MEAN)
    metrics["dpo_rejected_reward"] = Metric.from_value(hax.mean(rejected_reward).scalar(), ReductionType.MEAN)
    return loss, metrics


def _cache_identity_payload(
    spec: ValidationDatasetSpec,
    *,
    reference_identity: dict[str, Any],
    seq_len: int,
) -> dict[str, Any]:
    return {
        "kind": REFERENCE_LOGPROBS_CACHE_KIND,
        "source_cache_path": spec.source_cache_path,
        "source_split": spec.source_split,
        "slice_start": spec.slice_start,
        "slice_end": spec.slice_end,
        "reference_identity": reference_identity,
        "seq_len": seq_len,
    }


def reference_eval_cache_metadata(
    spec: ValidationDatasetSpec,
    *,
    reference_identity: dict[str, Any],
    seq_len: int,
) -> CacheMetadata:
    return CacheMetadata(
        preprocessor_metadata=_cache_identity_payload(spec, reference_identity=reference_identity, seq_len=seq_len)
    )


def reference_eval_cache_path(
    spec: ValidationDatasetSpec,
    *,
    reference_identity: dict[str, Any],
    seq_len: int,
    cache_dir: str | None,
) -> str:
    payload = _cache_identity_payload(spec, reference_identity=reference_identity, seq_len=seq_len)
    cache_hash = hashlib.sha256(
        json.dumps(payload, sort_keys=True, cls=FailSafeJSONEncoder).encode("utf-8")
    ).hexdigest()[:8]

    if cache_dir is None:
        cache_root = os.path.join(spec.source_cache_path.rsplit("/", 1)[0], "reference_logprobs")
    else:
        cache_root = cache_dir

    return os.path.join(cache_root, cache_hash)


def load_reference_eval_cache(
    cache_dir: str,
    *,
    metadata: CacheMetadata,
) -> tuple[np.ndarray, np.ndarray]:
    ledger = CacheLedger.load(cache_dir)
    if not ledger.is_finished:
        raise FileNotFoundError(f"Reference eval cache at {cache_dir} is not finished.")

    diff = ledger.metadata.compare_to(metadata)
    if diff:
        raise FileNotFoundError(f"Reference eval cache metadata mismatch at {cache_dir}: {diff}")

    cache = TreeCache.load(cache_dir, exemplar=_REFERENCE_LOGPROBS_EXEMPLAR)
    rows = cache.get_batch_sync(slice(0, len(cache)))
    chosen = np.asarray([row["logp_ref_chosen"] for row in rows], dtype=np.float32).reshape(-1)
    rejected = np.asarray([row["logp_ref_rejected"] for row in rows], dtype=np.float32).reshape(-1)
    return chosen, rejected


def build_reference_eval_cache(
    *,
    reference_model: LmHeadModel,
    dataset: AsyncDataset[DpoExample],
    eval_loader: DataLoader[DpoExample],
    compute_axis_mapping: ResourceMapping | None,
    mp: jmp.Policy,
    cache_dir: str,
    metadata: CacheMetadata,
) -> tuple[np.ndarray, np.ndarray]:
    logger.info("Building reference eval cache at %s", cache_dir)

    @named_jit(axis_resources=compute_axis_mapping)
    def ref_logprobs_fn(ref_model, batch: DpoExample):
        ref_model = mp.cast_to_compute(ref_model)
        ref_model = inference_mode(ref_model, True)
        return (
            jax.lax.stop_gradient(logp_sum(ref_model, batch.chosen, key=None)),
            jax.lax.stop_gradient(logp_sum(ref_model, batch.rejected, key=None)),
        )

    expected_len = len(dataset.as_sync_dataset())
    if expected_len == 0:
        raise ValueError("Reference eval cache requires a non-empty validation dataset.")

    warmup_iter = iter(eval_loader)
    batch = next(warmup_iter, None)
    if batch is None:
        raise ValueError("Reference eval cache requires a non-empty eval loader.")

    ref_logprobs_fn(reference_model, batch)

    all_chosen: list[np.ndarray] = []
    all_rejected: list[np.ndarray] = []

    for batch in eval_loader:
        logp_chosen, logp_rejected = ref_logprobs_fn(reference_model, batch)
        chosen_gathered = jax.experimental.multihost_utils.process_allgather(logp_chosen.array, tiled=True)
        rejected_gathered = jax.experimental.multihost_utils.process_allgather(logp_rejected.array, tiled=True)
        all_chosen.append(np.asarray(chosen_gathered, dtype=np.float32).reshape(-1))
        all_rejected.append(np.asarray(rejected_gathered, dtype=np.float32).reshape(-1))

    chosen = np.concatenate(all_chosen)[:expected_len] if all_chosen else np.zeros((0,), dtype=np.float32)
    rejected = np.concatenate(all_rejected)[:expected_len] if all_rejected else np.zeros((0,), dtype=np.float32)

    if jax.process_index() == 0:
        with SerialCacheWriter(cache_dir, _REFERENCE_LOGPROBS_EXEMPLAR, metadata=metadata) as writer:
            writer.write_batch({"logp_ref_chosen": chosen, "logp_ref_rejected": rejected})

    jax.experimental.multihost_utils.sync_global_devices("reference_eval_cache_written")
    return chosen, rejected


def build_or_load_reference_eval_cache(
    *,
    reference_model: LmHeadModel,
    dataset: AsyncDataset[DpoExample],
    eval_loader: DataLoader[DpoExample],
    compute_axis_mapping: ResourceMapping | None,
    mp: jmp.Policy,
    cache_dir: str,
    metadata: CacheMetadata,
) -> tuple[np.ndarray, np.ndarray]:
    try:
        logger.info("Loading reference eval cache from %s", cache_dir)
        return load_reference_eval_cache(cache_dir, metadata=metadata)
    except FileNotFoundError as exc:
        logger.info("Reference eval cache miss at %s: %s", cache_dir, exc)

    return build_reference_eval_cache(
        reference_model=reference_model,
        dataset=dataset,
        eval_loader=eval_loader,
        compute_axis_mapping=compute_axis_mapping,
        mp=mp,
        cache_dir=cache_dir,
        metadata=metadata,
    )
