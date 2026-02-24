# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
import dataclasses
import json
import logging
import os
import warnings
from collections import defaultdict
from typing import Callable, Generic, Mapping, Optional, Sequence, TypeVar

import equinox as eqx
import fsspec
import jax
import jax.numpy as jnp
import jmp
import numpy as np
from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec as P
from jaxtyping import Array, Float, Int
from tqdm_loggable.auto import tqdm

import haliax as hax
from haliax.partitioning import ResourceMapping

import levanter.tracker
from levanter.callbacks import StepInfo
from levanter.data import AsyncDataset, DataLoader
from levanter.data.text.examples import GrugLmExample, named_lm_example_from_grug
from levanter.models.lm_model import LmExample, LmHeadModel
from levanter.utils.hf_utils import HfTokenizer, byte_length_of_token
from levanter.utils.logging import LoadingTimeTrackerIterator
from levanter.utils.stat_utils import RunningMean
from levanter.utils.tree_utils import inference_mode


logger = logging.getLogger(__name__)


T = TypeVar("T")
M = TypeVar("M")
Ex = TypeVar("Ex")
LmEvalExample = LmExample | GrugLmExample
LossFnOutput = tuple[jax.Array, jax.Array, jax.Array]
TagArray = Int[Array, "tag"]
BatchedTagArray = Int[Array, "... tag"]


@dataclasses.dataclass
class EvalResult:
    micro_avg_loss: float  # per token across all datasets
    macro_avg_loss: float  # average of per-dataset average losses
    tag_macro_losses: dict[str, float]  # per tag average-per-token loss
    tag_micro_losses: dict[str, float]  # per tag total loss, for "parent" tags
    total_eval_loading_time: float
    micro_bpb: Optional[float] = None
    macro_bpb: Optional[float] = None
    tag_macro_bpb: Optional[dict[str, float]] = None
    tag_micro_bpb: Optional[dict[str, float]] = None


class DomainTaggedDataset(AsyncDataset[tuple[T, TagArray]]):
    """Holds multiple datasets, each with its own domain tag. Also indexes the tags to enable easier aggregation."""

    tag_index: Mapping[str, int]

    @property
    def tags(self):
        return self.tag_to_index.keys()

    def __init__(
        self, datasets: Sequence[tuple[AsyncDataset[T], Sequence[str]]], max_examples_per_dataset: Optional[int] = None
    ):
        super().__init__()
        self.datasets = []
        self._max_examples_per_dataset = max_examples_per_dataset

        tag_index: dict[str, int] = {}
        for i, (dataset, tags) in enumerate(datasets):
            if not tags and len(datasets) > 1:
                warnings.warn("Dataset has no tags. Giving it an index")
                tags = [f"domain_{i}"]
            for tag in tags:
                if tag not in tag_index:
                    tag_index[tag] = len(tag_index)

            if self._max_examples_per_dataset:
                dataset = dataset.take(self._max_examples_per_dataset)

            self.datasets.append((dataset, tags))

        self.tag_to_index = tag_index
        self.num_tags = len(self.tag_to_index)
        self._tag_arrays = self._compute_tag_arrays()
        self._offsets: Optional[np.ndarray] = None

    async def _get_offsets(self) -> np.ndarray:
        if self._offsets is None:
            lengths = await asyncio.gather(*[dataset.async_len() for dataset, _ in self.datasets])
            if self._max_examples_per_dataset is not None:
                lengths = [min(length, self._max_examples_per_dataset) for length in lengths]
            self._offsets = np.cumsum([0] + lengths)

        return self._offsets  # type: ignore

    def _compute_tag_arrays(self):
        tag_arrays: list[TagArray] = []
        for dataset, tags in self.datasets:
            indexed = [self.tag_to_index[tag] for tag in tags]
            tag_array = np.zeros(self.num_tags, dtype=np.int32)
            tag_array[indexed] = 1
            tag_arrays.append(jnp.asarray(tag_array))
        return tag_arrays

    async def async_len(self) -> int:
        return int((await self._get_offsets())[-1])

    async def getitem_async(self, index: int) -> tuple[T, TagArray]:
        offsets = await self._get_offsets()
        dataset_index = np.searchsorted(offsets, index, side="right") - 1
        offset = offsets[dataset_index]
        dataset, tags = self.datasets[dataset_index]
        return await dataset.getitem_async(int(index - offset)), self._tag_arrays[dataset_index]

    async def get_batch(self, indices: Sequence[int]) -> Sequence[tuple[T, TagArray]]:
        # Chatgpt wrote this. pretty sure it's correct
        offsets = await self._get_offsets()
        original_order = np.argsort(indices)
        sorted_indices = np.array(indices)[original_order]
        dataset_indices = (np.searchsorted(offsets, sorted_indices, side="right") - 1).tolist()

        # Group indices by the dataset they belong to
        grouped_indices = defaultdict(list)
        for idx, dataset_index in zip(sorted_indices, dataset_indices):
            grouped_indices[dataset_index].append(int(idx - offsets[dataset_index]))

        # Retrieve the batch for each group
        batch_futures: list = []
        for dataset_index, dataset_indices in grouped_indices.items():
            dataset, tags = self.datasets[dataset_index]
            dataset_batch = dataset.get_batch(dataset_indices)
            batch_futures.append(dataset_batch)

        batch_groups = await asyncio.gather(*batch_futures)
        batch = []
        for dataset_index, dataset_batch in zip(grouped_indices.keys(), batch_groups):
            batch.extend([(item, self._tag_arrays[dataset_index]) for item in dataset_batch])

        # Reorder the batch to match the original order of indices
        batch = [batch[i] for i in np.argsort(original_order)]

        return batch

    def is_finite(self) -> bool:
        return all(dataset.is_finite() for dataset, _ in self.datasets)


def _join_prefix(prefix: str, tag: str) -> str:
    if prefix:
        return f"{prefix}/{tag}"
    return tag


def _ensure_named_lm_example(batch: LmEvalExample, *, EvalBatch: hax.Axis, model_pos: hax.Axis) -> LmExample:
    if isinstance(batch, LmExample):
        return batch
    if not isinstance(batch, GrugLmExample):
        raise TypeError(f"Unsupported eval batch type: {type(batch)}")

    if batch.tokens.ndim == 1:
        Pos = model_pos.resize(batch.tokens.shape[0])
        return named_lm_example_from_grug(batch, Pos=Pos)
    if batch.tokens.ndim == 2:
        Pos = model_pos.resize(batch.tokens.shape[1])
        return named_lm_example_from_grug(batch, Pos=Pos, batch_axis=EvalBatch)

    raise ValueError(f"GrugLmExample tokens must be rank-1 or rank-2 for eval, got rank={batch.tokens.ndim}")


def _default_lm_eval_loss_fn(
    model: LmHeadModel,
    batch: LmEvalExample,
    *,
    EvalBatch: hax.Axis,
    mp: jmp.Policy | None,
    axis_mapping: ResourceMapping | None,
) -> LossFnOutput:
    model = inference_mode(model, True)
    named_batch = _ensure_named_lm_example(batch, EvalBatch=EvalBatch, model_pos=model.Pos)
    if mp is not None:
        model = mp.cast_to_compute(model)
    if axis_mapping is not None:
        with hax.axis_mapping(axis_mapping):
            per_pos_loss = model.compute_next_token_loss(named_batch, reduction=None, reduction_axis=()).array
    else:
        per_pos_loss = model.compute_next_token_loss(named_batch, reduction=None, reduction_axis=()).array
    per_pos_weight = named_batch.loss_weight.array
    per_pos_token_id = jnp.roll(named_batch.tokens.array, -1, axis=-1)
    return per_pos_loss, per_pos_weight, per_pos_token_id


def cb_tagged_lm_evaluate(
    EvalBatch: hax.Axis,
    tagged_eval_sets: Sequence[tuple[AsyncDataset[LmEvalExample], Sequence[str]]],
    tokenizer: Optional[HfTokenizer] = None,
    device_mesh: Optional[Mesh] = None,
    axis_mapping: ResourceMapping | None = None,
    max_examples_per_dataset: Optional[int] = None,
    eval_current: bool = True,
    eval_ema: bool = True,
    prefix: str = "eval",
    mp: jmp.Policy = None,
    checkpoint_path: Optional[str] = None,
    loss_fn: Callable[[LmHeadModel, LmEvalExample], LossFnOutput] | None = None,
) -> Callable[[StepInfo], None]:
    """
    Evaluates multiple tagged datasets using a given evaluation function.
    Scores for each tag are aggregated and logged separately, as well as getting
    an overall score.

    Tags can be hierarchical, with "/" as a separator. We log both a micro and macro average loss
    for each tag.

    !!! note

        The evaluator loss callback should produce per-position arrays with shape `[EvalBatch, Token]`:
        `(per_pos_loss, per_pos_weight, per_pos_token_id)`.

    Args:
        EvalBatch: The axis for the evaluation batch (mostly for the batch size)
        tagged_eval_sets: A list of datasets, each with its own domain tag
        tokenizer: The tokenizer to use for bits-per-byte evaluation (optional)
        device_mesh: The mesh to use for evaluation
        axis_mapping: The axis mapping to use for evaluation
        max_examples_per_dataset: The maximum number of examples to use from each dataset
        prefix: The prefix to use for logging the losses
        eval_current: Whether to evaluate the model's current parameters
        eval_ema: Whether to evaluate the EMA model (or other model averaged model)
        checkpoint_path: If provided, write eval metrics to a JSONL file in this directory
    """

    if loss_fn is None:

        def loss_fn(model: LmHeadModel, batch: LmEvalExample) -> LossFnOutput:
            return _default_lm_eval_loss_fn(model, batch, EvalBatch=EvalBatch, mp=mp, axis_mapping=axis_mapping)

    evaluator = TaggedEvaluator(
        EvalBatch=EvalBatch,
        tagged_eval_sets=tagged_eval_sets,
        loss_fn=loss_fn,
        tokenizer=tokenizer,
        device_mesh=device_mesh,
        axis_mapping=axis_mapping,
        max_examples_per_dataset=max_examples_per_dataset,
    )

    if not eval_current and not eval_ema:
        raise ValueError("At least one of eval_current or eval_ema should be True")

    def eval_callback(step: StepInfo):
        step_count = step.step
        metrics_to_write = {}

        if eval_current:
            log_dict = eval_model(evaluator, step.model, prefix=prefix)
            levanter.tracker.log(log_dict, step=step_count)
            metrics_to_write.update(log_dict)

        if not eval_current and step.state.model_averaging is None:
            raise ValueError("Cannot evaluate EMA model without model averaging, but you only want to evaluate EMA")

        if eval_ema and step.state.model_averaging is not None:
            log_dict = eval_model(evaluator, step.eval_model, prefix=_join_prefix(prefix, "ema"))
            levanter.tracker.log(log_dict, step=step_count)
            metrics_to_write.update(log_dict)

        # Write metrics to file if checkpoint_path is provided
        if checkpoint_path is not None and metrics_to_write:
            metrics_file = os.path.join(checkpoint_path, "eval_metrics.jsonl")
            fs, _, _ = fsspec.get_fs_token_paths(metrics_file)
            fs.makedirs(checkpoint_path, exist_ok=True)

            if fs.exists(metrics_file):
                with fs.open(metrics_file, "r") as f:
                    content = f.read()
            else:
                content = ""

            with fs.open(metrics_file, "w") as f:
                # Convert numpy/jax floats to Python floats for JSON serialization
                serializable_metrics = {
                    k: float(v) if isinstance(v, (np.floating, jnp.floating)) else v
                    for k, v in metrics_to_write.items()
                }
                record = {"step": int(step_count), **serializable_metrics}
                content += json.dumps(record, sort_keys=True) + "\n"
                f.write(content)

        return

    return eval_callback


def eval_model(evaluator, model, prefix: str = "") -> dict[str, float]:
    with levanter.tracker.capture_time() as time_fn:
        result = evaluator.evaluate(model)
    log_dict = construct_log_dict(evaluator, result, time_fn(), prefix=prefix)
    return log_dict


def construct_log_dict(evaluator, eval_result, total_time, prefix):
    tokenizer = evaluator.tokenizer
    log_dict = {
        # log micro average as just "loss"
        _join_prefix(prefix, "loss"): eval_result.micro_avg_loss,
        _join_prefix(prefix, "loading_time"): eval_result.total_eval_loading_time,
        _join_prefix(prefix, "total_time"): total_time,
    }
    logger.info(f"{prefix} loss: {eval_result.micro_avg_loss:.3f}")
    has_tags = len(evaluator.dataset.tag_to_index) > 1  # 1 tag means there's no difference between micro and macro
    if has_tags:
        log_dict[_join_prefix(prefix, "macro_loss")] = eval_result.macro_avg_loss

        for tag, loss in eval_result.tag_macro_losses.items():
            # don't log leaf tag macro losses because it doesn't mean anything different than micro loss
            if tag in evaluator.dataset.tag_to_index:
                continue
            if not tag:
                continue
            log_dict[_join_prefix(prefix, tag) + "/macro_loss"] = loss
            logger.info(f"{tag} macro loss: {loss:.3f}")
    for tag, loss in eval_result.tag_micro_losses.items():
        if not tag:
            continue
        if tag in evaluator.dataset.tag_to_index:
            log_dict[_join_prefix(prefix, tag) + "/loss"] = loss
            logger.info(f"{tag} loss: {loss:.3f}")
        else:
            log_dict[_join_prefix(prefix, tag) + "/micro_loss"] = loss
            logger.info(f"{tag} micro loss: {loss:.3f}")
    if tokenizer is not None:
        log_dict[_join_prefix(prefix, "bpb")] = eval_result.micro_bpb
        if has_tags:
            log_dict[_join_prefix(prefix, "macro_bpb")] = eval_result.macro_bpb
        for tag, bpb in eval_result.tag_micro_bpb.items():
            log_dict[_join_prefix(prefix, tag) + "/bpb"] = bpb

        if has_tags:
            for tag, bpb in eval_result.tag_macro_bpb.items():
                log_dict[_join_prefix(prefix, tag) + "/macro_bpb"] = bpb
    return log_dict


class TaggedEvaluator(Generic[Ex, M]):
    loss_fn: Callable[[M, Ex], LossFnOutput]

    def __init__(
        self,
        EvalBatch: hax.Axis,
        tagged_eval_sets: Sequence[tuple[AsyncDataset[Ex], Sequence[str]]],
        loss_fn: Callable[[M, Ex], LossFnOutput],
        tokenizer: Optional[HfTokenizer] = None,
        device_mesh=None,
        axis_mapping=None,
        max_examples_per_dataset=None,
    ):
        self.loss_fn = loss_fn
        self.dataset = DomainTaggedDataset(tagged_eval_sets, max_examples_per_dataset)
        self.loader = DataLoader(
            self.dataset.as_async_dataset(),
            EvalBatch,
            max_buffered_batches=100,
            mesh=device_mesh,
            axis_resources=axis_mapping,
        )
        self.device_mesh = device_mesh
        self.tokenizer = tokenizer
        self.axis_mapping = axis_mapping
        self.per_pos_out_sharding = None
        if device_mesh is not None and axis_mapping is not None:
            batch_axis_resource = axis_mapping.get(EvalBatch.name, axis_mapping.get("batch"))
            if batch_axis_resource is not None and self._axis_resource_is_explicit(device_mesh, batch_axis_resource):
                self.per_pos_out_sharding = NamedSharding(device_mesh, P(batch_axis_resource, None))

        self.bytes_per_token = self._calculate_bytes_per_token_type(tokenizer)
        self.hierarchy = self._construct_tag_hierarchy()
        self.accum_for_batch = self._make_accum_for_batch()

    @staticmethod
    def _flatten_axis_resource(axis_resource) -> tuple[str, ...]:
        if axis_resource is None:
            return ()
        if isinstance(axis_resource, str):
            return (axis_resource,)
        if isinstance(axis_resource, tuple):
            names: list[str] = []
            for axis in axis_resource:
                names.extend(TaggedEvaluator._flatten_axis_resource(axis))
            return tuple(names)
        return ()

    @staticmethod
    def _axis_resource_is_explicit(mesh: Mesh, axis_resource) -> bool:
        axis_types = dict(zip(mesh.axis_names, mesh.axis_types, strict=True))
        axis_names = TaggedEvaluator._flatten_axis_resource(axis_resource)
        if len(axis_names) == 0:
            return False
        for axis_name in axis_names:
            axis_type = axis_types.get(axis_name)
            if axis_type is None or axis_type != AxisType.Explicit:
                return False
        return True

    def _make_accum_for_batch(self) -> Callable[[M, "_EvalRunningMeans", Ex, BatchedTagArray], "_EvalRunningMeans"]:
        bytes_per_token = self.bytes_per_token
        log2e = jnp.log2(jnp.e)
        per_tag_out_sharding = None if self.device_mesh is None else NamedSharding(self.device_mesh, P(None))
        per_pos_out_sharding = self.per_pos_out_sharding

        @hax.named_jit
        def accum_for_batch(model: M, state: _EvalRunningMeans, batch: Ex, tags: BatchedTagArray):
            losses, weights, token_ids = self.loss_fn(model, batch)
            weighted_loss = losses * weights  # b t
            this_loss = jnp.sum(weighted_loss)  # scalar
            this_weights = jnp.sum(weights)  # scalar

            if losses.ndim != 2 or weights.ndim != 2 or token_ids.ndim != 2 or tags.ndim != 2:
                raise ValueError(
                    f"Expected batched eval tensors with rank 2, got losses={losses.ndim}, "
                    f"weights={weights.ndim}, token_ids={token_ids.ndim}, tags={tags.ndim}"
                )
            if per_tag_out_sharding is None:
                this_weights_per_tag = jnp.einsum("bt,bk->k", weights, tags)
                this_loss_per_tag = jnp.einsum("bt,bk->k", weighted_loss, tags)
            else:
                this_weights_per_tag = jnp.einsum("bt,bk->k", weights, tags, out_sharding=per_tag_out_sharding)
                this_loss_per_tag = jnp.einsum("bt,bk->k", weighted_loss, tags, out_sharding=per_tag_out_sharding)

            mean = state.token_avg_loss.add(this_loss / jnp.maximum(this_weights, 1.0), this_weights)
            state = dataclasses.replace(state, token_avg_loss=mean)

            if len(self.dataset.tag_to_index) > 0:
                nonzero_token_mask = this_weights_per_tag > 0
                safe_mean = jnp.where(nonzero_token_mask, this_loss_per_tag / this_weights_per_tag, 0.0)
                mean_per_tag = state.loss_per_tag.add(safe_mean, this_weights_per_tag)
                state = dataclasses.replace(state, loss_per_tag=mean_per_tag)

            if bytes_per_token is not None:
                if per_pos_out_sharding is None:
                    bytes_per_pos = bytes_per_token[token_ids]
                else:
                    bytes_per_pos = bytes_per_token.at[token_ids].get(out_sharding=per_pos_out_sharding)
                this_bytes = jnp.sum(bytes_per_pos * weights)
                if per_tag_out_sharding is None:
                    bytes_per_tag = jnp.einsum("bt,bt,bk->k", bytes_per_pos, weights, tags)
                else:
                    bytes_per_tag = jnp.einsum(
                        "bt,bt,bk->k", bytes_per_pos, weights, tags, out_sharding=per_tag_out_sharding
                    )

                bpb = this_loss / jnp.maximum(this_bytes, 1.0) * log2e
                bpb_per_tag = this_loss_per_tag / jnp.maximum(bytes_per_tag, 1.0) * log2e

                bpb_mean = state.bpb.add(bpb, this_weights)
                state = dataclasses.replace(state, bpb=bpb_mean)
                if len(self.dataset.tag_to_index) > 0:
                    bpb_per_tag_mean = state.bpb_per_tag.add(bpb_per_tag, this_weights_per_tag)
                    state = dataclasses.replace(state, bpb_per_tag=bpb_per_tag_mean)

            return state

        return accum_for_batch

    def evaluate(self, model: M) -> EvalResult:
        total_loss = jnp.zeros((), dtype=jnp.float32)
        mean_losses_per_tag = jnp.zeros((self.dataset.num_tags,), dtype=jnp.float32)

        state = _EvalRunningMeans.zeros_like(total_loss, mean_losses_per_tag)
        del total_loss, mean_losses_per_tag
        state = hax.shard(state)

        iterator = LoadingTimeTrackerIterator(self.loader)

        for batch, tags in tqdm(iterator, "eval", total=len(self.loader)):
            state = self.accum_for_batch(model, state, batch, tags)

        micro_avg_loss = state.token_avg_loss.mean.item()
        tag_avg_loss = state.loss_per_tag.mean
        macro_avg_loss = jnp.mean(tag_avg_loss).item()

        if self.bytes_per_token is not None:
            micro_bpb = state.bpb.mean.item()
            tag_avg_bpb = state.bpb_per_tag.mean
            macro_avg_bpb = jnp.mean(tag_avg_bpb).item()
        else:
            micro_bpb = None
            macro_avg_bpb = None

        tag_macro_loss: dict[str, float] = {}
        tag_micro_loss: dict[str, float] = {}
        tag_macro_bpb: dict[str, float] = {}
        tag_micro_bpb: dict[str, float] = {}

        mean_loss_per_tag_cpu = np.array(state.loss_per_tag.mean)
        total_tokens_per_tag_cpu = np.array(state.loss_per_tag.total)
        mean_bits_per_tag_cpu = np.array(state.bpb_per_tag.mean)
        total_bytes_per_tag_cpu = np.array(state.bpb_per_tag.total)

        for parent, children in self.hierarchy.items():
            mask = np.zeros(self.dataset.num_tags, dtype=bool)
            mask[children] = 1
            mask = mask & (total_tokens_per_tag_cpu > 0)

            tag_macro_loss[parent] = np.mean(mean_loss_per_tag_cpu, where=mask)
            tag_micro_loss[parent] = np.average(mean_loss_per_tag_cpu, weights=total_tokens_per_tag_cpu * mask)

            if self.bytes_per_token is not None:
                tag_macro_bpb[parent] = np.mean(mean_bits_per_tag_cpu, where=mask)
                tag_micro_bpb[parent] = np.average(mean_bits_per_tag_cpu, weights=total_bytes_per_tag_cpu * mask)

        for tag, index in self.dataset.tag_to_index.items():
            tag_micro_loss[tag] = float(mean_loss_per_tag_cpu[index])
            if self.bytes_per_token is not None:
                tag_micro_bpb[tag] = float(mean_bits_per_tag_cpu[index])

        return EvalResult(
            micro_avg_loss,
            macro_avg_loss,
            tag_macro_loss,
            tag_micro_loss,
            iterator.total_time,
            micro_bpb,
            macro_avg_bpb,
            tag_macro_bpb,
            tag_micro_bpb,
        )

    def _construct_tag_hierarchy(self) -> dict[str, list[int]]:
        hierarchy: dict[str, list[int]] = {}
        for tag, index in self.dataset.tag_to_index.items():
            parts = tag.split("/")
            for i in range(1, len(parts)):
                parent = "/".join(parts[:i])
                assert parent != tag
                if parent not in hierarchy:
                    hierarchy[parent] = []
                hierarchy[parent].append(index)
        return hierarchy

    def _calculate_bytes_per_token_type(self, tokenizer: HfTokenizer) -> Optional[Int[Array, "vocab"]]:
        if tokenizer is None:
            return None
        else:
            # calculate the number of bytes in each token
            vocab_size = len(tokenizer.get_vocab())
            bytes = np.ndarray((vocab_size,), dtype=np.int32)

            for i in range(vocab_size):
                bytes[i] = byte_length_of_token(tokenizer, i)

            return jnp.array(bytes)


class _EvalRunningMeans(eqx.Module):
    token_avg_loss: RunningMean  # average loss averaged over all tokens
    loss_per_tag: RunningMean  # average loss per tag
    bpb: RunningMean  # bits per byte averaged over all tokens
    bpb_per_tag: RunningMean  # bits per byte per tag

    @staticmethod
    def zeros_like(total: Float[Array, "..."], per_tag: Float[Array, "tag"]) -> "_EvalRunningMeans":
        z = RunningMean.zeros_like(total)
        per_tag = RunningMean.zeros_like(per_tag)
        return _EvalRunningMeans(z, per_tag, z, per_tag)
