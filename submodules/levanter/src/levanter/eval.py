import asyncio
import dataclasses
import logging
import warnings
from collections import defaultdict
from typing import Callable, Mapping, Optional, Sequence, TypeVar

import equinox as eqx
import jax.numpy as jnp
import jmp
import numpy as np
from jax.sharding import Mesh
from tqdm_loggable.auto import tqdm

import haliax as hax
from haliax.partitioning import ResourceMapping

import levanter.tracker
from levanter.callbacks import StepInfo
from levanter.data import AsyncDataset, DataLoader
from levanter.models.lm_model import LmExample, LmHeadModel, compute_next_token_loss
from levanter.utils.hf_utils import HfTokenizer, byte_length_of_token
from levanter.utils.logging import LoadingTimeTrackerIterator
from levanter.utils.stat_utils import Arrayish, RunningMean
from levanter.utils.tree_utils import inference_mode


logger = logging.getLogger(__name__)


T = TypeVar("T")
M = TypeVar("M")


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


# This class doesn't try to be async or work with incomplete datasets, because it's eval


class DomainTaggedDataset(AsyncDataset[tuple[T, hax.NamedArray]]):
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
        self.Tag = hax.Axis("tag", len(self.tag_to_index))
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
        tag_arrays = []
        for dataset, tags in self.datasets:
            indexed = [self.tag_to_index[tag] for tag in tags]
            tags = np.zeros(self.Tag.size, dtype=np.int32)
            tags[indexed] = 1
            tags = hax.named(tags, self.Tag)

            tag_arrays.append(tags)
        return tag_arrays

    async def async_len(self) -> int:
        return int((await self._get_offsets())[-1])

    async def getitem_async(self, item: int) -> tuple[T, hax.NamedArray]:
        offsets = await self._get_offsets()
        dataset_index = np.searchsorted(offsets, item, side="right") - 1
        offset = offsets[dataset_index]
        dataset, tags = self.datasets[dataset_index]
        return await dataset.getitem_async(int(item - offset)), self._tag_arrays[dataset_index]

    async def get_batch(self, indices: Sequence[int]) -> Sequence[tuple[T, hax.NamedArray]]:
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

    async def final_length_is_known(self) -> bool:
        return all(await asyncio.gather(*[dataset.final_length_is_known() for dataset, _ in self.datasets]))

    def is_finite(self) -> bool:
        return all(dataset.is_finite() for dataset, _ in self.datasets)

    async def current_len(self) -> Optional[int]:
        # We currently require all datasets to be finished before we do anything with this dataset, so...
        return await self.async_len()


def _join_prefix(prefix: str, tag: str) -> str:
    if prefix:
        return f"{prefix}/{tag}"
    return tag


def cb_tagged_lm_evaluate(
    EvalBatch: hax.Axis,
    tagged_eval_sets: Sequence[tuple[AsyncDataset[LmExample], Sequence[str]]],
    tokenizer: Optional[HfTokenizer] = None,
    device_mesh: Optional[Mesh] = None,
    axis_mapping: ResourceMapping = None,
    max_examples_per_dataset: Optional[int] = None,
    eval_current: bool = True,
    eval_ema: bool = True,
    prefix: str = "eval",
    mp: jmp.Policy = None,
) -> Callable[[StepInfo], None]:
    """
    Evaluates multiple tagged datasets using a given evaluation function.
    Scores for each tag are aggregated and logged separately, as well as getting
    an overall score.

    Tags can be hierarchical, with "/" as a separator. We log both a micro and macro average loss
    for each tag.

    !!! note

        loss_fn should return *per-token* loss (shape [EvalBatch, Token])

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
    """

    evaluator = TaggedEvaluator(
        EvalBatch, tagged_eval_sets, tokenizer, device_mesh, axis_mapping, max_examples_per_dataset, mp=mp
    )

    if not eval_current and not eval_ema:
        raise ValueError("At least one of eval_current or eval_ema should be True")

    def eval_callback(step: StepInfo):
        step_count = step.step

        if eval_current:
            log_dict = eval_model(evaluator, step.model, prefix=prefix)
            levanter.tracker.log(log_dict, step=step_count)

        if not eval_current and step.state.model_averaging is None:
            raise ValueError("Cannot evaluate EMA model without model averaging, but you only want to evaluate EMA")

        if eval_ema and step.state.model_averaging is not None:
            log_dict = eval_model(evaluator, step.eval_model, prefix=_join_prefix(prefix, "ema"))
            levanter.tracker.log(log_dict, step=step_count)

        return

    return eval_callback


def eval_model(evaluator: "TaggedEvaluator", model: LmHeadModel, prefix: str = "") -> dict[str, float]:
    with levanter.tracker.capture_time() as time_fn:
        result = evaluator.evaluate(model)
    log_dict = _construct_log_dict(evaluator, result, time_fn(), prefix=prefix)
    return log_dict


def _construct_log_dict(evaluator, eval_result, total_time, prefix):
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


class TaggedEvaluator:
    """
    Evaluates multiple tagged datasets using a given evaluation function.
    Scores for each tag are aggregated and logged separately, as well as getting an overall score.

    TaggedEvaluator computes both log-perplexity and bits-per-byte for each tag, if a tokenizer is provided.

    Tags are arranged hierarchically with "/" as separator, and we log both a micro and macro average loss
    for each tag.

    """

    def __init__(
        self,
        EvalBatch: hax.Axis,
        tagged_eval_sets: Sequence[tuple[AsyncDataset, Sequence[str]]],
        tokenizer: Optional[HfTokenizer] = None,
        device_mesh=None,
        axis_mapping=None,
        max_examples_per_dataset=None,
        mp: Optional[jmp.Policy] = None,
    ):
        self.EvalBatch = EvalBatch
        self.dataset = DomainTaggedDataset(tagged_eval_sets, max_examples_per_dataset)
        self.loader = DataLoader(
            self.dataset.as_async_dataset(),
            EvalBatch,
            max_buffered_batches=100,
            mesh=device_mesh,
            axis_resources=axis_mapping,
        )
        self.mp = mp
        self.tokenizer = tokenizer
        self.bytes_per_token = self._calculate_bytes_per_token_type(tokenizer)

        # tags are arranged hierarchically with "/" as separator. We want to log the average loss for each tag.
        hierarchy: dict[str, list[int]] = {}
        for tag, index in self.dataset.tag_to_index.items():
            parts = tag.split("/")
            for i in range(1, len(parts)):
                parent = "/".join(parts[:i])
                assert parent != tag
                if parent not in hierarchy:
                    hierarchy[parent] = []
                hierarchy[parent].append(index)

        self.hierarchy = hierarchy

        @hax.named_jit
        def accum_for_batch(m: LmHeadModel, state: _EvalRunningMeans, batch: LmExample, tags: hax.NamedArray):
            m = inference_mode(m, True)

            if self.mp is not None:
                m = self.mp.cast_to_compute(m)

            from contextlib import ExitStack

            context = ExitStack()

            with context:
                if axis_mapping is not None:
                    context.enter_context(hax.axis_mapping(axis_mapping))
                losses = compute_next_token_loss(m, batch, reduction=None, reduction_axis=())
                mask = batch.loss_mask  # [Batch, Pos]
                this_tokens = hax.sum(mask)
                this_loss = hax.einsum("->", losses, mask)  # to scalar

                # all the *_per_tag variables are [Tag]
                this_tokens_per_tag = hax.einsum("-> tag", mask, tags)
                this_loss_per_tag = hax.einsum("-> tag", mask, losses, tags)  # [Tag]

                mean = state.token_avg_loss.add(this_loss / this_tokens, this_tokens)
                state = dataclasses.replace(state, token_avg_loss=mean)

                if len(self.dataset.tag_to_index) > 0:
                    # careful: this_tokens_per_tag can be 0 if there are no tokens for that tag
                    safe_mean = hax.where(this_tokens_per_tag, this_loss_per_tag / this_tokens_per_tag, 0.0)
                    mean_per_tag = state.loss_per_tag.add(safe_mean, this_tokens_per_tag)
                    state = dataclasses.replace(state, loss_per_tag=mean_per_tag)

                if self.bytes_per_token is not None:
                    next_tokens = hax.roll(
                        batch.tokens, -1, m.Pos.name
                    )  # [Batch, Pos], rolled by 1 for next token task
                    bytes_per_pos = self.bytes_per_token.take("vocab", next_tokens)  # [Batch, Pos]
                    bytes_per_tag = hax.einsum("-> tag", mask, bytes_per_pos, tags)  # [Tag]
                    this_bytes = hax.einsum("->", bytes_per_pos, mask)  # Scalar

                    # log loss -> bits is log2(e) * loss
                    bpb_per_tag = this_loss_per_tag / hax.maximum(bytes_per_tag, 1) * jnp.log2(jnp.e)
                    bpb = this_loss / hax.maximum(this_bytes, 1) * jnp.log2(jnp.e)

                    bpb_mean = state.bpb.add(bpb, this_tokens)
                    state = dataclasses.replace(state, bpb=bpb_mean)
                    if len(self.dataset.tag_to_index) > 0:
                        bpb_per_tag_mean = state.bpb_per_tag.add(bpb_per_tag, this_tokens_per_tag)
                        state = dataclasses.replace(state, bpb_per_tag=bpb_per_tag_mean)

            return state

        self.accum_for_batch = accum_for_batch

    def evaluate(self, m: LmHeadModel):
        total_loss = jnp.zeros(())
        mean_losses_per_tag = hax.zeros(self.dataset.Tag, dtype=np.float32)

        state = _EvalRunningMeans.zeros_like(total_loss, mean_losses_per_tag)
        del total_loss, mean_losses_per_tag
        state = hax.shard(state)

        iterator = LoadingTimeTrackerIterator(self.loader)
        n = 0

        for batch, tags in tqdm(iterator, "eval", total=len(self.loader)):
            state = self.accum_for_batch(m, state, batch, tags)
            n += 1

        micro_avg_loss = state.token_avg_loss.mean.item()
        tag_avg_loss = state.loss_per_tag.mean

        # TODO: why do i have to jit this
        macro_avg_loss = hax.named_jit(lambda x: hax.mean(x).array)(tag_avg_loss).item()

        if self.bytes_per_token is not None:
            micro_bpb = state.bpb.mean.item()
            tag_avg_bpb = state.bpb_per_tag.mean
            macro_avg_bpb = hax.named_jit(lambda x: hax.mean(x).array)(tag_avg_bpb).item()
        else:
            micro_bpb = None
            macro_avg_bpb = None

        tag_macro_loss: dict[str, float] = {}
        tag_micro_loss: dict[str, float] = {}
        tag_macro_bpb: dict[str, float] = {}
        tag_micro_bpb: dict[str, float] = {}

        mean_loss_per_tag_cpu = np.array(state.loss_per_tag.mean.array)
        total_tokens_per_tag_cpu = np.array(state.loss_per_tag.mean.array)

        mean_bits_per_tag_cpu = np.array(state.bpb_per_tag.mean.array)
        total_bytes_per_tag_cpu = np.array(state.bpb_per_tag.mean.array)

        # add in the hierarchy
        for parent, children in self.hierarchy.items():
            mask = np.zeros(self.dataset.Tag.size, dtype=bool)
            mask[children] = 1
            assert total_tokens_per_tag_cpu.shape == mask.shape

            # don't consider tags with no tokens in macro average
            mask = mask & (total_tokens_per_tag_cpu > 0)

            # macro is the average of the averages
            tag_macro_loss[parent] = np.mean(mean_loss_per_tag_cpu, where=mask)
            # micro is the total loss for the parent tag
            # (average doesn't support where directly so we just 0 out the weights)
            tag_micro_loss[parent] = np.average(mean_loss_per_tag_cpu, weights=total_tokens_per_tag_cpu * mask)

            if self.bytes_per_token is not None:
                tag_macro_bpb[parent] = np.mean(mean_bits_per_tag_cpu, where=mask)
                tag_micro_bpb[parent] = np.average(mean_bits_per_tag_cpu, weights=total_bytes_per_tag_cpu * mask)

        for tag, index in self.dataset.tag_to_index.items():
            tag_micro_loss[tag] = float(mean_loss_per_tag_cpu[index])
            # no macro loss for the leaf tags

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

    def _calculate_bytes_per_token_type(self, tokenizer: HfTokenizer) -> Optional[hax.NamedArray]:
        if tokenizer is None:
            return None
        else:
            # calculate the number of bytes in each token
            Vocab = hax.Axis("vocab", len(tokenizer.get_vocab()))
            bytes = np.ndarray((Vocab.size,), dtype=np.int32)

            for i in range(Vocab.size):
                bytes[i] = byte_length_of_token(tokenizer, i)

            return hax.named(jnp.array(bytes), Vocab)


class _EvalRunningMeans(eqx.Module):
    token_avg_loss: RunningMean  # average loss averaged over all tokens
    loss_per_tag: RunningMean  # average loss per tag
    bpb: RunningMean  # bits per byte averaged over all tokens
    bpb_per_tag: RunningMean  # bits per byte per tag

    @staticmethod
    def zeros_like(total: Arrayish, per_tag: Arrayish) -> "_EvalRunningMeans":
        z = RunningMean.zeros_like(total)
        per_tag = RunningMean.zeros_like(per_tag)
        return _EvalRunningMeans(z, per_tag, z, per_tag)
