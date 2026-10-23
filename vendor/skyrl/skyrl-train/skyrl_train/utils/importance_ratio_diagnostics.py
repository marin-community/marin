"""Importance-ratio and token probability-change diagnostics for policy training."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, fields
import math

import torch
from loguru import logger

from skyrl_train.tensor_math import LOG_PROB_DELTA_CLIP

# Absolute-position bucket width; the async RL dashboard reads pos_first256 and pos_last256 by name.
POSITION_WINDOW = 256


LOG_RATIO_TAIL_CAPACITY = 8192

# Thresholds for a token whose probability halved or doubled, and one that fell below 1e-5.
LOG_RATIO_DOUBLING = math.log(2)
LOG_RATIO_COLLAPSE = math.log(1e-5)

# Mismatch staleness buckets and each one's inclusive lower bound.
MISMATCH_STALENESS_BUCKETS = (
    ("staleness0", 0),
    ("staleness1", 1),
    ("staleness2", 2),
    ("staleness3", 3),
    ("staleness4-7", 4),
    ("staleness8+", 8),
)

LOG_RATIO_BASE_METRIC_KEYS = (
    "log_ratio_abs_mean",
    "log_ratio_abs_max",
    "n_tokens_dp_gt_1pct",
    "n_tokens_dp_gt_10pct",
    "n_tokens_dp_gt_50pct",
    "log_ratio_abs_p99",
    "log_ratio_diagnostics_failed",
)


def linear_quantiles(values: torch.Tensor, probabilities: tuple[float, ...]) -> list[float]:
    """Linear-interpolated quantiles of a finite 1-D tensor; torch.quantile refuses more than 2**24 values."""
    count = values.numel()
    positions = torch.tensor(probabilities, dtype=torch.float64) * (count - 1)
    lowest = int(positions.min().floor())
    threshold = values.kthvalue(lowest + 1).values
    greater = values[values > threshold].sort().values
    tail = torch.cat([threshold.expand(count - lowest - greater.numel()), greater]).double()
    lower = positions.floor()
    below = tail[(lower - lowest).long()]
    above = tail[(positions.ceil() - lowest).long()]
    return (below + (above - below) * (positions - lower)).tolist()


def ratio_statistics(delta: torch.Tensor, *, eps_clip_low: float = 0.2, eps_clip_high: float = 0.2) -> dict[str, float]:
    """Summarize per-token trainer-minus-sampler log-ratios over the finite tokens."""
    delta = delta.detach().double().reshape(-1)
    selected = delta.numel()
    finite = torch.isfinite(delta)
    if not finite.all():
        delta = delta[finite]
    count = delta.numel()
    result = {"selected_tokens": float(selected), "finite_tokens": float(count)}
    if selected:
        result["finite_fraction"] = count / selected
    if not count:
        return result
    absolute = delta.abs()
    quantiles = linear_quantiles(absolute, (0.95, 0.99, 0.999))
    maximum = delta.max()
    weights = (delta - maximum).exp()
    weight_sum = weights.sum()
    clipped_ratio = delta.clamp(-LOG_PROB_DELTA_CLIP, LOG_PROB_DELTA_CLIP).exp()
    lower_bound = math.log1p(-eps_clip_low) if eps_clip_low < 1 else -math.inf

    def fraction(selection: torch.Tensor) -> float:
        return selection.sum().item() / count

    mean = delta.mean().item()
    result.update(
        log_ratio_mean=mean,
        log_ratio_abs_mean=absolute.mean().item(),
        log_ratio_abs_p95=quantiles[0],
        log_ratio_abs_p99=quantiles[1],
        log_ratio_abs_p999=quantiles[2],
        log_ratio_abs_max=absolute.max().item(),
        lower_clip_pressure=fraction(delta < lower_bound),
        upper_clip_pressure=fraction(delta > math.log1p(eps_clip_high)),
        frac_outside_0_5_2=fraction(absolute > LOG_RATIO_DOUBLING),
        frac_below_1e_5=fraction(delta < LOG_RATIO_COLLAPSE),
        ess_fraction=(weight_sum.square() / (count * weights.square().sum())).item(),
        kl_k1=-mean,
        kl_k3=clipped_ratio.mean().item() - mean - 1,
        chi2=clipped_ratio.square().mean().item() - 1,
        log_ratio_mean_squared=delta.square().mean().item(),
    )
    return result


def _grouped_moments(values: torch.Tensor, groups: torch.Tensor, n_groups: int) -> dict[str, torch.Tensor]:
    """Per-group token counts and finite-token sums of delta, |delta| and the 2x indicator."""

    def total(weights: torch.Tensor | None = None) -> torch.Tensor:
        return torch.bincount(groups, weights, minlength=n_groups).double()

    finite = torch.isfinite(values)
    finite_values = torch.where(finite, values, 0)
    absolute = finite_values.abs()
    return {
        "selected": total(),
        "finite": total(finite.double()),
        "sum": total(finite_values),
        "abs_sum": total(absolute),
        "outside": total((absolute > LOG_RATIO_DOUBLING).double()),
    }


def _moment_metrics(totals: dict[str, float]) -> dict[str, float]:
    selected, finite = totals["selected"], totals["finite"]
    result = {"selected_tokens": selected, "finite_tokens": finite}
    if selected:
        result["finite_fraction"] = finite / selected
    if finite:
        result.update(
            log_ratio_mean=totals["sum"] / finite,
            log_ratio_abs_mean=totals["abs_sum"] / finite,
            frac_outside_0_5_2=totals["outside"] / finite,
        )
    return result


def mismatch_ratio_metrics(
    learner_logprobs: torch.Tensor,
    rollout_logprobs: torch.Tensor,
    loss_mask: torch.Tensor,
    rollout_staleness: torch.Tensor,
    *,
    eps_clip_low: float = 0.2,
    eps_clip_high: float = 0.2,
) -> dict[str, float]:
    """Trainer-minus-vLLM log-ratio statistics by per-token staleness bucket and response position."""
    mask = loss_mask.detach().cpu() > 0
    staleness = rollout_staleness.detach().cpu()
    bucket = torch.zeros(mask.shape, dtype=torch.uint8)
    for index, (_, lower) in enumerate(MISMATCH_STALENESS_BUCKETS):
        bucket[staleness >= lower] = index
    first, last = _position_masks(mask)
    # A token's group is its staleness bucket times four plus its position class:
    # 1 in the first window, 2 in the last window, 3 in both and 0 in neither.
    groups = torch.masked_select(4 * bucket + first.to(torch.uint8) + 2 * last.to(torch.uint8), mask).long()
    # CPU float64 and masking before subtraction avoid padded NaNs.
    values = (
        torch.masked_select(learner_logprobs.detach().cpu(), mask).double()
        - torch.masked_select(rollout_logprobs.detach().cpu(), mask).double()
    )
    grid = {
        key: total.view(len(MISMATCH_STALENESS_BUCKETS), 4)
        for key, total in _grouped_moments(values, groups, 4 * len(MISMATCH_STALENESS_BUCKETS)).items()
    }
    position_classes = {f"first{POSITION_WINDOW}": [1, 3], f"last{POSITION_WINDOW}": [2, 3], "middle": [0]}
    # Only these two buckets get the sorted statistics; the others take moments from the grid.
    staleness0 = groups < 4
    pooled = ratio_statistics(values, eps_clip_low=eps_clip_low, eps_clip_high=eps_clip_high)
    full_statistics = {
        "pooled": pooled,
        "staleness0": pooled
        if bool(staleness0.all())
        else ratio_statistics(values[staleness0], eps_clip_low=eps_clip_low, eps_clip_high=eps_clip_high),
    }
    bucket_rows = {"pooled": slice(None)}
    bucket_rows.update({name: slice(index, index + 1) for index, (name, _) in enumerate(MISMATCH_STALENESS_BUCKETS)})

    def moments(rows: slice, classes: list[int]) -> dict[str, float]:
        return _moment_metrics({key: total[rows][:, classes].sum().item() for key, total in grid.items()})

    result = {}
    for name, rows in bucket_rows.items():
        prefix = f"policy/mismatch/{name}/"
        metrics = {**(full_statistics.get(name) or moments(rows, [0, 1, 2, 3])), "missing_behavior": 0.0}
        result.update({prefix + key: value for key, value in metrics.items()})
        for position, classes in position_classes.items():
            stats = moments(rows, classes)
            for key in ("selected_tokens", "finite_tokens", "log_ratio_abs_mean", "frac_outside_0_5_2"):
                if key in stats:
                    result[f"{prefix}pos_{position}/{key}"] = stats[key]
    return result


def _position_masks(mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Masks of the first and last POSITION_WINDOW response positions; they overlap on short responses."""
    positions = torch.arange(mask.shape[1], device=mask.device).unsqueeze(0)
    return (positions < POSITION_WINDOW).expand_as(mask), positions >= mask.sum(-1, keepdim=True) - POSITION_WINDOW


_RATIO_EXTRA_KEYS = (
    "log_ratio_mean",
    "log_ratio_abs_p999",
    "log_ratio_frac_outside_0_5_2",
    "log_ratio_ess_fraction",
    "log_ratio_kl_k3",
    "log_ratio_chi2",
    "log_ratio_statistics_valid",
    "log_ratio_p999_valid",
    *(
        f"log_ratio_pos_{position}/log_ratio_abs_mean"
        for position in (f"first{POSITION_WINDOW}", f"last{POSITION_WINDOW}", "middle")
    ),
)


def _log_ratio_position_metric_keys(n_position_buckets: int) -> tuple[str, ...]:
    return tuple(f"log_ratio_abs_pos{i * (100 // n_position_buckets):02d}" for i in range(n_position_buckets))


@dataclass
class LogRatioAccumulator:
    """Tensor statistics accumulated across policy micro-batches."""

    abs_sum: torch.Tensor
    n_valid: torch.Tensor
    abs_max: torch.Tensor
    n_gt_1pct: torch.Tensor
    n_gt_10pct: torch.Tensor
    n_gt_50pct: torch.Tensor
    topk_abs: torch.Tensor
    bucket_sums: torch.Tensor
    bucket_counts: torch.Tensor
    moments: torch.Tensor  # delta, clipped exp(delta), clipped exp(2delta), outside2x
    maximum_delta: torch.Tensor
    shifted_weight_sum: torch.Tensor
    shifted_weight_square_sum: torch.Tensor
    top_per_mille: torch.Tensor
    position_sums: torch.Tensor
    position_counts: torch.Tensor


class LogRatioMonitor:
    """Accumulate a fixed-key log-ratio metric contract across microbatches."""

    def __init__(self, device: torch.device):
        self._accumulator = _empty_log_ratio_accumulator(device)
        self._failed = False

    def add(self, log_probs: torch.Tensor, old_log_probs: torch.Tensor, loss_mask: torch.Tensor) -> None:
        if self._failed:
            return
        try:
            partial = compute_log_ratio_partial(log_probs, old_log_probs, loss_mask)
            merge_log_ratio_partial(self._accumulator, partial)
        except Exception as error:
            logger.warning(f"Log-ratio diagnostics skipped after accumulation failed: {error!r}")
            self._failed = True

    def metrics(
        self,
        *,
        gather_fn: Callable[[torch.Tensor], list[torch.Tensor]] | None = None,
    ) -> dict[str, float]:
        accumulator = self._accumulator
        failed = self._failed
        if gather_fn is not None:
            # Every rank gathers, even after a local failure, so the collective stays matched.
            header, tail = pack_log_ratio_accumulator(accumulator, failed=failed)
            headers, tails = gather_fn(header), gather_fn(tail)
            accumulator, failed = pool_log_ratio_accumulators(headers, tails)
        if failed:
            return _failed_log_ratio_metrics()
        try:
            return finalize_log_ratio_metrics(accumulator)
        except Exception as error:
            logger.warning(f"Log-ratio diagnostics marked failed after finalization failed: {error!r}")
            return _failed_log_ratio_metrics()


def gather_ratio_tensor(tensor: torch.Tensor, *, group=None) -> list[torch.Tensor]:
    """Gather a fixed-size statistics tensor over the supplied ownership group."""
    if not torch.distributed.is_initialized():
        return [tensor]
    values = [torch.empty_like(tensor) for _ in range(torch.distributed.get_world_size(group))]
    torch.distributed.all_gather(values, tensor, group=group)
    return values


def pack_log_ratio_accumulator(accumulator, *, failed: bool = False):
    """Encode bounded sufficient statistics."""
    scalar_fields = [field.name for field in fields(accumulator) if field.name not in {"topk_abs", "top_per_mille"}]
    approximate_p99 = (
        accumulator.topk_abs.min() if accumulator.topk_abs.numel() else accumulator.abs_sum.new_tensor(math.inf)
    )
    header = torch.cat(
        [accumulator.abs_sum.new_tensor([float(failed)], dtype=torch.float64), approximate_p99.double().reshape(1)]
        + [getattr(accumulator, name).double().reshape(-1) for name in scalar_fields]
    )
    tail = accumulator.abs_sum.new_full((LOG_RATIO_TAIL_CAPACITY,), -math.inf, dtype=torch.float32)
    tail[: accumulator.top_per_mille.numel()] = accumulator.top_per_mille
    return header, tail


def pool_log_ratio_accumulators(headers, tails):
    """Merge rank payloads before computing any nonlinear scalar statistic."""
    if len(headers) != len(tails) or not headers:
        raise ValueError("Ratio reduction needs one header and tail per rank")
    pooled = _empty_log_ratio_accumulator(headers[0].device)
    scalar_fields = [field.name for field in fields(pooled) if field.name not in {"topk_abs", "top_per_mille"}]
    failed = False
    for header, tail in zip(headers, tails, strict=True):
        failed = failed or bool(header[0].item())
        partial = _empty_log_ratio_accumulator(header.device)
        offset = 2
        for name in scalar_fields:
            target = getattr(partial, name)
            target.copy_(header[offset : offset + target.numel()].reshape(target.shape))
            offset += target.numel()
        if offset != header.numel() or tail.numel() != LOG_RATIO_TAIL_CAPACITY:
            raise ValueError("Ratio reduction payload shape changed")
        partial.topk_abs = header[1:2].float() if torch.isfinite(header[1]) else partial.topk_abs
        partial.top_per_mille = tail[torch.isfinite(tail)]
        merge_log_ratio_partial(pooled, partial)
    return pooled, failed


def _failed_log_ratio_metrics() -> dict[str, float]:
    metrics = _log_ratio_diag_zero_metrics()
    metrics["log_ratio_diagnostics_failed"] = 1.0
    return metrics


def _log_ratio_diag_zero_metrics(n_position_buckets: int = 10) -> dict:
    """The full key set the diagnostic emits, with all values zero.

    Used as a fallback so every rank contributes identical keys to
    `strategy.all_reduce(status)` even if a rank's input is empty/all-padded
    or the helper raises. Mismatched keysets across ranks would deadlock the
    per-key NCCL all-reduce.
    """
    keys = LOG_RATIO_BASE_METRIC_KEYS + _log_ratio_position_metric_keys(n_position_buckets) + _RATIO_EXTRA_KEYS
    metrics = dict.fromkeys(keys, 0.0)
    return metrics


def _empty_log_ratio_accumulator(device, n_position_buckets: int = 10) -> LogRatioAccumulator:
    """Return a zeroed cross-micro-batch accumulator for log-ratio diagnostics.

    Each micro-batch contributes partial statistics. The last micro-batch finalizes
    them while preserving identical metric keysets across ranks.
    """
    return LogRatioAccumulator(
        abs_sum=torch.zeros((), device=device, dtype=torch.float32),
        n_valid=torch.zeros((), device=device, dtype=torch.float32),
        abs_max=torch.zeros((), device=device, dtype=torch.float32),
        n_gt_1pct=torch.zeros((), device=device, dtype=torch.float32),
        n_gt_10pct=torch.zeros((), device=device, dtype=torch.float32),
        n_gt_50pct=torch.zeros((), device=device, dtype=torch.float32),
        topk_abs=torch.zeros((0,), device=device, dtype=torch.float32),
        bucket_sums=torch.zeros(n_position_buckets, device=device, dtype=torch.float32),
        bucket_counts=torch.zeros(n_position_buckets, device=device, dtype=torch.float32),
        moments=torch.zeros(4, device=device, dtype=torch.float64),
        maximum_delta=torch.tensor(-math.inf, device=device, dtype=torch.float64),
        shifted_weight_sum=torch.zeros((), device=device, dtype=torch.float64),
        shifted_weight_square_sum=torch.zeros((), device=device, dtype=torch.float64),
        top_per_mille=torch.zeros((0,), device=device, dtype=torch.float32),
        position_sums=torch.zeros(3, device=device, dtype=torch.float64),
        position_counts=torch.zeros(3, device=device, dtype=torch.float64),
    )


def compute_log_ratio_partial(
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    loss_mask: torch.Tensor,
    n_position_buckets: int = 10,
) -> LogRatioAccumulator:
    """Compute mergeable log-ratio statistics for one policy micro-batch.

    `topk_abs` stores this micro-batch's top-1%-of-valid values. At finalize,
    the concatenated tensor across micro-batches is the global "top of top"
    sample; its min approximates p99.
    """
    device = log_probs.device

    if log_probs.numel() == 0:
        return _empty_log_ratio_accumulator(device, n_position_buckets)

    selected = loss_mask > 0
    delta = torch.zeros_like(log_probs, dtype=torch.float64)
    delta[selected] = log_probs.detach().double()[selected] - old_log_probs.detach().double()[selected]
    if not torch.isfinite(delta[selected]).all():
        raise ValueError("Nonfinite selected log-ratio token")
    abs_log_ratio = delta.abs().float()
    mask_f = selected.float()
    masked = abs_log_ratio * mask_f

    # Bound top-k by valid tokens so padding cannot contaminate short responses.
    n_valid_int = int(mask_f.sum().item())
    if n_valid_int == 0:
        return _empty_log_ratio_accumulator(device, n_position_buckets)

    # Topk over masked-aware flat. Sentinel ensures topk never picks a padded
    # position when k_local <= n_valid (always true by construction).
    sentinel = torch.tensor(-1.0e9, device=device, dtype=abs_log_ratio.dtype)
    flat = torch.where(mask_f.bool(), abs_log_ratio, sentinel).flatten()
    k_local = max(1, n_valid_int // 100)
    topk_abs = torch.topk(flat, k=min(k_local, flat.numel()), largest=True).values.float()

    # Per-position bucket sums via single scatter_add (no Python loop allocs).
    B, T = log_probs.shape
    seq_lens = mask_f.sum(dim=-1, keepdim=True).clamp(min=1)
    positions = torch.arange(T, device=device, dtype=torch.float32).unsqueeze(0).expand(B, T)
    buckets = (positions / seq_lens * n_position_buckets).clamp(0, n_position_buckets - 1).long()
    bsums = torch.zeros(n_position_buckets, device=device, dtype=torch.float32)
    bcounts = torch.zeros(n_position_buckets, device=device, dtype=torch.float32)
    bsums.scatter_add_(0, buckets.flatten(), masked.flatten())
    bcounts.scatter_add_(0, buckets.flatten(), mask_f.flatten())

    values = delta[selected]
    maximum = values.max()
    weights = (values - maximum).exp()
    first, last = _position_masks(selected)
    absolute_masks = torch.stack([first, last, ~first & ~last]) & selected.unsqueeze(0)
    clipped = values.clamp(-LOG_PROB_DELTA_CLIP, LOG_PROB_DELTA_CLIP)
    moments = torch.stack(
        [
            values.sum(),
            clipped.exp().sum(),
            (2 * clipped).exp().sum(),
            (values.abs() > LOG_RATIO_DOUBLING).sum().double(),
        ]
    )
    return LogRatioAccumulator(
        abs_sum=masked.sum().float(),
        n_valid=torch.as_tensor(float(n_valid_int), device=device, dtype=torch.float32),
        abs_max=masked.max().float(),
        n_gt_1pct=((abs_log_ratio > 0.01) * mask_f).sum().float(),
        n_gt_10pct=((abs_log_ratio > 0.10) * mask_f).sum().float(),
        n_gt_50pct=((abs_log_ratio > 0.50) * mask_f).sum().float(),
        topk_abs=topk_abs,
        bucket_sums=bsums,
        bucket_counts=bcounts,
        moments=moments,
        maximum_delta=maximum,
        shifted_weight_sum=weights.sum(),
        shifted_weight_square_sum=weights.square().sum(),
        top_per_mille=torch.topk(values.abs(), k=min(LOG_RATIO_TAIL_CAPACITY, n_valid_int)).values.float(),
        position_sums=(absolute_masks * delta.abs()).sum((1, 2)),
        position_counts=absolute_masks.sum((1, 2)).double(),
    )


def merge_log_ratio_partial(acc: LogRatioAccumulator, partial: LogRatioAccumulator) -> None:
    """In-place merge: additive for sums/counts, max for abs_max, concat for
    topk_abs. Mutates `acc`.
    """
    acc.abs_sum = acc.abs_sum + partial.abs_sum
    acc.n_valid = acc.n_valid + partial.n_valid
    acc.abs_max = torch.maximum(acc.abs_max, partial.abs_max)
    acc.n_gt_1pct = acc.n_gt_1pct + partial.n_gt_1pct
    acc.n_gt_10pct = acc.n_gt_10pct + partial.n_gt_10pct
    acc.n_gt_50pct = acc.n_gt_50pct + partial.n_gt_50pct
    acc.topk_abs = torch.cat([acc.topk_abs, partial.topk_abs])
    acc.bucket_sums = acc.bucket_sums + partial.bucket_sums
    acc.bucket_counts = acc.bucket_counts + partial.bucket_counts
    maximum = torch.maximum(acc.maximum_delta, partial.maximum_delta)
    safe_maximum = torch.where(torch.isfinite(maximum), maximum, maximum.new_zeros(()))
    old_scale, new_scale = (acc.maximum_delta - safe_maximum).exp(), (partial.maximum_delta - safe_maximum).exp()
    acc.shifted_weight_sum = acc.shifted_weight_sum * old_scale + partial.shifted_weight_sum * new_scale
    acc.shifted_weight_square_sum = (
        acc.shifted_weight_square_sum * old_scale.square() + partial.shifted_weight_square_sum * new_scale.square()
    )
    acc.maximum_delta = maximum
    acc.moments = acc.moments + partial.moments
    tail = torch.cat([acc.top_per_mille, partial.top_per_mille])
    acc.top_per_mille = torch.topk(tail, k=min(LOG_RATIO_TAIL_CAPACITY, tail.numel())).values
    acc.position_sums = acc.position_sums + partial.position_sums
    acc.position_counts = acc.position_counts + partial.position_counts


def finalize_log_ratio_metrics(acc: LogRatioAccumulator, n_position_buckets: int = 10) -> dict:
    """Reduce the accumulator to the public scalar metric dictionary.

    Returns the full keyset always (zeros where input was empty), so downstream
    per-key `all_reduce(status)` stays keyset-compatible across ranks.
    """
    device = acc.abs_sum.device

    n_valid_safe = acc.n_valid.clamp(min=1.0)
    abs_mean = acc.abs_sum / n_valid_safe
    abs_max = acc.abs_max

    # The minimum concatenated per-batch top-1% value approximates global p99.
    # Uneven micro-batch sizes introduce acceptable monitoring-grade bias.
    if acc.topk_abs.numel() > 0:
        abs_p99 = acc.topk_abs.min()
    else:
        abs_p99 = torch.zeros((), device=device, dtype=torch.float32)

    bucket_means = acc.bucket_sums / acc.bucket_counts.clamp(min=1.0)

    base_vals = (
        torch.stack(
            [
                abs_mean.float(),
                abs_max.float(),
                acc.n_gt_1pct.float(),
                acc.n_gt_10pct.float(),
                acc.n_gt_50pct.float(),
                abs_p99.float(),
                torch.zeros((), device=device, dtype=torch.float32),
            ]
        )
        .cpu()
        .tolist()
    )
    metrics = dict(zip(LOG_RATIO_BASE_METRIC_KEYS, base_vals, strict=True))

    bucket_vals = bucket_means.cpu().tolist()
    metrics.update(dict(zip(_log_ratio_position_metric_keys(n_position_buckets), bucket_vals, strict=True)))

    count = acc.n_valid.double().clamp(min=1)
    means = acc.moments / count
    ess = acc.shifted_weight_sum.square() / (count * acc.shifted_weight_square_sum).clamp(min=1e-300)
    # The p99.9 interpolates within the bounded top tail kept across micro-batches.
    rank_from_top = (int(acc.n_valid.item()) - 1) * 0.001
    upper = math.ceil(rank_from_top)
    p999_valid = 0 <= upper < acc.top_per_mille.numel()
    p999 = acc.n_valid.new_zeros(())
    if p999_valid:
        lower = math.floor(rank_from_top)
        fraction = rank_from_top - lower
        p999 = acc.top_per_mille[lower] * (1 - fraction) + acc.top_per_mille[upper] * fraction
    valid = (acc.n_valid > 0).double()
    extra = torch.stack(
        [
            means[0],
            p999,
            means[3],
            ess,
            (means[1] - means[0] - 1) * valid,
            (means[2] - 1) * valid,
            valid,
            valid.new_tensor(float(p999_valid)),
        ]
    )
    positions = acc.position_sums / acc.position_counts.clamp(min=1)
    values = torch.cat([extra, positions]).cpu().tolist()
    metrics.update(dict(zip(_RATIO_EXTRA_KEYS, values, strict=True)))
    return metrics
