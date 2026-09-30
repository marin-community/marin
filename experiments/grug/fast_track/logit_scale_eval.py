# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Logit-scale (softmax temperature) sweep and scale-invariant metrics over the tagged eval sets.

One pass over the data scores ``softmax(scale * z)`` for every ``scale`` in a grid, where ``z`` is the
model's final (soft-capped) logits, so ``scale = 1`` is the trained model and ``1 / scale`` is a softmax
temperature. The same pass records two metrics no rescaling can change: top-1 accuracy and the mean
``log2(1 + rank)`` of the observed token. Eval batches alternate between a *fit* half and a *report*
half; the best scale is chosen on the fit half and its loss is read off the report half, so the
temperature-matched loss is never optimized on the tokens it is reported on.

Used by the ladder trainer after training to compare a cooled-down model against the EMA of a
constant-LR run: if the two differ only by output temperature, their ``best/*/macro_loss`` agree
while ``unit/*/macro_loss`` (the trained scale) does not, and their top-1 / rank metrics agree too.
"""

from collections.abc import Callable, Sequence

import haliax as hax
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from levanter.data.text.examples import GrugLmExample, grug_lm_example_from_named
from levanter.eval import TaggedEvaluator
from levanter.models.lm_model import LmExample
from tqdm import tqdm

from experiments.grug.fast_track.model import Transformer, _batch_spec

# The eval-set parent tag the best scale is selected on (its fit-half macro loss).
SELECT_PARENT = "paloma"
# Metric families accumulated per tag; ``loss`` carries one row per scale.
_SUM_KEYS = ("tokens", "loss", "top1", "log2_rank")


def apply_logit_soft_cap(logits: jax.Array, logit_soft_cap: float | tuple[float, float, float] | None) -> jax.Array:
    """The fused CE kernel's cap: ``c * tanh(z / c)`` for a float, ``A * sigmoid((z + B) / C)`` for a tuple."""
    if logit_soft_cap is None:
        return logits
    if isinstance(logit_soft_cap, tuple):
        a, b, c = logit_soft_cap
        return a * jax.nn.sigmoid((logits + b) / c)
    return jnp.tanh(logits / logit_soft_cap) * logit_soft_cap


def logit_scale_sums(
    logits: jax.Array,
    labels: jax.Array,
    weights: jax.Array,
    tags: jax.Array,
    scales: Sequence[float],
    *,
    out_sharding: NamedSharding | None = None,
) -> dict[str, jax.Array]:
    """Per-tag sums of one batch's metrics: ``tokens [T]``, ``loss [K, T]`` (cross-entropy at each of the
    ``K`` scales), ``top1 [T]`` and ``log2_rank [T]``; ``logits`` are ``[B, S, V]`` final logits, ``tags``
    the ``[B, T]`` membership matrix of the tagged evaluator."""
    weights = weights.astype(jnp.float32)
    tags = tags.astype(jnp.float32)
    label_logit = jnp.take_along_axis(logits, labels[..., None], axis=-1)[..., 0]
    top1 = (jnp.argmax(logits, axis=-1) == labels).astype(jnp.float32)
    rank = jnp.sum(logits > label_logit[..., None], axis=-1).astype(jnp.float32)
    log2_rank = jnp.log2(1.0 + rank)
    losses = jnp.stack([jax.nn.logsumexp(scale * logits, axis=-1) - scale * label_logit for scale in scales])
    return {
        "tokens": jnp.einsum("bt,bk->k", weights, tags, out_sharding=out_sharding),
        "loss": jnp.einsum("nbt,bt,bk->nk", losses, weights, tags, out_sharding=out_sharding),
        "top1": jnp.einsum("bt,bt,bk->k", top1, weights, tags, out_sharding=out_sharding),
        "log2_rank": jnp.einsum("bt,bt,bk->k", log2_rank, weights, tags, out_sharding=out_sharding),
    }


def _parabola_vertex(x: np.ndarray, y: np.ndarray, k: int) -> tuple[float, float]:
    """Vertex of the parabola through the grid points ``k - 1, k, k + 1``; the grid point itself at an edge
    or when the fit is not convex there."""
    if k == 0 or k == len(x) - 1:
        return float(x[k]), float(y[k])
    coeffs = np.polyfit(x[k - 1 : k + 2], y[k - 1 : k + 2], 2)
    if coeffs[0] <= 0:
        return float(x[k]), float(y[k])
    xv = -coeffs[1] / (2 * coeffs[0])
    return float(xv), float(np.polyval(coeffs, xv))


def summarize_logit_scale_sums(
    fit: dict[str, np.ndarray],
    report: dict[str, np.ndarray],
    *,
    scales: Sequence[float],
    hierarchy: dict[str, list[int]],
    select_parent: str = SELECT_PARENT,
) -> dict[str, float]:
    """Macro (mean of per-tag means over tags with tokens) metrics of both halves, the scale selected on the
    fit half's ``select_parent`` loss curve, and the report half's loss at that scale versus at scale 1."""
    scales = np.asarray(scales, dtype=np.float64)
    unit = np.flatnonzero(np.isclose(scales, 1.0))
    if len(unit) != 1:
        raise ValueError(f"scales must contain 1.0 exactly once, got {scales.tolist()}")
    num_tags = len(fit["tokens"])
    parents = {"all": list(range(num_tags)), **hierarchy}
    if select_parent not in parents:
        raise ValueError(f"select_parent {select_parent!r} is not an eval-set parent: {sorted(parents)}")
    out: dict[str, float] = {}
    curves: dict[tuple[str, str], np.ndarray] = {}

    def macro(sums: dict[str, np.ndarray], key: str, index: list[int]) -> np.ndarray:
        tokens = sums["tokens"][index]
        mask = tokens > 0
        per_tag = np.where(mask, sums[key][..., index] / np.where(mask, tokens, 1.0), 0.0)
        return per_tag[..., mask].mean(axis=-1)

    for half, sums in (("fit", fit), ("report", report)):
        for parent, index in parents.items():
            if not np.any(sums["tokens"][index] > 0):
                continue
            out[f"{half}/{parent}/tokens"] = float(sums["tokens"][index].sum())
            out[f"{half}/{parent}/top1_acc"] = float(macro(sums, "top1", index))
            out[f"{half}/{parent}/log2_rank"] = float(macro(sums, "log2_rank", index))
            curve = macro(sums, "loss", index)
            curves[(half, parent)] = curve
            for scale, loss in zip(scales, curve, strict=True):
                out[f"{half}/{parent}/scale{scale:g}/macro_loss"] = float(loss)

    fit_curve = curves[("fit", select_parent)]
    k = int(np.argmin(fit_curve))
    refined_scale, _ = _parabola_vertex(scales, fit_curve, k)
    out["best/scale"] = float(scales[k])
    out["best/scale_refined"] = refined_scale
    out["best/temperature"] = float(1.0 / scales[k])
    for (half, parent), curve in curves.items():
        if half != "report":
            continue
        out[f"unit/{parent}/macro_loss"] = float(curve[unit[0]])
        out[f"best/{parent}/macro_loss"] = float(curve[k])
        out[f"gain/{parent}/macro_loss"] = float(curve[unit[0]] - curve[k])
        # The report curve's own parabola at the fit half's refined scale, for a grid-free reading.
        coeffs = np.polyfit(scales[max(k - 1, 0) : k + 2], curve[max(k - 1, 0) : k + 2], min(2, k + 1 if k < 1 else 2))
        out[f"best/{parent}/macro_loss_refined"] = float(np.polyval(coeffs, refined_scale))
    return out


class LogitScaleEvaluator:
    """Scores a model at every logit scale in ``scales`` over a ``TaggedEvaluator``'s eval sets, in fit /
    report halves (alternating batches). ``prepare_model`` casts (and, for expert-parallel runs, converts)
    the params the way the tagged evaluator's loss does; ``logit_soft_cap`` is the model's lm_head cap."""

    def __init__(
        self,
        tagged: TaggedEvaluator,
        *,
        scales: Sequence[float],
        prepare_model: Callable[[Transformer], Transformer],
        logit_soft_cap: float | tuple[float, float, float] | None,
        select_parent: str = SELECT_PARENT,
    ):
        self.tagged = tagged
        self.scales = tuple(float(s) for s in scales)
        self.select_parent = select_parent
        mesh = tagged.device_mesh
        per_tag_out_sharding = None if mesh is None else NamedSharding(mesh, P(None))
        num_tags = tagged.dataset.num_tags
        num_scales = len(self.scales)

        @hax.named_jit(axis_resources=tagged.axis_mapping)
        def accum(model: Transformer, sums: dict[str, jax.Array], batch, tags: jax.Array) -> dict[str, jax.Array]:
            model = prepare_model(model)
            if isinstance(batch, LmExample):
                batch = grug_lm_example_from_named(batch)
            assert isinstance(batch, GrugLmExample)
            hidden, _ = model(batch.tokens, mask=batch.attn_mask)
            head_in, lm_head = model._lm_head_operands(hidden, batch.tokens)
            logits = jnp.einsum(
                "bse,ev->bsv",
                head_in,
                lm_head,
                preferred_element_type=jnp.float32,
                out_sharding=P(*_batch_spec(), None, None),
            )
            logits = apply_logit_soft_cap(logits, logit_soft_cap)
            labels = jnp.pad(batch.tokens[:, 1:], ((0, 0), (0, 1)))
            batch_sums = logit_scale_sums(
                logits, labels, batch.loss_weight, tags, self.scales, out_sharding=per_tag_out_sharding
            )
            return {key: sums[key] + batch_sums[key] for key in _SUM_KEYS}

        self._accum = accum
        self._zeros = lambda: hax.shard(
            {
                "tokens": jnp.zeros((num_tags,), jnp.float32),
                "loss": jnp.zeros((num_scales, num_tags), jnp.float32),
                "top1": jnp.zeros((num_tags,), jnp.float32),
                "log2_rank": jnp.zeros((num_tags,), jnp.float32),
            }
        )

    def evaluate(self, model: Transformer) -> dict[str, float]:
        halves = [self._zeros(), self._zeros()]
        loader = self.tagged.loader
        for i, (batch, tags) in enumerate(tqdm(loader, "logit-scale eval", total=len(loader))):
            halves[i % 2] = self._accum(model, halves[i % 2], batch, tags)
        fit, report = (jax.tree_util.tree_map(np.asarray, half) for half in halves)
        return summarize_logit_scale_sums(
            fit, report, scales=self.scales, hierarchy=self.tagged.hierarchy, select_parent=self.select_parent
        )


def parse_logit_scale_grid(spec: str) -> tuple[float, ...]:
    """``START:STOP:COUNT`` -> evenly spaced scales, with 1.0 added when the grid does not hit it."""
    parts = spec.split(":")
    if len(parts) != 3:
        raise ValueError(f"expected START:STOP:COUNT, got {spec!r}")
    start, stop, count = float(parts[0]), float(parts[1]), int(parts[2])
    if count < 2 or not 0 < start < stop:
        raise ValueError(f"expected 0 < START < STOP and COUNT >= 2, got {spec!r}")
    grid = np.linspace(start, stop, count)
    if not np.any(np.isclose(grid, 1.0)):
        grid = np.append(grid, 1.0)
    return tuple(float(s) for s in np.sort(grid))
