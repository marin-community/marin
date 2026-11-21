"""Megatron router replay (R3) — layout math and controller, megatron-free.

Everything here is CPU-testable without the ``megatron`` extra: the capture
layer mapping, the sequence-major flatten that mirrors mcore's router view
(``[S, B, E].view(-1, E)`` ⇒ token index ``s*B + b``), the TP sequence-parallel
slice, the per-layer replay controller, and the geometry validation. The
install step that touches real ``TopKRouter`` objects lives in
``skyrl_train.workers.megatron.router_replay_install``.

Controller lifecycle (per micro-batch forward through mcore's pipeline
scheduler)::

    controller.begin_forward(per_layer_targets, mask, prediction_mask)
    out = model(...)                      # each local router fires once
    controller.end_forward()
    ...
    controller.assert_drained()           # after every mini-batch / forward-only pass

Under activation recompute the router fires again during backward. Those calls
arrive in ``IDLE`` phase and are served from a per-layer FIFO recorded during
``FORWARD``, mirroring mcore's own ``RouterReplay`` design but keyed on the
global ``layer_number`` instead of a global instantiation-order list.
"""

from __future__ import annotations

from collections import deque
from contextlib import contextmanager
from skyrl_train.mismatch_probe.modes import FILTERED_REPLAY_MODE, REPLAY_MODE
from enum import Enum
import math
from typing import Callable, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch


__all__ = [
    "MegatronRouterReplay",
    "LayerReplayHandle",
    "MIN_ROUTER_TOPK",
    "SENTINEL_EXPERT_ID",
    "dense_replay_targets",
    "require_scalar_num_actions",
    "capture_layer_indices",
    "expand_moe_layer_freq",
    "num_moe_layers",
    "sequence_major_flatten",
    "slice_sequence_parallel",
    "validate_replay_geometry",
    "filtered_replay_topk",
]

# Sentinel expert id written during rollout capture for unmatched /
# non-generated token rows (trajectory_runners/trajectory_processing.py). Rows whose captured
# targets are all this value fall through to native routing.
SENTINEL_EXPERT_ID = 0


def require_scalar_num_actions(num_actions) -> None:
    """Reject per-sample ``num_actions`` lists/arrays (dense replay needs a scalar)."""
    if isinstance(num_actions, (list, np.ndarray)):
        raise NotImplementedError(
            "router_replay requires a scalar num_actions (dense unpacked path); got a per-sample list/array."
        )


def dense_replay_targets(rollout_routed_experts, batch_size, seq_len, num_actions):
    """Build the dense per-position replay target and mask, layout-agnostic.

    ``rollout_routed_experts`` is ``[B, local_response_len, L, K]``. It starts
    at the beginning of the global ``num_actions`` response window. Each row
    belongs on the model position that predicted that response token, beginning
    with the final prompt token. Returns ``(full, mask)`` where ``full`` is a
    ``[B, seq_len, L, K]`` long tensor. ``mask`` is True only for captured,
    non-sentinel prediction positions. Other positions use native routing.
    """
    require_scalar_num_actions(num_actions)
    device = rollout_routed_experts.device
    captured = rollout_routed_experts.to(dtype=torch.long)
    B, response_len, L, K = captured.shape
    assert B == batch_size, f"router_replay batch mismatch: {B} vs {batch_size}"
    assert response_len <= num_actions, f"router_replay response_len {response_len} exceeds num_actions {num_actions}"
    if num_actions >= seq_len:
        raise ValueError("router_replay requires a prompt token before the response")

    full = torch.full((batch_size, seq_len, L, K), SENTINEL_EXPERT_ID, dtype=torch.long, device=device)
    prediction_start = seq_len - num_actions - 1
    full[:, prediction_start : prediction_start + response_len, :, :] = captured

    prediction_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
    prediction_mask[:, prediction_start : seq_len - 1] = True
    # non-sentinel per [B, seq_len, L]; collapse over L: a position is valid
    # for replay only where every layer carries real data, then AND with prediction_mask.
    non_sentinel = (full != SENTINEL_EXPERT_ID).any(dim=-1).all(dim=-1)  # [B, seq_len]
    return full, prediction_mask & non_sentinel


# The all-K-sentinel capture convention is only unambiguous when native top-k
# indices are distinct, which requires top-k >= 2.
MIN_ROUTER_TOPK = 2


def capture_layer_indices(moe_layer_pattern: Sequence[int]) -> dict[int, int]:
    """Map global 1-indexed ``layer_number`` → capture index for MoE layers.

    ``moe_layer_pattern[i] == 1`` means the layer at 0-based position ``i``
    (global ``layer_number`` ``i + 1``) is a MoE layer. The capture index is
    the layer's position among MoE layers — the index into the ``L`` axis of
    the rollout's ``rollout_routed_experts`` tensor.
    """
    mapping: dict[int, int] = {}
    capture_idx = 0
    for position, is_moe in enumerate(moe_layer_pattern):
        if is_moe == 1:
            mapping[position + 1] = capture_idx
            capture_idx += 1
    return mapping


def num_moe_layers(moe_layer_pattern: Sequence[int]) -> int:
    """Count MoE entries in a 0/1 layer pattern."""
    return sum(1 for is_moe in moe_layer_pattern if is_moe == 1)


def expand_moe_layer_freq(moe_layer_freq, num_layers: int) -> list[int]:
    """Resolve mcore's ``config.moe_layer_freq`` to a 0/1 pattern list.

    Mirrors ``get_gpt_decoder_layer_specs`` in megatron-core: an integer N
    means one expert layer every N layers (``i % N == 0`` for 0-based ``i``);
    a list is taken as-is and must have one entry per layer. 0 = dense, 1 = MoE.
    """
    if isinstance(moe_layer_freq, int):
        return [1 if i % moe_layer_freq == 0 else 0 for i in range(num_layers)]
    if isinstance(moe_layer_freq, list):
        if len(moe_layer_freq) != num_layers:
            raise ValueError(
                f"Invalid length of moe_layer_freq: {len(moe_layer_freq)}, expected {num_layers}, "
                f"current moe layer pattern: {moe_layer_freq}"
            )
        return list(moe_layer_freq)
    raise ValueError(f"Invalid moe_layer_freq: {type(moe_layer_freq)}, {moe_layer_freq}")


def sequence_major_flatten(x: torch.Tensor) -> torch.Tensor:
    """Flatten ``[B, S', ...]`` to ``[S'*B, ...]`` in sequence-major order.

    Mirrors the router-side flatten mcore applies to ``[S, B, E]`` activations
    (``view(-1, E)``): flat index ``s*B + b``. Any tensor pushed through the
    same sequence transform as the model input lands, after this flatten, in
    exactly the router's token order.
    """
    if x.ndim < 2:
        raise ValueError(f"sequence_major_flatten expects [B, S', ...], got shape {tuple(x.shape)}")
    batch_size, seq_len = x.shape[0], x.shape[1]
    return x.transpose(0, 1).reshape(seq_len * batch_size, *x.shape[2:])


def slice_sequence_parallel(
    x: torch.Tensor, *, seq_len: int, batch_size: int, tp_rank: int, tp_size: int
) -> torch.Tensor:
    """Slice a sequence-major flat tensor to one TP rank's contiguous chunk.

    Under TP sequence parallelism each rank's router sees its contiguous
    ``seq_len // tp_size`` rows of the sequence dim, i.e. flat rows
    ``[tp_rank * local * batch_size, (tp_rank + 1) * local * batch_size)``.
    The padding helpers upstream guarantee ``seq_len % tp_size == 0``.
    """
    if seq_len % tp_size != 0:
        raise ValueError(
            f"router replay: seq_len {seq_len} not divisible by tp_size {tp_size}; "
            "the Megatron padding helpers must run before the slice"
        )
    local = seq_len // tp_size
    start = tp_rank * local * batch_size
    return x[start : start + local * batch_size]


class _Phase(Enum):
    IDLE = "idle"
    FORWARD = "forward"


class RouterScoreType(Enum):
    LOGITS = "logits"
    PROBABILITIES = "probabilities"
    BIASED_PROBABILITIES = "biased_probabilities"


def filtered_replay_topk(
    scores: torch.Tensor,
    native_idx: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    keep_fraction: float,
    score_type: RouterScoreType = RouterScoreType.LOGITS,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Filtered router replay (Composer 2 Technical Report, arXiv:2603.24477): replay vLLM's experts except those whose trainer selection score is below keep_fraction of the trainer's k-th best, refilling those slots with the trainer's next-best unused experts."""
    if score_type is RouterScoreType.BIASED_PROBABILITIES:
        raise NotImplementedError("filtered replay does not support expert bias added to probability scores")
    if not 0.0 <= keep_fraction <= 1.0 or not math.isfinite(keep_fraction):
        raise ValueError("filtered replay keep_fraction must be finite and in [0, 1]")
    if scores.ndim != 2 or native_idx.shape != targets.shape or native_idx.shape[0] != scores.shape[0]:
        raise ValueError("filtered replay scores, native choices and captured choices have incompatible shapes")
    if mask.shape != (scores.shape[0],):
        raise ValueError("filtered replay mask must have one value per token")
    if not torch.isfinite(scores).all():
        raise ValueError("filtered replay requires finite selection scores")
    selected = native_idx.clone()
    replacements = torch.zeros_like(targets, dtype=torch.bool)
    threshold_offset = math.log(keep_fraction) if keep_fraction else -math.inf
    native_scores = scores.detach().gather(1, native_idx)
    cutoff_scores = native_scores.amin(dim=-1)
    ranked_native = native_idx.gather(1, torch.argsort(native_scores, dim=-1, descending=True, stable=True))
    for row in torch.nonzero(mask, as_tuple=False).flatten().tolist():
        captured = targets[row].tolist()
        if len(set(captured)) != len(captured):
            raise ValueError("filtered replay captured experts must be distinct within a row")
        cutoff_score = cutoff_scores[row].item()
        cutoff_score = (
            cutoff_score + threshold_offset if score_type is RouterScoreType.LOGITS else cutoff_score * keep_fraction
        )
        kept = [scores[row, expert].item() >= cutoff_score for expert in captured]
        used = {expert for expert, accepted in zip(captured, kept, strict=True) if accepted}
        native_candidates = iter(expert for expert in ranked_native[row].tolist() if expert not in used)
        for slot, (expert, accepted) in enumerate(zip(captured, kept, strict=True)):
            if accepted:
                selected[row, slot] = expert
            else:
                selected[row, slot] = next(native_candidates)
                replacements[row, slot] = True
    return selected, replacements


class MegatronRouterReplay:
    """Per-layer replay controller for megatron-core ``TopKRouter.router_replay``.

    Duck-types the single method mcore calls (``get_replay_topk``, through
    :class:`LayerReplayHandle`). One instance per rank, installed on every
    local ``TopKRouter``; ``local_layer_indices`` are the capture indices this
    rank's model chunks own.
    """

    def __init__(self, local_layer_indices: Sequence[int], *, recompute_enabled: bool) -> None:
        self.local_layer_indices: tuple[int, ...] = tuple(sorted(local_layer_indices))
        # Filled by the installer from the resolved model config; the target
        # builder validates every batch against them before arming.
        self.num_moe_layers_total: Optional[int] = None
        self.topk: Optional[int] = None
        # id(model chunk) -> capture indices owned by that chunk. Empty for a
        # single chunk; with virtual pipelining each forward bracket arms only
        # the layers of the chunk mcore is about to call.
        self.local_indices_for_module: dict[int, tuple[int, ...]] = {}
        self._recompute_enabled = recompute_enabled
        self._phase = _Phase.IDLE
        # layer capture idx -> (targets [N, K], replay mask [N]) for the current forward
        self._current: dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        self._expected: tuple[int, ...] = ()
        self._consumed: set[int] = set()
        self._prediction_mask: Optional[torch.Tensor] = None
        self._probe_positions: Optional[torch.Tensor] = None
        self._probe_observations: list[dict[str, object]] = []
        self._record_recompute = False
        self._fifo: dict[int, deque] = {idx: deque() for idx in self.local_layer_indices}
        self._masked_rows = 0
        self._hit_rows = 0
        self._response_rows = 0
        self._sentinel_rows = 0
        self._scoring_mode = REPLAY_MODE
        self._keep_fraction: float | None = None

    @contextmanager
    def scoring_mode(self, mode: str, keep_fraction: float | None = None):
        """Scope a probe forward without changing subsequent training forwards."""
        if self._phase is not _Phase.IDLE:
            raise RuntimeError("router replay: scoring mode requires an idle controller")
        if mode not in (REPLAY_MODE, FILTERED_REPLAY_MODE):
            raise ValueError(f"unsupported replay scoring mode: {mode}")
        if mode == FILTERED_REPLAY_MODE and keep_fraction is None:
            raise ValueError("filtered replay requires keep_fraction")
        previous = (self._scoring_mode, self._keep_fraction)
        self._scoring_mode, self._keep_fraction = mode, keep_fraction
        try:
            yield
        finally:
            self._scoring_mode, self._keep_fraction = previous

    # ------------------------------------------------------------- drivers

    def begin_forward(
        self,
        per_layer_targets: Mapping[int, torch.Tensor],
        mask: torch.Tensor,
        prediction_mask: Optional[torch.Tensor] = None,
        *,
        record_recompute: bool = True,
        probe_positions: Optional[torch.Tensor] = None,
    ) -> None:
        """Arm the controller for one forward over the model's local layers.

        ``per_layer_targets`` maps capture index → ``[N, K]`` long targets in
        the router's token order; ``mask`` is the ``[N]`` replay gate.
        ``prediction_mask`` (optional, ``[N]``) marks the positions that predicted response tokens before
        sentinel exclusion and feeds the ``sentinel_fraction`` metric.
        ``record_recompute`` is true for training forwards whose backward will
        replay activation-checkpointed layers, even when forward runs under no_grad.
        ``probe_positions`` holds ``[N, 2]`` sample and response positions;
        non-response rows use -1 for the response position.
        """
        if self._phase is not _Phase.IDLE:
            raise RuntimeError("router replay: begin_forward while a forward is already armed (phase must be IDLE)")
        self._current = {idx: (targets, mask) for idx, targets in per_layer_targets.items()}
        self._expected = tuple(sorted(per_layer_targets))
        self._consumed = set()
        self._prediction_mask = prediction_mask
        if probe_positions is not None and (probe_positions.ndim != 2 or probe_positions.shape != (mask.numel(), 2)):
            raise ValueError("probe positions must have one (sample, response-position) pair per router row")
        self._probe_positions = probe_positions
        self._record_recompute = record_recompute
        self._phase = _Phase.FORWARD

    def end_forward(self) -> None:
        """Close the forward bracket and verify every armed layer fired once."""
        if self._phase is not _Phase.FORWARD:
            raise RuntimeError("router replay: end_forward without an armed forward")
        missing = [idx for idx in self._expected if idx not in self._consumed]
        if missing:
            raise ValueError(
                f"router replay: armed layer(s) {missing} never fired during the forward — "
                "the capture layer mapping does not match the model"
            )
        self._current = {}
        self._expected = ()
        self._prediction_mask = None
        self._probe_positions = None
        self._record_recompute = False
        self._phase = _Phase.IDLE

    def abort_forward(self) -> None:
        """Force-reset after a mid-forward failure; never raises.

        Clears the armed targets and any partial recompute FIFO entries so the
        controller is usable again; the caller re-raises the original error.
        """
        self._current = {}
        self._expected = ()
        self._consumed = set()
        self._prediction_mask = None
        self._probe_positions = None
        self._record_recompute = False
        self._phase = _Phase.IDLE
        for fifo in self._fifo.values():
            fifo.clear()

    def assert_drained(self) -> None:
        """Assert the recompute FIFO is empty and no forward is armed.

        Called at the end of every mini-batch and every forward-only pass; a
        violation means router calls and forward brackets went out of sync.
        """
        if self._phase is not _Phase.IDLE:
            raise RuntimeError("router replay: assert_drained while a forward is still armed")
        outstanding = ", ".join(f"layer {idx} has {len(fifo)} outstanding" for idx, fifo in self._fifo.items() if fifo)
        if outstanding:
            raise RuntimeError(f"router replay: recompute FIFO not drained: {outstanding}")

    def pop_metrics(self) -> dict[str, float]:
        """Return ``hit_fraction`` / ``sentinel_fraction`` since the last pop.

        ``hit_fraction`` is replayed rows over masked rows (1.0 unless a masked
        row carried an all-sentinel target — a layout bug). ``sentinel_fraction``
        is sentinel response rows over response rows (rollout capture loss).
        """
        hit_fraction = self._hit_rows / self._masked_rows if self._masked_rows else 1.0
        sentinel_fraction = self._sentinel_rows / self._response_rows if self._response_rows else 0.0
        self._masked_rows = 0
        self._hit_rows = 0
        self._response_rows = 0
        self._sentinel_rows = 0
        return {"hit_fraction": hit_fraction, "sentinel_fraction": sentinel_fraction}

    def take_probe_observations(self) -> list[dict[str, object]]:
        """Return route choices captured only for an explicitly marked probe forward."""
        observations, self._probe_observations = self._probe_observations, []
        return observations

    # ------------------------------------------------------ router-side entry

    def get_replay_topk(
        self,
        layer_idx: int,
        scores: torch.Tensor,
        topk: int,
        num_groups: Optional[int] = None,
        group_topk: Optional[int] = None,
        default_compute_topk: Optional[
            Callable[[torch.Tensor, int, Optional[int], Optional[int]], Tuple[torch.Tensor, torch.Tensor]]
        ] = None,
        score_type: RouterScoreType = RouterScoreType.LOGITS,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(probs, top_indices)`` with rollout choices on masked rows.

        Masked rows use captured experts, with implausible choices replaced by
        native experts in filtered mode. Other rows use the native top-k.
        ``probs`` are always the live routing scores gathered
        at the returned indices, so gradients flow through the gate on both
        replayed and native rows. Runs mcore's ``default_compute_topk``
        unconditionally to keep the autograd graph identical to a flag-off
        forward.
        """
        probs, native_idx = default_compute_topk(scores, topk, num_groups=num_groups, group_topk=group_topk)
        targets, mask = self._targets_for_call(layer_idx, scores)

        if targets.shape[0] != scores.shape[0]:
            raise ValueError(
                f"router replay: layer {layer_idx} targets have {targets.shape[0]} rows "
                f"but the router scored {scores.shape[0]} tokens — the route tensor and the "
                "model input took different layout transforms"
            )
        if targets.shape[1] != topk:
            raise ValueError(
                f"router replay: layer {layer_idx} targets carry K={targets.shape[1]} but the router uses topk={topk}"
            )

        mask = mask.to(device=scores.device, dtype=torch.bool)
        targets = targets.to(device=scores.device)
        replaced = torch.zeros_like(targets, dtype=torch.bool)
        if self._scoring_mode == FILTERED_REPLAY_MODE:
            idx, replaced = filtered_replay_topk(scores, native_idx, targets, mask, self._keep_fraction, score_type)
        else:
            idx = torch.where(mask.unsqueeze(-1), targets, native_idx)
        probs = scores.gather(1, idx)

        if self._probe_positions is not None:
            if self._probe_positions.shape[0] != scores.shape[0]:
                raise ValueError("probe route positions do not match the router's token order")
            positions = self._probe_positions.to(scores.device)
            active = positions[:, 1] >= 0
            packed = (
                torch.cat(
                    (
                        positions[active],
                        native_idx[active],
                        idx[active],
                        replaced[active].long(),
                        mask[active, None].long(),
                    ),
                    dim=-1,
                )
                .detach()
                .cpu()
                .tolist()
            )
            self._probe_observations.extend(
                {
                    "layer": layer_idx,
                    "sample": row[0],
                    "position": row[1],
                    "native": row[2 : 2 + topk],
                    "effective": row[2 + topk : 2 + 2 * topk],
                    "replaced": row[2 + 2 * topk : 2 + 3 * topk],
                    "route_valid": bool(row[-1]),
                }
                for row in packed
            )

        replayed = mask.sum().item()
        # A masked row whose target is all-sentinel means the mask and the
        # target tensor disagree (layout bug): count it so hit_fraction < 1.0
        # surfaces it as a hard error at mini-batch end.
        lost = (mask & (targets == SENTINEL_EXPERT_ID).all(dim=-1)).sum().item()
        self._masked_rows += replayed
        self._hit_rows += replayed - lost
        if self._prediction_mask is not None:
            response = self._prediction_mask.to(device=scores.device)
            self._response_rows += response.sum().item()
            self._sentinel_rows += (response & ~mask).sum().item()
        return probs, idx

    def _targets_for_call(self, layer_idx: int, scores: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._phase is _Phase.FORWARD:
            if layer_idx not in self._current:
                raise ValueError(
                    f"router replay: layer {layer_idx} fired with no targets armed for this forward "
                    f"(armed: {list(self._current)}); the capture layer mapping does not match the model"
                )
            if layer_idx in self._consumed:
                raise ValueError(f"router replay: layer {layer_idx} consumed twice in one forward")
            self._consumed.add(layer_idx)
            targets, mask = self._current[layer_idx]
            # Activation-checkpointed training forwards run under no_grad;
            # the caller, not grad mode, identifies whether backward follows.
            if self._recompute_enabled and self._record_recompute:
                if layer_idx not in self._fifo:
                    raise ValueError(f"router replay: layer {layer_idx} has no FIFO; not a local layer")
                self._fifo[layer_idx].append((targets, mask))
            return targets, mask
        if not self._recompute_enabled or layer_idx not in self._fifo or not self._fifo[layer_idx]:
            raise RuntimeError(
                f"router replay: recompute without a recorded forward (layer {layer_idx}, "
                f"recompute_enabled={self._recompute_enabled})"
            )
        return self._fifo[layer_idx].popleft()


class LayerReplayHandle:
    """Per-router duck-type of mcore's ``RouterReplay`` bound to one capture index.

    Assigned to ``TopKRouter.router_replay``; mcore calls
    ``get_replay_topk(scores, topk, num_groups, group_topk, _compute_topk)``
    positionally.
    """

    def __init__(
        self,
        controller: MegatronRouterReplay,
        layer_idx: int,
        score_type: RouterScoreType = RouterScoreType.LOGITS,
    ) -> None:
        self._controller = controller
        self.layer_idx = layer_idx
        self._score_type = score_type

    def get_replay_topk(
        self,
        scores: torch.Tensor,
        topk: int,
        num_groups: Optional[int] = None,
        group_topk: Optional[int] = None,
        default_compute_topk: Optional[Callable] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._controller.get_replay_topk(
            self.layer_idx, scores, topk, num_groups, group_topk, default_compute_topk, self._score_type
        )


def validate_replay_geometry(
    *,
    num_layers_captured: int,
    expected_moe_layers: int,
    topk_captured: int,
    expected_topk: int,
    num_experts: int,
    targets: torch.Tensor,
    response_len: int,
    num_actions,
) -> None:
    """Validate a ``rollout_routed_experts`` tensor against the model geometry.

    Raises ``ValueError`` naming the offending quantity on any mismatch;
    raises ``NotImplementedError`` for a per-sample ``num_actions`` list.
    """
    require_scalar_num_actions(num_actions)
    if num_layers_captured != expected_moe_layers:
        raise ValueError(
            f"router replay: rollout_routed_experts carries L={num_layers_captured} layers but the model has "
            f"expected_moe_layers={expected_moe_layers} MoE layers"
        )
    if expected_topk < MIN_ROUTER_TOPK:
        raise ValueError(
            f"router replay: expected_topk={expected_topk} < {MIN_ROUTER_TOPK}; the all-K sentinel convention is "
            "ambiguous when native top-k indices need not be distinct"
        )
    if topk_captured != expected_topk:
        raise ValueError(
            f"router replay: rollout_routed_experts carries K={topk_captured} but the model uses "
            f"expected_topk={expected_topk}"
        )
    if targets.numel() and (targets.min().item() < 0 or targets.max().item() >= num_experts):
        raise ValueError(
            f"router replay: rollout_routed_experts carries an expert id outside [0, num_experts={num_experts})"
        )
    if response_len > num_actions:
        raise ValueError(f"router replay: response_len={response_len} exceeds num_actions={num_actions}")
