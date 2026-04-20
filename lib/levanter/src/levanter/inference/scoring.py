# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Teacher-forced scoring on top of Levanter's paged KV cache.

Given a prompt P and N candidate completions (C_1, ..., C_N), returns
``log p(C_i | P)`` for each candidate, reusing the prompt's KV via
``GenState.clone_sequence``.

Public API mirrors the ``Scorer`` protocol in
``experiments/rerank_decode/scorer.py``:

    engine = ScoringEngine.from_model_with_config(
        model, tokenizer,
        ScoringEngineConfig(max_seq_len=1088, max_batch_size=16),
    )
    scores = engine.score(prompt_ids, [c1_ids, c2_ids, ...])   # caches prompt
    engine.accept(prompt_ids, chosen_ids)                      # extend anchor
    scores = engine.score(prompt_ids + chosen_ids, [...])      # cache hit
    engine.reset()                                             # new prompt

``score`` and ``accept`` accept an arbitrary ``prompt_tokens`` each call;
the engine internally extends or re-prefills the anchor to match.

Design notes:
  * Anchor slot (slot 0) holds the prompt. Scoring clones the anchor into
    slots 1..N so their KV shares the prompt's fully-used pages.
  * ``model.decode(tokens, cache, batch_info, pos_ids)`` returns per-position
    logits over the packed batch; we capture them instead of sampling.
  * First-token log-prob (``log p(C_i[0] | P)``) comes from the anchor's
    last-position logits saved during the prompt prefill; remaining log-probs
    come from the per-position logits produced during the scoring forward.
"""

from __future__ import annotations

import dataclasses
import functools
import logging
from dataclasses import dataclass
from typing import Optional

import equinox as eqx
import haliax as hax
import jax
import jax.numpy as jnp
import numpy as np
from haliax import NamedArray
from haliax.partitioning import ResourceMapping

from levanter.inference.engine import GenState, _infer_max_pages_from_hbm
from levanter.inference.jit_scheduler import DecodeState
from levanter.inference.page_table import PageTable
from levanter.inference.utils import INVALID, is_invalid
from levanter.layers.kv_cache import PageCache
from levanter.models.lm_model import LmHeadModel

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ScoringEngineConfig:
    """Sizing knobs for a ScoringEngine. Shape mirrors ``InferenceEngineConfig``.

    Required:
        max_seq_len: Maximum per-sequence length (anchor prompt + extensions + completion).
        max_batch_size: Max candidate completions per ``score()`` call.
        max_completion_len: Maximum length (tokens) of any single candidate completion.

    The engine reserves ``max_batch_size + 1`` sequence slots (one anchor + up to N clones).
    The anchor prompt is bounded by ``max_prompt_len = max_seq_len - max_completion_len``
    so that a max-length completion can always be scored against a max-length anchor.
    """

    max_seq_len: int
    max_batch_size: int
    max_completion_len: int

    hbm_utilization: float = 0.9
    """Fraction of device HBM to reserve for the KV cache when ``max_pages`` is ``None``."""

    page_size: int = 128
    """Tokens per KV page."""

    max_pages: Optional[int] = None
    """Total number of KV pages available. If ``None``, inferred from ``hbm_utilization``."""

    compute_dtype: jnp.dtype = jnp.bfloat16
    """KV cache dtype."""

    prompt_chunk_size: int = 512
    """Maximum packed size for prompt prefill/extension forwards."""

    def __post_init__(self):
        if self.max_batch_size <= 0:
            raise ValueError("max_batch_size must be positive")
        if self.max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")
        if self.max_completion_len <= 0:
            raise ValueError("max_completion_len must be positive")
        if self.max_completion_len > self.max_seq_len:
            raise ValueError(
                f"max_completion_len ({self.max_completion_len}) must be <= max_seq_len ({self.max_seq_len})"
            )

    @property
    def max_seqs(self) -> int:
        """Total local slots: one anchor + ``max_batch_size`` clones."""
        return self.max_batch_size + 1

    @property
    def max_prompt_len(self) -> int:
        """Upper bound on anchor prompt length that still leaves room for a max-length completion."""
        return self.max_seq_len - self.max_completion_len

    @property
    def max_pages_per_seq(self) -> int:
        return (self.max_seq_len + self.page_size - 1) // self.page_size


ANCHOR_SLOT = 0


@functools.partial(jax.jit, donate_argnums=0)
def _run_forward(
    gen_state: GenState,
    model: LmHeadModel,
    tokens: NamedArray,
    slot_ids: NamedArray,
    pos_ids: NamedArray,
) -> tuple[GenState, NamedArray]:
    """Run ``model.decode`` over a packed batch and return per-position logits.

    ``tokens``, ``slot_ids`` and ``pos_ids`` all share axis ``"position"``. No
    sampling, no token output — this is the scoring analog of ``_prefill_kernel``.
    Returns the updated GenState (KV cache mutated) and ``logits[position, vocab]``.
    """
    decode_state, binfo = gen_state.decode_state.allocate_for_seq(
        token_slot_ids=slot_ids, token_pos_ids=pos_ids
    )
    logits, cache = model.decode(tokens, gen_state.cache, binfo, pos_ids)
    new_state = dataclasses.replace(gen_state, cache=cache, decode_state=decode_state)
    return new_state, logits


@functools.partial(jax.jit, donate_argnums=0)
def _release_slots_by_mask(gen_state: GenState, release_mask: jnp.ndarray) -> GenState:
    """Decrement refcounts and free all slots whose ``release_mask[i]`` is True.

    Single JIT call regardless of how many slots — avoids per-slot retracing.
    """
    decode_state = gen_state.decode_state.free_pages_for_finished(release_mask)
    return dataclasses.replace(gen_state, decode_state=decode_state)


@eqx.filter_jit
def _gather_scores(
    logits: NamedArray,
    vocab_axis,
    anchor_last_log_probs: NamedArray,
    first_tokens: jnp.ndarray,
    candidate_ids: jnp.ndarray,
    target_tokens: jnp.ndarray,
    score_mask: jnp.ndarray,
    batch_size: int,
) -> jnp.ndarray:
    """Device-side: per-candidate ``log p(completion | prompt)``.

    Scoring packs all real completion tokens contiguously on the packed ``position``
    axis and tracks a flat candidate id per packed position. The first-token
    log-prob comes from ``anchor_last_log_probs``; remaining log-probs come from
    ``logits`` at each candidate's scored positions.

    Args:
        logits: ``[position, vocab]`` NamedArray from ``model.decode``.
        vocab_axis: Haliax Axis for the vocab dimension.
        anchor_last_log_probs: ``[vocab]`` NamedArray — log-probs at the anchor's
            last prompt position, computed in ``_set_prompt`` / ``_extend_prompt``.
        first_tokens: ``[batch_size]`` int32; first token of each candidate.
        candidate_ids: ``[position]`` int32; candidate index for each packed position.
        target_tokens: ``[position]`` int32; token whose log-prob
            to gather at each packed position.
        score_mask: ``[position]`` bool; True for positions that
            contribute to a candidate's score.
        batch_size: Number of candidates (``config.max_batch_size``).

    Returns raw ``[batch_size]`` fp32 array of combined per-candidate log-probs.
    """
    Batch = hax.Axis("batch", batch_size)

    # fp32 log_softmax for numerical stability.
    log_probs = hax.nn.log_softmax(logits.astype(jnp.float32), axis=vocab_axis)
    target_na = hax.named(target_tokens, axis="position")
    mask_na = hax.named(score_mask, axis="position")

    # Scoring-forward log-prob sum per candidate via one-hot + segment_sum over the
    # flat packed positions.
    target_one_hot = hax.nn.one_hot(target_na, vocab_axis, dtype=log_probs.dtype)
    gathered = (log_probs * target_one_hot).sum(vocab_axis)
    masked = hax.where(mask_na, gathered, 0.0)
    score_sum = hax.named(jax.ops.segment_sum(masked.array, candidate_ids, num_segments=batch_size), axis="batch")

    # First-token log-prob from the anchor.
    first_tokens_na = hax.named(first_tokens, axis="batch")
    first_one_hot = hax.nn.one_hot(first_tokens_na, vocab_axis, dtype=log_probs.dtype)
    anchor_lp = anchor_last_log_probs.astype(jnp.float32)
    first_lp = (anchor_lp * first_one_hot).sum(vocab_axis)

    per_batch = score_sum + first_lp
    return per_batch.array


class ScoringEngine:
    """Paged-KV teacher-forced scoring.

    Not thread-safe. Not reentrant. One prompt active at a time.
    """

    def __init__(
        self,
        *,
        model: LmHeadModel,
        tokenizer,
        cache: PageCache,
        decode_state: DecodeState,
        config: ScoringEngineConfig,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.gen_state: GenState = GenState(cache=cache, decode_state=decode_state)

        self._anchor_last_log_probs: jnp.ndarray | None = None  # [vocab] on device
        self._vocab_axis = model.Vocab

        # Precomputed mask of non-anchor slots, used at the top of every score() to
        # blanket-reset leftover clone state (mirrors InferenceEngine's reset-at-start lifecycle).
        non_anchor = np.ones((config.max_seqs,), dtype=bool)
        non_anchor[ANCHOR_SLOT] = False
        self._non_anchor_mask: jnp.ndarray = jnp.asarray(non_anchor)

    @classmethod
    def from_model_with_config(
        cls,
        model: LmHeadModel,
        tokenizer,
        config: ScoringEngineConfig,
        axis_resources: ResourceMapping | None = None,
    ) -> "ScoringEngine":
        """Build a ScoringEngine with HBM-inferred page budget if ``max_pages`` is unset."""
        if config.max_pages is None:
            inferred_pages = _infer_max_pages_from_hbm(model, config)
            config = dataclasses.replace(config, max_pages=int(inferred_pages))

        assert config.max_pages is not None

        table = PageTable.init(
            max_pages=config.max_pages,
            max_seqs=config.max_seqs,
            page_size=config.page_size,
            max_pages_per_seq=config.max_pages_per_seq,
        )
        cache = hax.named_jit(model.initial_cache, axis_resources=axis_resources)(
            table.spec(), dtype=config.compute_dtype
        )
        decode_state = DecodeState.init(
            page_table=table,
            max_stop_seqs=0,
            max_stop_tokens=0,
            max_queued_tokens=config.prompt_chunk_size,
        )
        return cls(
            model=model,
            tokenizer=tokenizer,
            cache=cache,
            decode_state=decode_state,
            config=config,
        )

    # ----- Public API -----

    def reset(self) -> None:
        """Free all slots and clear anchor state. KV memory stays allocated."""
        self.gen_state = eqx.filter_jit(self.gen_state.reset, donate="all")()
        self._anchor_last_log_probs = None

    def score(self, prompt_tokens: list[int], completions: list[list[int]]) -> list[float]:
        """Return ``log p(completion_i | prompt_tokens)`` for each completion.

        Aligns the anchor to ``prompt_tokens`` first: reuses the current anchor if
        it's a proper prefix (extending with the tail); otherwise reprefills from
        scratch. Then scores the completions as a batched clone + forward.
        """
        self._ensure_prefix(prompt_tokens)
        return self._score_completions(completions)

    def accept(self, prompt_tokens: list[int], completion_tokens: list[int]) -> None:
        """Align the anchor to ``prompt_tokens`` and extend it with ``completion_tokens``.

        After ``accept``, the anchor's KV covers ``prompt_tokens + completion_tokens``,
        so the next ``score`` with the same combined prefix will hit the cached KV.
        """
        self._release_non_anchor_slots()
        self._ensure_prefix(prompt_tokens)
        if completion_tokens:
            self._extend_prompt(completion_tokens)

    # ----- Internal helpers -----

    def _anchor_prompt_len(self) -> int:
        return int(jax.device_get(self.gen_state.decode_state.seq_lens["seq", ANCHOR_SLOT].scalar()))

    def _anchor_prompt_tokens(self) -> list[int]:
        prompt_len = self._anchor_prompt_len()
        if prompt_len == 0:
            return []
        tokens = self.gen_state.decode_state.tokens["seq", ANCHOR_SLOT, "position", hax.ds(0, prompt_len)].array
        return list(np.asarray(jax.device_get(tokens), dtype=np.int32))

    def _store_anchor_tokens(self, tokens: list[int]) -> None:
        prompt_len = len(tokens)
        decode_state = self.gen_state.decode_state
        axis_len = decode_state.tokens.axis_size("position")
        stored = np.full((axis_len,), INVALID, dtype=np.int32)
        stored[:prompt_len] = np.asarray(tokens, dtype=np.int32)
        stored_na = hax.named(jnp.asarray(stored, dtype=jnp.int32), axis="position")
        anchor_kv_pages = decode_state.kv_pages["seq", ANCHOR_SLOT]
        anchor_page_indices = decode_state.sequences.page_indices["seq", ANCHOR_SLOT]
        decode_state = decode_state.assign_seq(
            local_slot_id=ANCHOR_SLOT,
            tokens=stored_na,
            seq_len=prompt_len,
            kv_pages=anchor_kv_pages,
            page_indices=anchor_page_indices,
        )
        self.gen_state = dataclasses.replace(self.gen_state, decode_state=decode_state)

    def _ensure_prefix(self, prompt_tokens: list[int]) -> None:
        """Align the anchor with ``prompt_tokens``: extend when it's an extension, else re-prefill."""
        current_tokens = self._anchor_prompt_tokens()
        if not current_tokens:
            self._set_prompt(prompt_tokens)
            return

        if len(prompt_tokens) >= len(current_tokens) and prompt_tokens[: len(current_tokens)] == current_tokens:
            tail = prompt_tokens[len(current_tokens):]
            if tail:
                self._extend_prompt(tail)
            return

        # Divergence — reprefill from scratch.
        self._set_prompt(prompt_tokens)

    def _release_non_anchor_slots(self) -> None:
        self.gen_state = _release_slots_by_mask(self.gen_state, self._non_anchor_mask)

    def _set_prompt(self, prompt_tokens: list[int]) -> None:
        """Prefill the anchor slot with ``prompt_tokens``."""
        if len(prompt_tokens) == 0:
            raise ValueError("Prompt must be non-empty.")
        if len(prompt_tokens) > self.config.max_prompt_len:
            raise ValueError(
                f"Prompt length {len(prompt_tokens)} exceeds max_prompt_len {self.config.max_prompt_len} "
                f"(= max_seq_len - max_completion_len)"
            )

        self.reset()

        # Reserve anchor slot explicitly.
        decode_state, slot = self.gen_state.decode_state.reserve_slot(ANCHOR_SLOT)
        assert int(jax.device_get(slot)) == ANCHOR_SLOT, "Anchor slot reservation mismatch."
        self.gen_state = dataclasses.replace(self.gen_state, decode_state=decode_state)
        self._store_anchor_tokens(prompt_tokens)

        last_logits = self._forward_token_chunks(ANCHOR_SLOT, 0, prompt_tokens)
        self._anchor_last_log_probs = hax.nn.log_softmax(last_logits, axis=self._vocab_axis)

    def _extend_prompt(self, new_tokens: list[int]) -> None:
        """Append ``new_tokens`` to the anchor."""
        current_tokens = self._anchor_prompt_tokens()
        current_len = len(current_tokens)
        if current_len == 0:
            raise RuntimeError("_extend_prompt called before _set_prompt.")
        if len(new_tokens) == 0:
            return
        new_len = current_len + len(new_tokens)
        if new_len > self.config.max_prompt_len:
            raise ValueError(
                f"Extended prompt length {new_len} exceeds max_prompt_len {self.config.max_prompt_len} "
                f"(= max_seq_len - max_completion_len)."
            )

        updated_tokens = current_tokens + new_tokens
        self._store_anchor_tokens(updated_tokens)
        last_logits = self._forward_token_chunks(ANCHOR_SLOT, current_len, new_tokens)
        self._anchor_last_log_probs = hax.nn.log_softmax(last_logits, axis=self._vocab_axis)

    def _score_completions(self, completions: list[list[int]]) -> list[float]:
        """Clone anchor + forward over padded completions + gather per-completion scores.

        Each completion is padded to ``config.max_completion_len`` so the packed
        queue has a rectangular layout ``(max_batch_size, max_completion_len)``.
        Unused batch rows (``i >= len(completions)``) have ``slot_id = INVALID``
        so no K/V is written for them; padded positions within a used row run
        through the forward but are masked out of the score.
        """
        prompt_len = self._anchor_prompt_len()
        if prompt_len == 0:
            raise RuntimeError("_score_completions called before a prompt was prefilled.")
        if len(completions) == 0:
            return []
        if len(completions) > self.config.max_batch_size:
            raise ValueError(
                f"Got {len(completions)} completions; max_batch_size is {self.config.max_batch_size}"
            )
        if any(len(c) == 0 for c in completions):
            raise ValueError("All completions must be non-empty.")
        M = self.config.max_completion_len
        if any(len(c) > M for c in completions):
            raise ValueError(f"At least one completion exceeds max_completion_len={M}")

        n = len(completions)
        clone_slots = list(range(1, n + 1))
        L = prompt_len
        N = self.config.max_batch_size
        queue_len = N * M

        # 0. Clear any leftover non-anchor slot state from a prior score() call.
        self._release_non_anchor_slots()

        # 1. Clone anchor into each candidate slot. clone_sequence is JIT'd internally and donates gen_state.
        for cslot in clone_slots:
            self.gen_state, _ = self.gen_state.clone_sequence(ANCHOR_SLOT, child_local_id=cslot)

        # 2. Build a flat packed queue plus scoring metadata. Real completion tokens
        # are compacted contiguously at the front; only the tail is padded.
        tokens = np.zeros((queue_len,), dtype=np.int32)
        slots = np.full((queue_len,), INVALID, dtype=np.int32)
        positions = np.full((queue_len,), INVALID, dtype=np.int32)
        candidate_ids = np.zeros((queue_len,), dtype=np.int32)
        target_tokens = np.zeros((queue_len,), dtype=np.int32)
        score_mask = np.zeros((queue_len,), dtype=bool)
        first_tokens = np.zeros((N,), dtype=np.int32)
        offset = 0

        for i, comp in enumerate(completions):
            cslot = clone_slots[i]
            C = len(comp)
            start = offset
            # Real completion tokens at positions [L, L+C).
            tokens[start : start + C] = comp
            slots[start : start + C] = cslot
            positions[start : start + C] = np.arange(L, L + C, dtype=np.int32)
            candidate_ids[start : start + C] = i
            # logits at packed idx (start + k) predict token at pos L+k+1, which is
            # comp[k+1] for k in [0, C-1). Last scored position is start + C - 2.
            for k in range(C - 1):
                target_tokens[start + k] = comp[k + 1]
                score_mask[start + k] = True
            first_tokens[i] = comp[0]
            offset += C

        tokens_na = hax.named(jnp.asarray(tokens, dtype=jnp.int32), axis="position")
        slots_na = hax.named(jnp.asarray(slots, dtype=jnp.int32), axis="position")
        positions_na = hax.named(jnp.asarray(positions, dtype=jnp.int32), axis="position")

        # 3. Forward pass + device-side combined gather (anchor first-token + scoring).
        self.gen_state, logits = _run_forward(self.gen_state, self.model, tokens_na, slots_na, positions_na)
        per_batch = _gather_scores(
            logits,
            self._vocab_axis,
            self._anchor_last_log_probs,
            jnp.asarray(first_tokens),
            jnp.asarray(candidate_ids),
            jnp.asarray(target_tokens),
            jnp.asarray(score_mask),
            batch_size=N,
        )
        per_batch_host = np.asarray(jax.device_get(per_batch))

        scores: list[float] = [float(per_batch_host[i]) for i in range(n)]

        # Clones remain in the pool until the next score() cleans them up at step 0.
        return scores

    # ----- Internal helpers -----

    def _pack_single_seq(
        self, slot: int, start_pos: int, tokens: list[int]
    ) -> tuple[NamedArray, NamedArray, NamedArray]:
        """Pack a single-sequence chunk into fixed-size (tokens, slot_ids, pos_ids) NamedArrays.

        Returns three NamedArrays all with axis ``"position"`` of size ``prompt_chunk_size``.
        The first ``len(tokens)`` entries are the real chunk; the rest are ``INVALID``
        sentinel values that ``allocate_for_seq`` skips.
        """
        max_q = self.config.prompt_chunk_size
        n = len(tokens)
        if n > max_q:
            raise RuntimeError(
                f"chunk (slot={slot}, start={start_pos}, len={n}) exceeds prompt_chunk_size={max_q}"
            )
        tokens_np = np.full((max_q,), INVALID, dtype=np.int32)
        slots_np = np.full((max_q,), INVALID, dtype=np.int32)
        positions_np = np.full((max_q,), INVALID, dtype=np.int32)
        tokens_np[:n] = np.asarray(tokens, dtype=np.int32)
        slots_np[:n] = slot
        positions_np[:n] = np.arange(start_pos, start_pos + n, dtype=np.int32)
        return (
            hax.named(jnp.asarray(tokens_np, dtype=jnp.int32), axis="position"),
            hax.named(jnp.asarray(slots_np, dtype=jnp.int32), axis="position"),
            hax.named(jnp.asarray(positions_np, dtype=jnp.int32), axis="position"),
        )

    def _forward_token_chunks(self, slot: int, start_pos: int, tokens: list[int]) -> NamedArray:
        """Run a single sequence through the decode path in ``prompt_chunk_size`` chunks."""
        if not tokens:
            raise ValueError("tokens must be non-empty")

        max_q = self.config.prompt_chunk_size
        offset = 0
        last_logits: NamedArray | None = None

        while offset < len(tokens):
            chunk = tokens[offset : offset + max_q]
            tokens_na, slots_na, positions_na = self._pack_single_seq(slot, start_pos + offset, chunk)
            self.gen_state, logits = _run_forward(self.gen_state, self.model, tokens_na, slots_na, positions_na)
            last_logits = logits["position", len(chunk) - 1]
            offset += len(chunk)

        assert last_logits is not None
        return last_logits
