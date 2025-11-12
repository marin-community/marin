# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import List, Mapping, Optional, Sequence, Tuple
import hashlib

import jax
import jax.numpy as jnp
import numpy as np

import haliax as hax

from levanter.data import AsyncDataset
from levanter.data.permutation import EpochPermutationDataset
from levanter.models.lm_model import LmExample
from levanter.layers.attention import AttentionMask
from levanter.shapes import Axis
from levanter.data.text import (
    LMMixtureDatasetConfig,
    SingleDatasetLMConfigBase,
    LMTaskConfig,
)


class SpliceDocumentDataset(AsyncDataset[LmExample]):
    """
    Balanced splice dataset.

    Emits constant-length slices of a single document into a fixed-length model frame.
    For each in-document start t (striding by content_stride), we copy K tokens
    and place them flush-right in the frame, i.e., at offset s = S - K. This avoids
    offset sweeps and yields approximately uniform coverage across the document.

    Parameters
    ----------
    Pos: Axis
        Model positional axis; S = Pos.size.
    doc_tokens: np.ndarray
        Tokens of the selected document (length L). Must be 1-D int array.
    pad_token_id: int
        Token ID used for filler. These positions carry loss_mask=0 and form a separate segment.
    eos_token_id: Optional[int]
        Optional EOS token ID. Not strictly required; the attention segment break prevents attending into filler.
    content_length: Optional[int]
        K, constant number of tokens to copy per example. If None, K = min(S, L).
    content_stride: int
        Stride for the in-document start t.
    offset_stride: int
        Ignored in balanced mode (retained for config compatibility).
    content_start_mode: str
        Must be "coverage_balanced" (other modes are not supported here).
    min_copy_len: int
        Minimum number of copied tokens required to keep an example (default 2).
    """

    def __init__(
        self,
        *,
        Pos: Axis,
        doc_tokens: np.ndarray,
        pad_token_id: int,
        eos_token_id: Optional[int] = None,
        content_length: Optional[int] = None,
        content_stride: int = 1,
        offset_stride: int = 1,
        content_start_mode: str = "coverage_balanced",
        min_copy_len: int = 2,
        alpha: float = 0.0,
        rng_key=None,
    ):
        super().__init__()
        if doc_tokens.ndim != 1:
            raise ValueError("doc_tokens must be 1D")
        if content_stride <= 0 or offset_stride <= 0:
            raise ValueError("strides must be positive")

        self.Pos = Pos
        self.S = int(Pos.size)
        self.doc = np.asarray(doc_tokens, dtype=np.int32)
        self.L = int(self.doc.shape[0])
        self.pad_id = int(pad_token_id)
        self.eos_id = int(eos_token_id) if eos_token_id is not None else None
        self.K = int(content_length) if content_length is not None else None
        self.k_t = int(content_stride)
        self.k_s = int(offset_stride)
        self.mode = content_start_mode
        self.min_copy_len = max(1, int(min_copy_len))
        self.alpha = float(alpha)
        # RNG seeded from trainer key for probabilistic edge upsampling
        try:
            if rng_key is not None:
                key_u32 = np.asarray(jax.device_get(rng_key), dtype=np.uint32)
                seed = int(hashlib.blake2b(key_u32.tobytes(), digest_size=8).hexdigest(), 16) & 0x7FFFFFFF
            else:
                seed = 0
        except Exception:
            seed = 0
        self._rng = np.random.default_rng(seed)

        self._pairs: List[Tuple[int, int]] = self._enumerate_pairs()

        # Debug about constructed splice dataset
        try:
            print(
                f"[splice] Pos.size={self.S} doc_len={self.L} content_length={self.K} "
                f"mode={self.mode} t_stride={self.k_t} s_stride={self.k_s} pairs={len(self._pairs)}",
                flush=True,
            )
        except Exception:
            pass

    def _enumerate_pairs(self) -> List[Tuple[int, int]]:
        # Only coverage-balanced mode is supported in this simplified implementation.
        if self.mode not in ("coverage_balanced", None):
            raise ValueError(f"Unsupported content_start_mode: {self.mode}. Only 'coverage_balanced' is supported.")

        K_eff = self.K if self.K is not None else min(self.S, self.L)
        K_eff = int(max(0, min(K_eff, self.S)))
        if K_eff < self.min_copy_len:
            return []

        t_max = max(0, self.L - K_eff)
        t_values = list(range(0, t_max + 1, self.k_t))
        s_fixed = max(0, self.S - K_eff)

        pairs: List[Tuple[int, int]] = []
        # Precompute w_max for normalized Bernoulli duplication
        if self.alpha > 0.0:
            # distance to nearest edge in number of valid starts affecting coverage
            def _d(t):
                left = t + 1
                right = (self.L - K_eff) - t + 1
                return max(1, min(left, right, K_eff))

            w_max = (K_eff / 1.0) ** self.alpha  # at extreme edge d=1
        else:
            w_max = 1.0

        for t in t_values:
            pairs.append((t, s_fixed))
            if self.alpha > 0.0 and w_max > 1.0:
                left = t + 1
                right = (self.L - K_eff) - t + 1
                d = max(1, min(left, right, K_eff))
                w_t = (K_eff / float(d)) ** self.alpha
                # Probability for ONE extra duplicate. Normalized so interior (w=1) => p=0, extreme edge => p=1.
                p = (w_t - 1.0) / (w_max - 1.0)
                p = float(max(0.0, min(1.0, p)))
                if self._rng.random() < p:
                    pairs.append((t, s_fixed))

        return pairs

    async def async_len(self) -> int:
        return len(self._pairs)

    async def final_length_is_known(self) -> bool:
        return True

    def is_finite(self) -> bool:
        return True

    async def current_len(self) -> Optional[int]:
        return len(self._pairs)

    async def get_batch(self, indices: Sequence[int]) -> Sequence[LmExample]:
        out: List[LmExample] = []
        for idx in indices:
            t, s = self._pairs[idx]
            out.append(self._make_example(t, s))
        return out

    def _make_example(self, t: int, s: int) -> LmExample:
        S = self.S
        L = self.L
        doc = self.doc
        K = self.K if self.K is not None else (L - t)
        K = max(0, min(K, L - t))
        copy_len = min(K, L - t, S - s)
        assert copy_len >= self.min_copy_len

        # tokens
        toks = np.full(S, self.pad_id, dtype=np.int32)
        if copy_len > 0:
            toks[s : s + copy_len] = doc[t : t + copy_len]

        # loss mask: predict within doc span, excluding final in-span position
        loss_mask = np.zeros(S, dtype=np.bool_)
        if copy_len >= 2:
            loss_mask[s : s + copy_len - 1] = True

        # segment ids: 0 on prefix [0:s), 1 on [s:S)
        seg_ids = np.zeros(S, dtype=np.int32)
        if s > 0:
            seg_ids[s:] = 1
        else:
            seg_ids[:] = 1  # single segment if s==0

        # Build named arrays and LmExample
        tokens_named = hax.named(jnp.asarray(toks, dtype=jnp.int32), self.Pos)
        loss_named = hax.named(jnp.asarray(loss_mask, dtype=jnp.bool_), self.Pos)
        seg_named = hax.named(jnp.asarray(seg_ids, dtype=jnp.int32), self.Pos)

        ex = LmExample.causal(tokens_named, loss_mask=loss_named, segment_ids=seg_named)
        # Ensure attention uses segment_ids as provided
        ex = LmExample(tokens=ex.tokens, loss_mask=ex.loss_mask, attn_mask=AttentionMask.causal().with_segment_ids(seg_named))
        return ex


@dataclass(frozen=True)
class SpliceSingleDocumentLMConfig(LMTaskConfig):
    """
    LMTaskConfig that builds a SpliceDocumentDataset for training on a single selected document.

    This config wraps an existing dataset config (single or mixture) only to locate and load one
    document's tokens from its training cache, then emits spliced examples as per SpliceDocumentDataset.
    """

    # Use LMMixtureDatasetConfig as the concrete type (works for both single and mixture)
    # Since SingleDatasetLMConfigBase can be wrapped in a mixture with one component
    base: Optional[LMMixtureDatasetConfig] = None
    dataset_name: Optional[str] = None  # required if base is mixture
    doc_index: Optional[int] = None  # if None, select via length policy

    # Length-based selection
    min_doc_length: Optional[int] = None  # require doc length >= this; else fallback
    max_doc_length: Optional[int] = None  # require doc length <= this; else fallback
    doc_select_mode: str = "first"  # one of: "first", "longest", "random"

    # Placement/content schedule (balanced mode only)
    content_length: Optional[int] = None
    content_stride: int = 1
    offset_stride: int = 1  # ignored in balanced mode
    content_start_mode: str = "coverage_balanced"
    min_copy_len: int = 2
    alpha: float = 0.0  # probabilistic edge upsampling strength (0 = off)

    # Training-subset gating (only meaningful for mixture configs with caps)
    restrict_to_training_subset: bool = False
    initial_batch_size: Optional[int] = None

    def build_caches(self, split: str, monitors=True) -> Mapping[str, "TreeCache[dict]"]:
        # Delegate to base config
        if self.base is None:
            raise ValueError("SpliceSingleDocumentLMConfig.base must be provided")
        return self.base.build_caches(split, monitors)

    @property
    def sources(self) -> Mapping[str, "LmDatasetSourceConfigBase"]:  # type: ignore[name-defined]
        # Surface underlying sources for tagging purposes
        return self.base.sources

    def _select_doc_tokens_and_index(
        self, caches: Mapping[str, "TreeCache[dict]"], Pos: Axis
    ) -> tuple[np.ndarray, int]:
        # Determine which dataset name to read from
        if self.base is None:
            raise ValueError("SpliceSingleDocumentLMConfig.base must be provided")

        # Always treat as mixture config (single datasets are wrapped in mixture with one component)
        ds_name = self.dataset_name or next(iter(self.base.sources.keys()))

        if ds_name not in caches:
            raise ValueError(f"Dataset '{ds_name}' not found in caches: {list(caches.keys())}")

        cache = caches[ds_name]
        input_store = cache.store.tree["input_ids"]  # type: ignore[index]

        # Read offsets to compute lengths and select a document efficiently (no token materialization yet)
        num_rows = int(input_store.num_rows)
        offsets = input_store.offsets[0 : num_rows + 1].read().result()
        if len(offsets) > 0:
            offsets = offsets.copy()
            offsets[0] = 0

        lengths = offsets[1:] - offsets[:-1]

        # Helper: choose an index from candidates according to policy
        def _choose_index(cands: np.ndarray) -> int:
            if cands.size == 0:
                # fallback to longest overall
                return int(np.argmax(lengths)) if num_rows > 0 else 0
            mode = (self.doc_select_mode or "first").lower()
            if mode == "first":
                return int(cands[0])
            elif mode == "longest":
                # among candidates, pick the one with max length
                sub = lengths[cands]
                return int(cands[int(np.argmax(sub))])
            elif mode == "random":
                # deterministic seed based on dataset metadata
                h = hashlib.blake2b(digest_size=8)
                h.update((ds_name or "").encode("utf-8"))
                h.update(int(num_rows).to_bytes(8, "little", signed=False))
                # include total tokens and threshold for stability
                total_tokens = int(offsets[-1]) if len(offsets) > 0 else 0
                h.update(int(total_tokens).to_bytes(8, "little", signed=False))
                min_thr = int(self.min_doc_length or 0)
                max_thr = int(self.max_doc_length or 0)
                h.update(min_thr.to_bytes(8, "little", signed=False))
                h.update(max_thr.to_bytes(8, "little", signed=False))
                seed = int.from_bytes(h.digest(), "little", signed=False) & 0x7FFFFFFF
                rng = np.random.default_rng(seed)
                return int(rng.choice(cands))
            else:
                raise ValueError(f"Unsupported doc_select_mode: {self.doc_select_mode}")

        # If a specific doc_index is provided, always use it (ignoring length constraints)
        chosen_index: Optional[int] = None
        if self.doc_index is not None:
            if self.doc_index < 0 or self.doc_index >= num_rows:
                raise ValueError(f"doc_index {self.doc_index} out of bounds for dataset rows={num_rows}")
            chosen_index = int(self.doc_index)
        else:
            # Select by policy with optional min length constraint
            min_len = int(self.min_doc_length) if self.min_doc_length is not None else None
            max_len = int(self.max_doc_length) if self.max_doc_length is not None else None
            mask = np.ones_like(lengths, dtype=bool)
            if min_len is not None:
                mask &= lengths >= min_len
            if max_len is not None:
                mask &= lengths <= max_len
            cands = np.nonzero(mask)[0]
            chosen_index = _choose_index(cands)

        # Compute token span and materialize
        start = int(offsets[chosen_index])
        end = int(offsets[chosen_index + 1])
        if end <= start:
            raise ValueError("Selected document has no tokens")

        data = input_store.data[start:end].read().result().astype(np.int32)
        return data, int(chosen_index)

    def _select_doc_tokens(self, caches: Mapping[str, "TreeCache[dict]"], Pos: Axis) -> np.ndarray:
        data, _idx = self._select_doc_tokens_and_index(caches, Pos)
        return data

    def train_set(
        self,
        Pos: Axis,
        batch_schedule,
        monitors=True,
        *,
        key,
        epochs: Optional[int] = None,
    ) -> AsyncDataset[LmExample]:
        # Build training caches and extract one document
        caches = self.build_caches("train", monitors)
        doc_tokens = self._select_doc_tokens(caches, Pos)

        tok = self.the_tokenizer
        pad_id = tok.pad_token_id or tok.eos_token_id
        eos_id = tok.eos_token_id

        ds: AsyncDataset[LmExample] = SpliceDocumentDataset(
            Pos=Pos,
            doc_tokens=doc_tokens,
            pad_token_id=int(pad_id),
            eos_token_id=int(eos_id) if eos_id is not None else None,
            content_length=self.content_length,
            content_stride=self.content_stride,
            offset_stride=self.offset_stride,
            content_start_mode=self.content_start_mode,
            min_copy_len=self.min_copy_len,
            alpha=self.alpha,
            rng_key=key,
        )

        # Apply shuffling/permutation semantics similarly to SingleDatasetLMConfigBase
        perm_type = self.permutation_type
        if perm_type is None:
            perm_type = "feistel"

        if self.shuffle is True:
            ds = ds.shuffle(key, perm_type=perm_type)
        elif isinstance(self.shuffle, int) and self.shuffle > 0:
            ds = ds.era_shuffle(self.shuffle, key=key, perm_type=perm_type)

        if getattr(self, "shuffle_per_epoch", False):
            # Infinite stream with new permutation each epoch
            ds = EpochPermutationDataset(ds, key=key, perm_type=perm_type)

        return ds

    def train_sets(self, Pos: Axis, monitors=True, *, key, epochs: Optional[int] = None):
        # epochs argument is ignored; training length is controlled by trainer.num_train_steps
        ds = self.train_set(Pos, batch_schedule=None, monitors=monitors, key=key)
        return {"splice": ds}

    def validation_sets(self, Pos: Axis, monitors=True):
        # No default validation for splice; users can add small sampled offsets if desired.
        return {}


class MultiSpliceDocumentDataset(AsyncDataset[LmExample]):
    """
    Multi-document splice dataset with temperature-based document balancing and optional offset jitter.

    For each sample index, deterministically selects (doc_idx, t, s) using seed-based hashing over the index:
    - doc_idx ~ P(doc) ∝ (L_i)^tau over valid documents (those with at least one valid start)
    - t sampled from valid starts with optional edge weighting alpha
    - s = S - K_i with optional offset_jitter moving content left by up to J positions

    If `epoch_length` is not provided, it defaults to sum over documents of the count of valid in-document starts
    (with stride `content_stride`), i.e. ∑_i (floor((L_i - K_i)/k_t) + 1) over valid docs.
    """

    def __init__(
        self,
        *,
        Pos: Axis,
        doc_tokens_list: List[np.ndarray],
        pad_token_id: int,
        eos_token_id: Optional[int] = None,
        content_length: Optional[int] = None,
        content_stride: int = 1,
        offset_stride: int = 1,  # ignored in balanced placement
        content_start_mode: str = "coverage_balanced",
        min_copy_len: int = 2,
        alpha: float = 0.0,
        adaptive_k: bool = True,
        offset_jitter: int = 0,
        jitter_mode: str = "uniform",
        balance_mode: str = "by_temperature",
        balance_tau: float = 1.0,
        epoch_length: Optional[int] = None,
        rng_key=None,
    ):
        super().__init__()
        if content_start_mode not in ("coverage_balanced", None):
            raise ValueError(
                f"Unsupported content_start_mode: {content_start_mode}. Only 'coverage_balanced' is supported."
            )
        if content_stride <= 0:
            raise ValueError("content_stride must be positive")
        if offset_stride <= 0:
            raise ValueError("offset_stride must be positive")
        if jitter_mode not in ("uniform", "low_discrepancy"):
            raise ValueError("jitter_mode must be 'uniform' or 'low_discrepancy'")

        self.Pos = Pos
        self.S = int(Pos.size)
        self.pad_id = int(pad_token_id)
        self.eos_id = int(eos_token_id) if eos_token_id is not None else None
        self.K_cfg = int(content_length) if content_length is not None else None
        self.k_t = int(content_stride)
        self.mode = content_start_mode
        self.min_copy_len = max(1, int(min_copy_len))
        self.alpha = float(alpha)
        self.adaptive_k = bool(adaptive_k)
        self.offset_jitter = max(0, int(offset_jitter))
        self.jitter_mode = jitter_mode
        self.balance_mode = balance_mode
        self.balance_tau = float(balance_tau)

        # No per-sample RNG: ordering is deterministic; shuffling is delegated to PermutationDataset.
        # We keep a seed only for deterministic doc selection in configs (handled there) and for potential
        # future extensions; it's unused in the dataset itself.
        self._seed = 1

        # Normalize and store documents
        self.docs: List[np.ndarray] = [np.asarray(d, dtype=np.int32).reshape(-1) for d in doc_tokens_list]
        self.Ls: np.ndarray = np.asarray([int(d.shape[0]) for d in self.docs], dtype=np.int64)

        # Compute per-document effective K and valid t indices
        self._doc_valid: List[bool] = []
        self._Ks: List[int] = []
        self._t_values: List[np.ndarray] = []
        self._t_cdfs: List[Optional[np.ndarray]] = []

        for L_i in self.Ls.tolist():
            if self.K_cfg is None:
                K_i = min(self.S, L_i)
            else:
                K_i = self.K_cfg
                if self.adaptive_k:
                    K_i = min(K_i, self.S, L_i)
            # If not adaptive and K_i > L_i, this doc may have no valid spans
            if K_i < self.min_copy_len or K_i > self.S or K_i > L_i:
                # mark invalid
                self._doc_valid.append(False)
                self._Ks.append(K_i)
                self._t_values.append(np.asarray([], dtype=np.int64))
                self._t_cdfs.append(None)
                continue

            t_max = L_i - K_i
            if t_max < 0:
                self._doc_valid.append(False)
                self._Ks.append(K_i)
                self._t_values.append(np.asarray([], dtype=np.int64))
                self._t_cdfs.append(None)
                continue

            t_vals = np.arange(0, t_max + 1, self.k_t, dtype=np.int64)
            if t_vals.size == 0:
                self._doc_valid.append(False)
                self._Ks.append(K_i)
                self._t_values.append(np.asarray([], dtype=np.int64))
                self._t_cdfs.append(None)
                continue

            # Optional edge weighting over t via alpha
            if self.alpha > 0.0:
                # d(t) = distance to nearest edge in valid-start space (bounded by K_i)
                left = t_vals + 1
                right = (t_max - t_vals) + 1
                d = np.maximum(1, np.minimum(np.minimum(left, right), K_i)).astype(np.float32)
                weights = (K_i / d) ** float(self.alpha)
                wsum = float(np.sum(weights))
                # If degenerate, fallback to uniform
                if not np.isfinite(wsum) or wsum <= 0.0:
                    cdf = np.linspace(0.0, 1.0, num=t_vals.size + 1, dtype=np.float64)[1:]
                else:
                    probs = weights / wsum
                    cdf = np.cumsum(probs, dtype=np.float64)
            else:
                cdf = np.linspace(0.0, 1.0, num=t_vals.size + 1, dtype=np.float64)[1:]

            self._doc_valid.append(True)
            self._Ks.append(int(K_i))
            self._t_values.append(t_vals)
            self._t_cdfs.append(cdf)

        # Build document sampling probabilities among valid docs using temperature tau
        valid_indices = [i for i, v in enumerate(self._doc_valid) if v]
        if len(valid_indices) == 0:
            raise ValueError("MultiSpliceDocumentDataset: no valid documents after applying K and stride constraints")

        valid_lengths = np.asarray([self.Ls[i] for i in valid_indices], dtype=np.float64)
        if self.balance_mode in ("by_temperature", None):
            tau = float(self.balance_tau)
            # Avoid zero when length=0
            base = np.power(np.maximum(1.0, valid_lengths), tau, dtype=np.float64)
        elif self.balance_mode == "by_coverage":
            base = valid_lengths.astype(np.float64)
        elif self.balance_mode == "by_document":
            base = np.ones_like(valid_lengths, dtype=np.float64)
        else:
            raise ValueError(f"Unsupported balance_mode: {self.balance_mode}")

        base_sum = float(np.sum(base))
        if base_sum <= 0.0:
            probs = np.ones_like(base, dtype=np.float64) / float(base.size)
        else:
            probs = base / base_sum
        self._valid_doc_indices = np.asarray(valid_indices, dtype=np.int64)
        self._doc_cdf = np.cumsum(probs, dtype=np.float64)

        # Default epoch length: sum of valid t counts across docs (finite, deterministic)
        if epoch_length is None:
            epoch_length = int(
                np.sum([self._t_values[i].size for i in range(len(self.docs)) if self._doc_valid[i]], dtype=np.int64)
            )
            epoch_length = max(epoch_length, len(valid_indices))
        self._epoch_length = int(epoch_length)

        # Compute integer quotas per valid doc using largest-remainder rounding
        # probs over valid docs are implicit in self._doc_cdf; reconstruct for quotas
        probs = np.empty_like(self._doc_cdf)
        probs[0] = self._doc_cdf[0]
        probs[1:] = self._doc_cdf[1:] - self._doc_cdf[:-1]
        # Map probs to full-indexed array aligned with valid_indices for convenience
        doc_probs = {int(vi): float(p) for vi, p in zip(self._valid_doc_indices.tolist(), probs.tolist())}
        quotas = np.zeros(len(self.docs), dtype=np.int64)
        remainders: List[Tuple[float, int]] = []
        allocated = 0
        for vi in valid_indices:
            p = doc_probs[int(vi)]
            exact = p * float(self._epoch_length)
            q = int(np.floor(exact))
            quotas[vi] = q
            allocated += q
            remainders.append((exact - float(q), int(vi)))
        # Distribute leftover to largest remainders
        leftover = self._epoch_length - allocated
        if leftover > 0:
            remainders.sort(key=lambda x: (-x[0], x[1]))
            for k in range(min(leftover, len(remainders))):
                quotas[remainders[k][1]] += 1

        # Build pair list deterministically per doc using quantile mapping over t CDF
        pairs: List[Tuple[int, int]] = []  # (doc_idx, t)
        for di in valid_indices:
            Q_i = int(quotas[di])
            if Q_i <= 0:
                continue
            t_vals = self._t_values[di]
            cdf = self._t_cdfs[di]
            T = t_vals.size
            if T == 0:
                continue

            if cdf is None or cdf.size == 0:
                # Uniform coverage over t via evenly spaced indices
                for k in range(Q_i):
                    # Map k to an index in 0..T-1 using midpoints
                    j = int(np.floor(((k + 0.5) * T) / Q_i))
                    if j >= T:
                        j = T - 1
                    pairs.append((int(di), int(t_vals[j])))
            else:
                for k in range(Q_i):
                    u = (k + 0.5) / float(Q_i)
                    j = int(np.searchsorted(cdf, u, side="right"))
                    if j >= T:
                        j = T - 1
                    pairs.append((int(di), int(t_vals[j])))

        # If rounding produced fewer/more than epoch_length due to numeric issues, adjust conservatively
        if len(pairs) > self._epoch_length:
            pairs = pairs[: self._epoch_length]
        elif len(pairs) < self._epoch_length:
            # Pad by cycling through valid docs' first t
            needed = self._epoch_length - len(pairs)
            cycle = []
            for di in valid_indices:
                if self._t_values[di].size > 0:
                    cycle.append((int(di), int(self._t_values[di][0])))
            while needed > 0 and len(cycle) > 0:
                for item in cycle:
                    pairs.append(item)
                    needed -= 1
                    if needed <= 0:
                        break

        self._pairs: List[Tuple[int, int]] = pairs

        # Debug summary
        try:
            pairs_per_doc = [int(self._t_values[i].size) for i in range(len(self.docs)) if self._doc_valid[i]]
            dbg = (
                f"[multi_splice] num_docs={len(self.docs)} valid_docs={len(valid_indices)} "
                f"Pos.size={self.S} K_cfg={self.K_cfg} adaptive_k={self.adaptive_k} "
                f"balance={self.balance_mode}(tau={self.balance_tau}) epoch_len={self._epoch_length} "
                f"offset_jitter={self.offset_jitter}"
            )
            print(dbg, flush=True)
            print(f"[multi_splice] pairs_per_doc={pairs_per_doc}", flush=True)
        except Exception:
            pass

    async def async_len(self) -> int:
        return self._epoch_length

    async def final_length_is_known(self) -> bool:
        return True

    def is_finite(self) -> bool:
        return True

    async def current_len(self) -> Optional[int]:
        return self._epoch_length


    async def get_batch(self, indices: Sequence[int]) -> Sequence[LmExample]:
        out: List[LmExample] = []
        for idx in indices:
            doc_idx, t = self._pairs[idx]
            K_i = self._Ks[doc_idx]
            # Deterministic offset jitter derived from index
            s_base = max(0, self.S - int(K_i))
            if self.offset_jitter > 0 and s_base > 0:
                j = idx % (self.offset_jitter + 1)
                j = min(j, s_base)
                s = s_base - j
            else:
                s = s_base
            out.append(self._make_example(doc_idx, t, s))
        return out

    def _make_example(self, doc_idx: int, t: int, s: int) -> LmExample:
        S = self.S
        doc = self.docs[doc_idx]
        L = int(doc.shape[0])
        K = self._Ks[doc_idx]
        # In balanced placement, copy_len is clamped by K and frame bounds
        copy_len = min(K, L - t, S - s)
        assert copy_len >= self.min_copy_len

        # tokens
        toks = np.full(S, self.pad_id, dtype=np.int32)
        if copy_len > 0:
            toks[s : s + copy_len] = doc[t : t + copy_len]

        # loss mask: ones on in-span except last
        loss_mask = np.zeros(S, dtype=np.bool_)
        if copy_len >= 2:
            loss_mask[s : s + copy_len - 1] = True

        # segment ids: 0 on prefix, 1 on [s:]
        seg_ids = np.zeros(S, dtype=np.int32)
        if s > 0:
            seg_ids[s:] = 1
        else:
            seg_ids[:] = 1

        tokens_named = hax.named(jnp.asarray(toks, dtype=jnp.int32), self.Pos)
        loss_named = hax.named(jnp.asarray(loss_mask, dtype=jnp.bool_), self.Pos)
        seg_named = hax.named(jnp.asarray(seg_ids, dtype=jnp.int32), self.Pos)

        ex = LmExample.causal(tokens_named, loss_mask=loss_named, segment_ids=seg_named)
        ex = LmExample(tokens=ex.tokens, loss_mask=ex.loss_mask, attn_mask=AttentionMask.causal().with_segment_ids(seg_named))
        return ex


@dataclass(frozen=True)
class SpliceMultiDocumentLMConfig(LMTaskConfig):
    """
    LMTaskConfig for training on multiple documents with splice examples.

    Selects `num_docs` documents from a base dataset using length filters and a selection policy, then
    constructs a `MultiSpliceDocumentDataset` with temperature-based balancing and optional offset jitter.
    """

    base: Optional[LMMixtureDatasetConfig] = None
    dataset_name: Optional[str] = None

    # Multi-document selection
    num_docs: int = 10
    min_doc_length: Optional[int] = None
    max_doc_length: Optional[int] = None
    doc_select_mode: str = "longest"  # "longest", "shortest", "random", "first"

    # Splice parameters (same as single-doc)
    content_length: Optional[int] = None
    content_stride: int = 1
    offset_stride: int = 1
    content_start_mode: str = "coverage_balanced"
    min_copy_len: int = 2
    alpha: float = 0.0
    adaptive_k: bool = True
    offset_jitter: int = 0
    jitter_mode: str = "uniform"

    # Balancing
    balance_mode: str = "by_temperature"  # or "by_coverage", "by_document"
    balance_tau: float = 1.0

    # Streaming/epoch sizing
    epoch_length: Optional[int] = None

    # Training-subset gating (for compatibility)
    restrict_to_training_subset: bool = False
    initial_batch_size: Optional[int] = None
    # Strictness: if not enough docs satisfy filters, fail loudly
    strict_num_docs: bool = True

    def build_caches(self, split: str, monitors=True) -> Mapping[str, "TreeCache[dict]"]:
        if self.base is None:
            raise ValueError("SpliceMultiDocumentLMConfig.base must be provided")
        return self.base.build_caches(split, monitors)

    @property
    def sources(self) -> Mapping[str, "LmDatasetSourceConfigBase"]:  # type: ignore[name-defined]
        return self.base.sources

    def _select_multiple_docs(
        self, caches: Mapping[str, "TreeCache[dict]"], num_docs: int
    ) -> tuple[List[np.ndarray], List[int]]:
        if self.base is None:
            raise ValueError("SpliceMultiDocumentLMConfig.base must be provided")

        ds_name = self.dataset_name or next(iter(self.base.sources.keys()))
        if ds_name not in caches:
            raise ValueError(f"Dataset '{ds_name}' not found in caches: {list(caches.keys())}")

        cache = caches[ds_name]
        input_store = cache.store.tree["input_ids"]  # type: ignore[index]

        num_rows = int(input_store.num_rows)
        offsets = input_store.offsets[0 : num_rows + 1].read().result()
        if len(offsets) > 0:
            offsets = offsets.copy()
            offsets[0] = 0
        lengths = offsets[1:] - offsets[:-1]

        # Build candidate indices under length filters
        mask = np.ones_like(lengths, dtype=bool)
        if self.min_doc_length is not None:
            mask &= lengths >= int(self.min_doc_length)
        if self.max_doc_length is not None:
            mask &= lengths <= int(self.max_doc_length)
        candidates = np.nonzero(mask)[0]

        mode = (self.doc_select_mode or "longest").lower()
        selected: List[int] = []
        if mode == "first":
            selected = candidates[:num_docs].tolist()
        elif mode == "longest":
            order = np.argsort(-lengths[candidates], kind="stable")
            selected = candidates[order[:num_docs]].tolist()
        elif mode == "shortest":
            order = np.argsort(lengths[candidates], kind="stable")
            selected = candidates[order[:num_docs]].tolist()
        elif mode == "random":
            # deterministic RNG based on dataset metadata
            h = hashlib.blake2b(digest_size=8)
            h.update((ds_name or "").encode("utf-8"))
            h.update(int(num_rows).to_bytes(8, "little", signed=False))
            tot = int(offsets[-1]) if len(offsets) > 0 else 0
            h.update(int(tot).to_bytes(8, "little", signed=False))
            h.update(int(self.num_docs).to_bytes(8, "little", signed=False))
            seed = int.from_bytes(h.digest(), "little", signed=False) & 0x7FFFFFFF
            rng = np.random.default_rng(seed)
            if candidates.size <= num_docs:
                selected = candidates.tolist()
            else:
                selected = rng.choice(candidates, size=num_docs, replace=False).tolist()
        else:
            raise ValueError(f"Unsupported doc_select_mode: {self.doc_select_mode}")

        # If not enough matched candidates, either fail loudly (strict) or backfill with longest overall
        if len(selected) < num_docs:
            if self.strict_num_docs:
                matched = len(selected)
                min_len = self.min_doc_length if self.min_doc_length is not None else None
                max_len = self.max_doc_length if self.max_doc_length is not None else None
                raise ValueError(
                    f"SpliceMultiDocumentLMConfig: only {matched} documents satisfy length constraints "
                    f"(min_doc_length={min_len}, max_doc_length={max_len}); required num_docs={num_docs}."
                )
            else:
                remaining = num_docs - len(selected)
                all_order = np.argsort(-lengths, kind="stable")
                for idx in all_order:
                    if int(idx) in selected:
                        continue
                    selected.append(int(idx))
                    remaining -= 1
                    if remaining <= 0:
                        break

        # Materialize tokens for selected indices
        docs: List[np.ndarray] = []
        sel_indices: List[int] = []
        for i in selected:
            start = int(offsets[i])
            end = int(offsets[i + 1])
            if end <= start:
                continue
            data = input_store.data[start:end].read().result().astype(np.int32)
            if data.size == 0:
                continue
            docs.append(data)
            sel_indices.append(int(i))

        return docs, sel_indices

    def train_set(
        self,
        Pos: Axis,
        batch_schedule,
        monitors=True,
        *,
        key,
        epochs: Optional[int] = None,
    ) -> AsyncDataset[LmExample]:
        caches = self.build_caches("train", monitors)

        docs, idxs = self._select_multiple_docs(caches, self.num_docs)
        if len(docs) == 0:
            raise ValueError("SpliceMultiDocumentLMConfig: no documents selected/materialized")

        tok = self.the_tokenizer
        pad_id = tok.pad_token_id or tok.eos_token_id
        eos_id = tok.eos_token_id

        ds: AsyncDataset[LmExample] = MultiSpliceDocumentDataset(
            Pos=Pos,
            doc_tokens_list=docs,
            pad_token_id=int(pad_id),
            eos_token_id=int(eos_id) if eos_id is not None else None,
            content_length=self.content_length,
            content_stride=self.content_stride,
            offset_stride=self.offset_stride,
            content_start_mode=self.content_start_mode,
            min_copy_len=self.min_copy_len,
            alpha=self.alpha,
            adaptive_k=self.adaptive_k,
            offset_jitter=self.offset_jitter,
            jitter_mode=self.jitter_mode,
            balance_mode=self.balance_mode,
            balance_tau=self.balance_tau,
            epoch_length=self.epoch_length,
            rng_key=key,
        )

        # Shuffling/permutation consistent with SingleDataset semantics
        perm_type = self.permutation_type
        if perm_type is None:
            perm_type = "feistel"

        if self.shuffle is True:
            ds = ds.shuffle(key, perm_type=perm_type)
        elif isinstance(self.shuffle, int) and self.shuffle > 0:
            ds = ds.era_shuffle(self.shuffle, key=key, perm_type=perm_type)

        if getattr(self, "shuffle_per_epoch", False):
            ds = EpochPermutationDataset(ds, key=key, perm_type=perm_type)

        return ds

    def train_sets(self, Pos: Axis, monitors=True, *, key, epochs: Optional[int] = None):
        ds = self.train_set(Pos, batch_schedule=None, monitors=monitors, key=key)
        return {"multi_splice": ds}

    def validation_sets(self, Pos: Axis, monitors=True):
        return {}
