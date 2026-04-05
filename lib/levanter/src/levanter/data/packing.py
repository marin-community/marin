# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""
Implements sequence packing, mostly for doing evaluation on lots of short sequences.

Our strategy is basically to maintain a pool of SequencePackers, each of which can hold a fixed number of tokens
(and a maximum number of segments). We then iterate over the sequences, adding them to the packers if they fit, and
yielding the packed examples when they are full.

This achieves about a 90% "real token" rate, compared to like 10% without packing.
"""
import asyncio
from dataclasses import dataclass
from typing import Iterable, Iterator, Literal, Sequence, TypeVar

import haliax as hax
import jax
import jax.numpy as jnp
import numpy as np
import tensorstore as ts
from jaxtyping import PyTree

from levanter.data import AsyncDataset
from levanter.layers.attention import AttentionMask
from levanter.models.lm_model import LmExample
from levanter.store.jagged_array import JaggedArrayStore
from levanter.utils.jax_utils import leaf_key_paths, local_cpu_mesh, tree_broadcast_to

# cf https://github.com/tensorflow/tensor2tensor/blob/bafdc1b67730430d38d6ab802cbd51f9d053ba2e/tensor2tensor/data_generators/generator_utils.py#L623

# todo should we use something like this: https://arxiv.org/pdf/2107.02027?

T = TypeVar("T", bound=PyTree)
L = TypeVar("L")


# Python 3.10 can't handle this
# @dataclass(frozen=True)
# class LeafType:
#     leaf_type: type
#
#     def __class_getitem__(cls, item):
#         return cls(item)
#
#
# WithLeaf: TypeAlias = Annotated[T, LeafType[L]]


class SequencePacker:
    """
    Packs sequences into a single LmExample.
    """

    def __init__(self, Pos: hax.Axis, max_pack_size: int, pad_token: int):
        self.Pos = Pos
        self._ids: list[int] = []
        self._segment_ids: list[int] = []
        self._loss_weight: list[float] = []
        self.num_segments = 0
        self.pad_token = pad_token
        self.max_pack_size = max_pack_size
        assert pad_token is not None, "pad_token must be set"

    def can_pack(self, ids: list[int]) -> bool:
        return len(ids) + len(self._ids) <= self.Pos.size and self.num_segments < self.max_pack_size

    def add_example(self, ids: list[int], loss_weight: list[float] | np.ndarray, segment_id: int | None = None):
        if len(ids) != len(loss_weight):
            raise ValueError("ids and loss_weight must have the same length")

        if len(ids) == 0:
            return

        if len(ids) + len(self._ids) > self.Pos.size:
            raise ValueError("Too many tokens")

        if self.num_segments >= self.max_pack_size:
            raise ValueError("Too many segments")

        self._ids.extend(ids)
        if segment_id is None:
            segment_id = self.num_segments

        self.num_segments += 1

        self._segment_ids.extend([segment_id] * len(ids))

        self._loss_weight.extend(loss_weight)

    def pack(self) -> LmExample:
        ids = self._ids + [self.pad_token] * (self.Pos.size - len(self._ids))

        segment_ids = self._segment_ids + [-1] * (self.Pos.size - len(self._segment_ids))

        loss_weight = self._loss_weight + [0.0] * (self.Pos.size - len(self._loss_weight))

        with local_cpu_mesh():
            tokens = hax.named(ids, self.Pos).astype(jnp.int32)
            segment_ids = hax.named(segment_ids, self.Pos).astype(jnp.int32)
            loss_weight = hax.named(loss_weight, self.Pos).astype(jnp.float32)

            attn_mask = AttentionMask.causal().with_segment_ids(segment_ids)

            return LmExample(tokens=tokens, loss_weight=loss_weight, attn_mask=attn_mask)


@dataclass(frozen=True)
class PromptCompletion:
    ids: list[int]
    prompt_length: int
    segment_id: int | None = None

    def __post_init__(self):
        if len(self.ids) == 0:
            raise ValueError("PromptCompletion must have at least one token")

        # check that there is at least one token in the response
        if len(self.ids) <= self.prompt_length:
            raise ValueError(
                f"PromptCompletion must have strictly more tokens than the prompt length. Got {len(self.ids)} tokens"
                f" and prompt length {self.prompt_length}"
            )


def pack_prompt_completions(
    Pos: hax.Axis,
    sequences: Iterable[PromptCompletion],
    pad_token: int,
    max_segments_per_example: int = 64,
    max_buffered_examples: int = 64,
) -> Iterator[LmExample]:
    """
    Packs a list of prompt completions into LmExamples using the SequencePacker
    """

    packers = [SequencePacker(Pos, max_segments_per_example, pad_token)]

    for sequence in sequences:
        loss_weight = (np.arange(len(sequence.ids)) >= sequence.prompt_length - 1).astype(np.float32)
        loss_weight[-1] = 0
        assert np.any(loss_weight)

        for packer in packers:
            if packer.can_pack(sequence.ids):
                packer.add_example(sequence.ids, loss_weight, sequence.segment_id)

                if packer.num_segments == max_segments_per_example:
                    yield packer.pack()
                    packers.remove(packer)
                break
        else:
            # no packer could fit the example, create a new one
            packer = SequencePacker(Pos, max_segments_per_example, pad_token)
            packer.add_example(sequence.ids, loss_weight, sequence.segment_id)
            packers.append(packer)

        while len(packers) >= max_buffered_examples:
            yield packers.pop(0).pack()

    for packer in packers:
        yield packer.pack()


def per_segment_loss(
    packed_example: LmExample, losses: hax.NamedArray, max_Segments: hax.Axis
) -> tuple[hax.NamedArray, hax.NamedArray]:
    """
    Returns a pair of arrays of shape (Segments,), where:

    * the first array is segment ids
    * the second is loss per segment.

    This code is designed to run in a jit-compiled function, meaning we have to careful of shapes
    """

    assert packed_example.attn_mask.segment_ids is not None, "segment_ids must be set in the AttentionMask"

    segment_ids = packed_example.attn_mask.segment_ids
    if isinstance(segment_ids, tuple):
        segment_ids = segment_ids[0]

    assert (
        segment_ids.ndim == 1
    ), f"Expected segment_ids to be 1D, got {segment_ids.ndim}. Use vmap if you have multiple examples"
    Pos = packed_example.tokens.axes[0]

    # mask out padding etc
    masked_losses = losses * packed_example.loss_weight

    # sum the losses for each segment
    unique_segment_ids = _unique_segment_ids(max_Segments, segment_ids)

    # Create a mask matrix where each row corresponds to a unique segment
    segment_mask = unique_segment_ids == segment_ids.broadcast_axis(max_Segments)

    segment_mask = segment_mask.astype(masked_losses.dtype)

    segment_losses = hax.dot(segment_mask, masked_losses, axis=Pos)

    return unique_segment_ids, segment_losses


def _unique_segment_ids(max_Segments, segment_ids):
    # Extract unique segment IDs with padding
    # TODO: add unique to haliax
    unique_segment_ids = jnp.unique(segment_ids.array, size=max_Segments.size, fill_value=-1)
    unique_segment_ids = hax.named(unique_segment_ids, max_Segments)
    return unique_segment_ids


def per_segment_correct(
    packed_example: LmExample, correct: hax.NamedArray, max_Segments: hax.Axis
) -> tuple[hax.NamedArray, hax.NamedArray]:
    """
    Returns a pair of arrays of shape (max_segments,), where:

    * the first array is segment ids
    * the second is whether all tokens in the segment are correct.

    This code is designed to run in a jit-compiled function, meaning we have to careful of shapes

    correct is a boolean array of the same shape as the losses array indicating whether the token was correct
    """

    assert packed_example.attn_mask.segment_ids is not None, "segment_ids must be set in the AttentionMask"

    segment_ids = packed_example.attn_mask.segment_ids
    if isinstance(segment_ids, tuple):
        segment_ids = segment_ids[0]

    assert (
        segment_ids.ndim == 1
    ), f"Expected segment_ids to be 1D, got {segment_ids.ndim}. Use vmap if you have multiple examples"

    Pos = packed_example.tokens.axes[0]

    # mask out padding etc
    valid_positions = packed_example.loss_weight > 0
    masked_correct = hax.logical_or(correct, hax.logical_not(valid_positions))

    # sum the losses for each segment
    # Extract unique segment IDs with padding
    unique_segment_ids = _unique_segment_ids(max_Segments, segment_ids)

    # Create a mask matrix where each row corresponds to a unique segment
    segment_mask = unique_segment_ids == segment_ids.broadcast_axis(max_Segments)

    segment_mask = segment_mask.astype(masked_correct.dtype)

    segment_correct = hax.all(hax.where(segment_mask, masked_correct, True), axis=Pos)

    return unique_segment_ids, segment_correct


def greedy_pack_prompt_completions(
    Pos: hax.Axis,
    sequences: Iterable[PromptCompletion],
    pad_token: int,
    max_segments_per_example: int = 64,
) -> list[LmExample]:
    """
    Greedy packing of prompt completions into LmExamples using [pack_documents][]
    """

    def make_loss_weight(id, prompt_length):
        loss_weight = (np.arange(len(id)) >= prompt_length - 1).astype(np.float32)
        loss_weight[-1] = 0
        return loss_weight

    # Convert sequences to lists for easier access
    sequences = list(sequences)
    ids = [sequence.ids for sequence in sequences]

    # Pack documents based on their lengths
    packs = pack_documents(
        lengths=np.array([len(token_ids) for token_ids in ids]),
        max_length=Pos.size,
        max_segments_per_example=max_segments_per_example,
        slice_strategy="right",
    )

    out = []

    # Yield packed examples for normal sequences
    for docs_in_pack in packs:
        # Get the documents in this pack
        pack_sequences = [sequences[i] for i in docs_in_pack]
        pack_prompt_lengths = [sequence.prompt_length for sequence in pack_sequences]

        # Concatenate the IDs and create loss weights
        concat_ids = []
        concat_loss_weight = []
        segment_ids = []

        for doc_id, seq, prompt_len in zip(docs_in_pack, pack_sequences, pack_prompt_lengths):
            concat_ids.extend(seq.ids)
            concat_loss_weight.extend(make_loss_weight(seq.ids, prompt_len))
            segment_ids.extend([doc_id] * len(seq.ids))

        # Pad to max length
        pad_length = Pos.size - len(concat_ids)

        if pad_length > 0:
            concat_ids.extend([pad_token] * pad_length)
            concat_loss_weight.extend([0] * pad_length)
            segment_ids.extend([-1] * pad_length)
        elif pad_length < 0:
            # too long, this should only happen if there's 1 document in the pack
            if len(pack_sequences) != 1:
                raise ValueError("Too many tokens in a pack with more than one document")
            concat_ids = concat_ids[-Pos.size :]
            concat_loss_weight = concat_loss_weight[-Pos.size :]
            segment_ids = segment_ids[-Pos.size :]

        # Create the LmExample
        tokens = hax.named(np.array(concat_ids), Pos)
        loss_weight = hax.named(np.array(concat_loss_weight), Pos)
        segment_ids = hax.named(np.array(segment_ids), Pos)
        attn_mask = AttentionMask.causal().with_segment_ids(segment_ids)

        out.append(LmExample(tokens=tokens, loss_weight=loss_weight, attn_mask=attn_mask))

    return out


def pack_documents(
    lengths: PyTree[np.ndarray],
    max_length: PyTree[int],
    max_segments_per_example: int | None = None,
    slice_strategy: Literal["left", "right", "raise", "drop"] = "raise",
) -> list[range]:
    """
    Greedily pack documents into contiguous groups without storing full token ranges.

    Args:
        lengths: A PyTree of numpy arrays, each containing the lengths of documents for a leaf.
            Each array should be of length n_docs, where n_docs is the number of documents.
            The i-th document has length lengths[i].
        max_length: A PyTree of integers, each specifying the maximum number of tokens allowed per pack for that leaf
        max_segments_per_example: Optional maximum number of documents per pack
        slice_strategy: One of "left", "right", "raise", or "drop".

    Returns:
        A list of ranges, where each range represents the document indices in a pack
    """
    # Input validation
    if max_segments_per_example is not None and (
        not isinstance(max_segments_per_example, int) or max_segments_per_example <= 0
    ):
        raise ValueError(f"max_segments_per_example must be a positive integer, got {max_segments_per_example}")

    # Broadcast max_length to match the structure of lengths
    max_length_tree = tree_broadcast_to(max_length, lengths)

    lengths_leaves = jax.tree.leaves(lengths)
    max_length_leaves = jax.tree.leaves(max_length_tree)
    leaf_names = jax.tree.leaves(leaf_key_paths(lengths))

    if len(lengths_leaves) != len(max_length_leaves):
        raise ValueError("Lengths and max_length PyTrees must have the same number of leaves.")

    # Check that all leaves have the same number of documents.
    n_docs = None
    for lens in lengths_leaves:
        if n_docs is None:
            n_docs = len(lens)
        elif len(lens) != n_docs:
            raise ValueError("All leaves must have the same number of documents.")

    if n_docs is None:
        raise ValueError("Could not determine the number of documents from lengths.")

    if slice_strategy not in ["left", "right", "raise", "drop"]:
        raise ValueError(f"slice_strategy must be one of 'left', 'right', 'raise', or 'drop', got {slice_strategy}")

    # Validate document lengths
    drop_mask = np.ones(n_docs, dtype=bool) if slice_strategy == "drop" else None
    for lens, allowed, leaf_name in zip(lengths_leaves, max_length_leaves, leaf_names):
        for i in range(n_docs):
            if lens[i] > allowed:
                if drop_mask is not None:
                    drop_mask[i] = False
                    continue
                if slice_strategy == "raise":
                    raise ValueError(
                        f"Document {i} in leaf '{leaf_name}' has length {lens[i]} which exceeds "
                        f"maximum allowed length {allowed}. Consider setting slice_strategy to 'left', 'right', "
                        "'drop', or increasing max_length."
                    )

    pack_doc_ranges = []
    i = 0
    while i < n_docs:
        if drop_mask is not None and not drop_mask[i]:
            i += 1
            continue
        start = i
        total_segments = 0
        # Accumulate documents while for each leaf the token span remains within the allowed max.
        while i < n_docs:
            if drop_mask is not None and not drop_mask[i]:
                break
            # Check optional segment constraint: if adding one more document would exceed max_segments_per_example.
            if max_segments_per_example is not None and (total_segments + 1) > max_segments_per_example:
                break
            # For each leaf, check if adding document i would keep the token count within allowed capacity.
            valid = True
            end_pack_after_this = False
            for lens, allowed, leaf_name in zip(lengths_leaves, max_length_leaves, leaf_names, strict=True):
                # Compute token count from document start to document i+1.
                token_sum = sum(lens[start : i + 1])
                if token_sum > allowed:
                    if i == start:
                        if slice_strategy == "raise":
                            raise ValueError(
                                f"Document {i} in leaf '{leaf_name}' has length {lens[i]} which exceeds "
                                f"maximum allowed length {allowed}. Consider setting slice_strategy to 'left', "
                                "'right', 'drop', or increasing max_length."
                            )
                        if slice_strategy == "drop":
                            valid = False
                            break
                        valid = True
                        end_pack_after_this = True
                        break
                    valid = False
                    break
            if not valid:
                break
            total_segments += 1
            i += 1
            if end_pack_after_this:
                break

        # If no document could be added (i.e. a single document exceeds capacity)
        if i == start:
            if slice_strategy != "left" and slice_strategy != "right":
                raise ValueError(f"Document {start} exceeds allowed capacity.")
            else:
                i = start + 1

        pack_doc_ranges.append(range(start, i))
    return pack_doc_ranges


def pack_documents_sorted(
    lengths: PyTree[np.ndarray],
    max_length: PyTree[int],
    max_segments_per_example: int | None = None,
    slice_strategy: Literal["left", "right", "raise", "drop"] = "raise",
) -> list[list[int]]:
    """Pack documents using sorted greedy: sort by length descending, then greedily pack consecutive docs.

    This reorders documents so that similar-length docs are adjacent, which improves greedy packing
    efficiency. Returns non-contiguous doc index lists.
    """
    max_length_tree = tree_broadcast_to(max_length, lengths)
    lengths_leaves = jax.tree.leaves(lengths)
    max_length_leaves = jax.tree.leaves(max_length_tree)

    # Use the first leaf's lengths as the primary sort key
    primary_lengths = lengths_leaves[0].copy()
    primary_max = max_length_leaves[0]

    # Clamp lengths for sorting (oversized docs treated as max_length)
    effective_lengths = np.minimum(primary_lengths, primary_max)

    # Sort descending by length
    perm = np.argsort(-effective_lengths)

    # Apply permutation to all length leaves
    permuted_lengths = jax.tree.map(lambda lens: lens[perm], lengths)

    # Run the standard greedy packer on the permuted order
    contiguous_packs = pack_documents(
        permuted_lengths,
        max_length,
        max_segments_per_example=max_segments_per_example,
        slice_strategy=slice_strategy,
    )

    # Map permuted indices back to original doc indices
    return [perm[list(r)].tolist() for r in contiguous_packs]


def pack_documents_bfd(
    lengths: PyTree[np.ndarray],
    max_length: PyTree[int],
    max_segments_per_example: int | None = None,
    slice_strategy: Literal["left", "right", "raise", "drop"] = "raise",
) -> list[list[int]]:
    """Best-fit decreasing bin packing for documents.

    Sort documents by length descending. For each document, find the open bin with the least
    remaining capacity that still fits. Uses a bisect-based sorted list for O(n log n) performance.

    Returns a list of doc-index lists (non-contiguous).
    """
    import bisect

    max_length_tree = tree_broadcast_to(max_length, lengths)
    lengths_leaves = jax.tree.leaves(lengths)
    max_length_leaves = jax.tree.leaves(max_length_tree)

    primary_lengths = lengths_leaves[0].copy()
    primary_max = max_length_leaves[0]

    # Clamp oversized docs to max_length (they'll be sliced later)
    effective_lengths = np.minimum(primary_lengths, primary_max)

    # Sort descending by length
    order = np.argsort(-effective_lengths)

    bins: list[list[int]] = []  # bin_index -> list of doc indices
    bin_remaining: list[int] = []  # bin_index -> remaining capacity
    bin_segments: list[int] = []  # bin_index -> number of segments

    # Sorted list of (remaining_capacity, bin_index) for O(log n) best-fit lookup
    # We maintain this sorted so we can bisect to find the smallest remaining >= doc_len
    sorted_bins: list[tuple[int, int]] = []  # sorted by remaining capacity

    for doc_idx_sorted in order:
        doc_idx = int(doc_idx_sorted)
        doc_len = int(effective_lengths[doc_idx])

        # Binary search for the smallest remaining capacity >= doc_len
        search_pos = bisect.bisect_left(sorted_bins, (doc_len, -1))

        best_bin = -1
        # Scan forward from search_pos to find a valid bin (respecting segment cap)
        scan_limit = min(search_pos + 50, len(sorted_bins))  # bounded scan for segment-cap misses
        for i in range(search_pos, scan_limit):
            remaining, b = sorted_bins[i]
            if remaining < doc_len:
                continue
            if max_segments_per_example is not None and bin_segments[b] >= max_segments_per_example:
                continue
            best_bin = b
            # Remove old entry from sorted list
            sorted_bins.pop(i)
            break

        if best_bin >= 0:
            bins[best_bin].append(doc_idx)
            bin_remaining[best_bin] -= doc_len
            bin_segments[best_bin] += 1
            new_remaining = bin_remaining[best_bin]
            # Re-insert with updated remaining capacity
            bisect.insort(sorted_bins, (new_remaining, best_bin))
        else:
            # Open a new bin
            new_bin_idx = len(bins)
            bins.append([doc_idx])
            new_remaining = primary_max - doc_len
            bin_remaining.append(new_remaining)
            bin_segments.append(1)
            bisect.insort(sorted_bins, (new_remaining, new_bin_idx))

    return bins


class GreedyPrepackedDataset(AsyncDataset[tuple[T, T]]):
    """
    Prepacks a dataset into a new dataset where examples are packed into a single example.

    As per usual, I can't help but make this generic.

    Args:
        dataset: A PyTree of JaggedArrayStore objects, each representing a leaf in the dataset.
        max_length: A PyTree of integers, each representing the maximum number of tokens allowed per leaf.
        max_segments_per_example: Maximum number of documents that can be packed into a single example.
        pad_with_zeros: If True, pad examples to max_length with zeros. If False, return examples as-is.
        slice_strategy: One of "left", "right", "raise", or "drop". Determines how to handle examples that exceed max_length:
            - "left": Slice from the beginning of the example
            - "right": Slice from the end of the example
            - "raise": Raise an error when an example exceeds max_length
            - "drop": Drop examples that exceed max_length
        packing_strategy: One of "greedy" (default), "sorted", or "bfd".
            - "greedy": Pack consecutive documents in cache order (current behavior).
            - "sorted": Sort documents by length descending, then pack greedily.
            - "bfd": Best-fit decreasing bin packing for near-optimal utilization.
    """

    def __init__(
        self,
        dataset: T,  # PyTree[JaggedArrayStore],
        max_length: int | T,  # PyTree[int],
        max_segments_per_example: int | None = None,
        pad_with_zeros: bool = True,
        slice_strategy: Literal["left", "right", "raise", "drop"] = "raise",
        packing_strategy: Literal["greedy", "sorted", "bfd"] = "greedy",
    ):
        """
        Args:
            dataset: A PyTree of JaggedArrayStore objects, each representing a leaf in the dataset.
            max_length: A PyTree of integers, each representing the maximum number of tokens allowed per leaf.
            max_segments_per_example: Maximum number of documents that can be packed into a single example.
            pad_with_zeros: If True, pad examples to max_length with zeros. If False, return examples as-is.
            slice_strategy: One of "left", "right", "raise", or "drop". Determines how to handle examples that exceed max_length.
            packing_strategy: One of "greedy", "sorted", or "bfd". Controls how documents are assigned to packs.
        """
        super().__init__()

        if slice_strategy not in ["left", "right", "raise", "drop"]:
            raise ValueError(
                f"slice_strategy must be one of 'left', 'right', 'raise', or 'drop', got {slice_strategy}"
            )

        if packing_strategy not in ["greedy", "sorted", "bfd"]:
            raise ValueError(f"packing_strategy must be one of 'greedy', 'sorted', or 'bfd', got {packing_strategy}")

        self.dataset = dataset
        self.max_length = max_length
        self.max_segments_per_example = max_segments_per_example
        self.pad_with_zeros = pad_with_zeros
        self.slice_strategy = slice_strategy
        self.packing_strategy = packing_strategy

        _offsets = jax.tree.map(lambda store: store.offsets[0 : store.num_rows + 1].read(), self.dataset)
        self._offsets = jax.tree.map(lambda fut: fut.result(), _offsets)

        def diff_offsets(offsets: np.ndarray):
            # fine to mutate since we have a copy
            # the array store has the number of rows in the 0th offset
            offsets[0] = 0
            return offsets[1:] - offsets[:-1]

        # Convert offsets to lengths
        self._lengths = jax.tree.map(diff_offsets, self._offsets)

        # Build pack indices using the selected strategy
        # Type is list[range] for greedy, list[list[int]] for sorted/bfd
        self._pack_indices: list
        if packing_strategy == "sorted":
            self._pack_indices = pack_documents_sorted(
                self._lengths,
                max_length,
                max_segments_per_example,
                slice_strategy=slice_strategy,
            )
        elif packing_strategy == "bfd":
            self._pack_indices = pack_documents_bfd(
                self._lengths,
                max_length,
                max_segments_per_example,
                slice_strategy=slice_strategy,
            )
        else:
            self._pack_indices = pack_documents(
                self._lengths,
                max_length,
                max_segments_per_example,
                slice_strategy=slice_strategy,
            )

    def is_finite(self) -> bool:
        return True

    async def async_len(self) -> int:
        return len(self._pack_indices)

    async def get_batch(self, indices: Sequence[int]) -> Sequence[tuple[PyTree[np.ndarray], PyTree[np.ndarray]]]:
        """
        For each requested packed example (by index into self._pack_indices), reconstruct the
        token data on the fly from the underlying dataset.

        Supports both contiguous packs (range objects from greedy packing) and non-contiguous packs
        (list[int] from sorted or BFD packing). For contiguous packs, reads a single slice. For
        non-contiguous packs, reads each document individually and concatenates.

        Returns a list of tuples (data, segment_ids), where each is a PyTree (with the same structure as self.dataset),
        and each leaf is a numpy array representing the data or segment IDs for that packed example.
        """

        pack_doc_indices = [self._pack_indices[i] for i in indices]

        async def get_data_for_leaf(store, offsets, allowed: int) -> tuple[list[np.ndarray], list[np.ndarray]]:
            out_data = []
            out_segment_ids = []
            # Using ts.Batch to group reads.
            with ts.Batch():
                for doc_ids in pack_doc_indices:
                    is_contiguous = isinstance(doc_ids, range)

                    if is_contiguous and len(doc_ids) > 0:
                        # Fast path: contiguous range — single slice read
                        token_start = offsets[doc_ids.start] if doc_ids.start > 0 else 0
                        token_end = offsets[doc_ids.stop]
                        token_count = token_end - token_start
                        if token_count > allowed:
                            if self.slice_strategy == "raise":
                                raise ValueError(
                                    f"Token count {token_count} exceeds allowed maximum {allowed} for documents "
                                    f"{list(doc_ids)}. Consider using slice_strategy='left', 'right', or 'drop', or "
                                    "increasing max_length."
                                )
                            assert (
                                len(doc_ids) == 1
                            ), "We shouldn't have packed two examples together if one is too long."
                            if self.slice_strategy == "right":
                                token_start = token_end - allowed
                            else:  # left
                                token_end = token_start + allowed
                        out_data.append(store.data[token_start:token_end].read())
                    else:
                        # Non-contiguous path: read each doc individually, will concatenate after await
                        doc_reads = []
                        total_tokens = 0
                        for doc_id in doc_ids:
                            doc_start = offsets[doc_id] if doc_id > 0 else 0
                            doc_end = offsets[doc_id + 1]
                            doc_len = doc_end - doc_start
                            if doc_len > allowed and len(doc_ids) == 1:
                                # Single oversized doc — apply slice strategy
                                if self.slice_strategy == "raise":
                                    raise ValueError(
                                        f"Document {doc_id} has length {doc_len} which exceeds maximum {allowed}."
                                    )
                                if self.slice_strategy == "right":
                                    doc_start = doc_end - allowed
                                else:  # left
                                    doc_end = doc_start + allowed
                                doc_len = doc_end - doc_start
                            read_len = min(doc_len, allowed - total_tokens)
                            if read_len > 0:
                                doc_reads.append(store.data[doc_start : doc_start + read_len].read())
                                total_tokens += read_len
                        # Store the list of futures; we'll concatenate after resolving
                        out_data.append(doc_reads)

                    # Build segment IDs for this pack
                    segment_ids = []
                    for doc_id in doc_ids:
                        doc_start = offsets[doc_id] if doc_id > 0 else 0
                        doc_end = offsets[doc_id + 1]
                        doc_length = doc_end - doc_start
                        if doc_length > allowed and self.slice_strategy != "raise":
                            segment_ids.extend([doc_id] * min(doc_length, allowed))
                        else:
                            segment_ids.extend([doc_id] * doc_length)
                    # Truncate segment_ids to allowed length
                    segment_ids = segment_ids[:allowed]
                    out_segment_ids.append(np.array(segment_ids))

            # Await all reads concurrently.
            resolved_data = []
            for item in out_data:
                if isinstance(item, list):
                    # Non-contiguous: list of futures -> concatenate
                    chunks = await asyncio.gather(*item)
                    resolved_data.append(np.concatenate(chunks) if len(chunks) > 0 else np.array([], dtype=np.uint32))
                else:
                    # Contiguous: single future
                    resolved_data.append(await item)
            out_data = resolved_data

            if self.pad_with_zeros:
                out_data = [np.pad(x, (0, allowed - x.shape[0])) for x in out_data]
                out_segment_ids = [np.pad(x, (0, allowed - x.shape[0]), constant_values=-1) for x in out_segment_ids]

            return out_data, out_segment_ids

        # For each leaf, we want to map our get_data_for_leaf over:
        # - the dataset leaf (a JaggedArrayStore)
        # - the allowed maximum from self.max_length (an int)
        # - and the corresponding doc_range for each requested pack.
        #
        # We extract the list of doc_range PyTrees for each requested pack:
        # Use tree.map to combine the leaves from: dataset, max_length and, for each pack, its doc_range.
        # Note: jax.tree.map will map over each pack in parallel across the leaves.
        max_length_tree = tree_broadcast_to(self.max_length, self._offsets)
        leaf_batch_futures = jax.tree.map(get_data_for_leaf, self.dataset, self._offsets, max_length_tree)

        # Flatten the resulting PyTree: each leaf is now an Awaitable returning a tuple of lists of np.ndarray—one per requested pack.
        leaves, treedef = jax.tree.flatten(leaf_batch_futures)
        # Await all leaf futures in one go.
        resolved_leaves = await asyncio.gather(*leaves)
        # resolved_leaves is a list (one per leaf) of tuples of lists of np.ndarray;
        # each inner list has length equal to len(indices) (the number of requested packs).
        # Reassemble the original tree structure.
        # We then want to return a list of packed examples. We do so by, for each pack index i, collecting the i'th
        # element of each leaf.
        results = []
        for i in range(len(indices)):
            data = jax.tree.unflatten(treedef, [leaf[0][i] for leaf in resolved_leaves])
            segment_ids = jax.tree.unflatten(treedef, [leaf[1][i] for leaf in resolved_leaves])
            results.append((data, segment_ids))
        return results


if __name__ == "__main__":
    # demo the GreedyPrepackedDataset
    import time

    import numpy as np

    path = "gs://marin-us-central2/tokenized/tulu_sft_v3_llama3_tokenizer-f88fdb/input_ids/"

    store = JaggedArrayStore.open(path, mode="r", dtype=np.uint32, cache_metadata=True)

    time_in = time.time()
    packed = GreedyPrepackedDataset(store, max_length=4096, pad_with_zeros=True, slice_strategy="right")
    time_out = time.time()
    print(f"Took {time_out - time_in:.2f}s to build pack")

    packed_sync = packed.as_sync_dataset()

    padding_count = 0
    total_tokens = 0

    for i in range(10):
        example_batch = packed_sync.get_batch(range(i * 100, (i + 1) * 100))

        for example in example_batch:
            padding_count += np.sum(example[0] == 0)
            total_tokens += example[0].size

    print(f"Padding rate: {padding_count / total_tokens:.3f}")
