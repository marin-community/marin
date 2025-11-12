# Copyright 2025 The Levanter Authors
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
from typing import Iterable, Iterator, Literal, Optional, Sequence, TypeVar

import jax
import jax.numpy as jnp
import numpy as np
import tensorstore as ts
from jaxtyping import PyTree

import haliax as hax

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
        self._loss_mask: list[int] = []
        self.num_segments = 0
        self.pad_token = pad_token
        self.max_pack_size = max_pack_size
        assert pad_token is not None, "pad_token must be set"

    def can_pack(self, ids: list[int]) -> bool:
        return len(ids) + len(self._ids) <= self.Pos.size and self.num_segments < self.max_pack_size

    def add_example(self, ids: list[int], loss_mask: list[int] | np.ndarray, segment_id: int | None = None):
        if len(ids) != len(loss_mask):
            raise ValueError("ids and loss_mask must have the same length")

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

        self._loss_mask.extend(loss_mask)

    def pack(self) -> LmExample:
        ids = self._ids + [self.pad_token] * (self.Pos.size - len(self._ids))

        segment_ids = self._segment_ids + [-1] * (self.Pos.size - len(self._segment_ids))

        loss_mask = self._loss_mask + [0] * (self.Pos.size - len(self._loss_mask))

        with local_cpu_mesh():
            tokens = hax.named(ids, self.Pos).astype(jnp.int32)
            segment_ids = hax.named(segment_ids, self.Pos).astype(jnp.int32)
            loss_mask = hax.named(loss_mask, self.Pos).astype(jnp.int32)

            attn_mask = AttentionMask.causal().with_segment_ids(segment_ids)

            return LmExample(tokens=tokens, loss_mask=loss_mask, attn_mask=attn_mask)


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
        loss_mask = np.arange(len(sequence.ids)) >= sequence.prompt_length - 1
        loss_mask[-1] = 0
        assert np.any(loss_mask)

        for packer in packers:
            if packer.can_pack(sequence.ids):
                packer.add_example(sequence.ids, loss_mask, sequence.segment_id)

                if packer.num_segments == max_segments_per_example:
                    yield packer.pack()
                    packers.remove(packer)
                break
        else:
            # no packer could fit the example, create a new one
            packer = SequencePacker(Pos, max_segments_per_example, pad_token)
            packer.add_example(sequence.ids, loss_mask, sequence.segment_id)
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
    masked_losses = losses * packed_example.loss_mask

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
    masked_correct = hax.logical_or(correct, hax.logical_not(packed_example.loss_mask))

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

    def make_loss_mask(id, prompt_length):
        loss_mask = np.arange(len(id)) >= prompt_length - 1
        loss_mask[-1] = 0
        return loss_mask

    # Convert sequences to lists for easier access
    sequences = list(sequences)
    ids = [sequence.ids for sequence in sequences]

    # Pack documents based on their lengths
    packs = pack_documents(
        lengths=np.array([len(token_ids) for token_ids in ids]),
        max_length=Pos.size,
        max_segments_per_example=max_segments_per_example,
        slice_too_long_examples=True,
    )

    out = []

    # Yield packed examples for normal sequences
    for docs_in_pack in packs:
        # Get the documents in this pack
        pack_sequences = [sequences[i] for i in docs_in_pack]
        pack_prompt_lengths = [sequence.prompt_length for sequence in pack_sequences]

        # Concatenate the IDs and create loss masks
        concat_ids = []
        concat_loss_mask = []
        segment_ids = []

        for doc_id, seq, prompt_len in zip(docs_in_pack, pack_sequences, pack_prompt_lengths):
            concat_ids.extend(seq.ids)
            concat_loss_mask.extend(make_loss_mask(seq.ids, prompt_len))
            segment_ids.extend([doc_id] * len(seq.ids))

        # Pad to max length
        pad_length = Pos.size - len(concat_ids)

        if pad_length > 0:
            concat_ids.extend([pad_token] * pad_length)
            concat_loss_mask.extend([0] * pad_length)
            segment_ids.extend([-1] * pad_length)
        elif pad_length < 0:
            # too long, this should only happen if there's 1 document in the pack
            if len(pack_sequences) != 1:
                raise ValueError("Too many tokens in a pack with more than one document")
            concat_ids = concat_ids[-Pos.size :]
            concat_loss_mask = concat_loss_mask[-Pos.size :]
            segment_ids = segment_ids[-Pos.size :]

        # Create the LmExample
        tokens = hax.named(np.array(concat_ids), Pos)
        loss_mask = hax.named(np.array(concat_loss_mask), Pos)
        segment_ids = hax.named(np.array(segment_ids), Pos)
        attn_mask = AttentionMask.causal().with_segment_ids(segment_ids)

        out.append(LmExample(tokens=tokens, loss_mask=loss_mask, attn_mask=attn_mask))

    return out


def _segment_ids_from_lengths(doc_ids: list[int], lengths: list[int]) -> list[int]:
    segment_ids = []
    for doc_id, length in zip(doc_ids, lengths):
        segment_ids.extend([doc_id] * length)
    return segment_ids


def pack_documents(
    lengths: PyTree[np.ndarray],
    max_length: PyTree[int],
    max_segments_per_example: int | None = None,
    slice_too_long_examples: bool = False,
) -> list[range]:
    """
    Greedily pack documents into contiguous groups without storing full token ranges.

    Args:
        lengths: A PyTree of numpy arrays, each containing the lengths of documents for a leaf.
            Each array should be of length n_docs, where n_docs is the number of documents.
            The i-th document has length lengths[i].
        max_length: A PyTree of integers, each specifying the maximum number of tokens allowed per pack for that leaf
        max_segments_per_example: Optional maximum number of documents per pack
        slice_too_long_examples: If True, slice documents that exceed max_length instead of raising an error

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

    # Validate document lengths
    for lens, allowed, leaf_name in zip(lengths_leaves, max_length_leaves, leaf_names):
        for i in range(n_docs):
            if lens[i] > allowed and not slice_too_long_examples:
                raise ValueError(
                    f"Document {i} in leaf '{leaf_name}' has length {lens[i]} which exceeds "
                    f"maximum allowed length {allowed}. Consider setting slice_too_long_examples=True "
                    "or increasing max_length."
                )

    pack_doc_ranges = []
    i = 0
    while i < n_docs:
        start = i
        total_segments = 0
        # Accumulate documents while for each leaf the token span remains within the allowed max.
        while i < n_docs:
            # Check optional segment constraint: if adding one more document would exceed max_segments_per_example.
            if max_segments_per_example is not None and (total_segments + 1) > max_segments_per_example:
                break
            # For each leaf, check if adding document i would keep the token count within allowed capacity.
            valid = True
            for lens, allowed, leaf_name in zip(lengths_leaves, max_length_leaves, leaf_names, strict=True):
                # Compute token count from document start to document i+1.
                token_sum = sum(lens[start : i + 1])
                if token_sum > allowed:
                    valid = False
                    if not slice_too_long_examples and i == start:
                        # If this is the first document in a new pack and it's too long, raise an error
                        raise ValueError(
                            f"Document {i} in leaf '{leaf_name}' has length {lens[i]} which exceeds "
                            f"maximum allowed length {allowed}. Consider setting slice_too_long_examples=True "
                            "or increasing max_length."
                        )
                    break
            if not valid:
                break
            total_segments += 1
            i += 1

        # If no document could be added (i.e. a single document exceeds capacity)
        if i == start:
            if not slice_too_long_examples:
                raise ValueError(f"Document {start} exceeds allowed capacity.")
            else:
                i = start + 1

        pack_doc_ranges.append(range(start, i))
    return pack_doc_ranges


class GreedyPrepackedDataset(AsyncDataset[tuple[T, T]]):
    """
    Prepacks a dataset into a new dataset where examples are packed into a single example.

    As per usual, I can't help but make this generic.

    Args:
        dataset: A PyTree of JaggedArrayStore objects, each representing a leaf in the dataset.
        max_length: A PyTree of integers, each representing the maximum number of tokens allowed per leaf.
        max_segments_per_example: Maximum number of documents that can be packed into a single example.
        pad_with_zeros: If True, pad examples to max_length with zeros. If False, return examples as-is.
        slice_strategy: One of "left", "right", or "raise". Determines how to handle examples that exceed max_length:
            - "left": Slice from the beginning of the example
            - "right": Slice from the end of the example
            - "raise": Raise an error when an example exceeds max_length
    """

    def __init__(
        self,
        dataset: T,  # PyTree[JaggedArrayStore],
        max_length: int | T,  # PyTree[int],
        max_segments_per_example: int | None = None,
        pad_with_zeros: bool = True,
        slice_strategy: Literal["left", "right", "raise"] = "raise",
    ):
        """
        Args:
            dataset: A PyTree of JaggedArrayStore objects, each representing a leaf in the dataset.
            max_length: A PyTree of integers, each representing the maximum number of tokens allowed per leaf.
            max_segments_per_example: Maximum number of documents that can be packed into a single example.
            pad_with_zeros: If True, pad examples to max_length with zeros. If False, return examples as-is.
            slice_strategy: One of "left", "right", or "raise". Determines how to handle examples that exceed max_length.
        """
        super().__init__()

        if slice_strategy not in ["left", "right", "raise"]:
            raise ValueError(f"slice_strategy must be one of 'left', 'right', or 'raise', got {slice_strategy}")

        self.dataset = dataset
        self.max_length = max_length
        self.max_segments_per_example = max_segments_per_example
        self.pad_with_zeros = pad_with_zeros
        self.slice_strategy = slice_strategy

        _offsets = jax.tree.map(lambda store: store.offsets[0 : store.num_rows + 1].read(), self.dataset)
        self._offsets = jax.tree.map(lambda fut: fut.result(), _offsets)

        def diff_offsets(offsets: np.ndarray):
            # fine to mutate since we have a copy
            # the array store has the number of rows in the 0th offset
            offsets[0] = 0
            return offsets[1:] - offsets[:-1]

        # Convert offsets to lengths
        self._lengths = jax.tree.map(diff_offsets, self._offsets)

        # Build pack indices
        self._pack_indices: list[range] = pack_documents(
            self._lengths, max_length, max_segments_per_example, slice_strategy != "raise"
        )

    def is_finite(self) -> bool:
        return True

    async def async_len(self) -> int:
        return len(self._pack_indices)

    async def final_length_is_known(self) -> bool:
        return True

    async def current_len(self) -> Optional[int]:
        return len(self._pack_indices)

    async def get_batch(self, indices: Sequence[int]) -> Sequence[tuple[PyTree[np.ndarray], PyTree[np.ndarray]]]:
        """
        For each requested packed example (by index into self._pack_indices), reconstruct the
        token data on the fly from the underlying dataset. In our packing scheme the pack holds, for each leaf,
        a range of document IDs. Using the JaggedArrayStore's offsets and allowed maximum (from self.max_length),
        we compute the corresponding token slice (data range) and then read that slice using tensorstore's ts.Batch context.
        We then pad the data (if pad_with_zeros is set) up to allowed length.

        Returns a list of tuples (data, segment_ids), where each is a PyTree (with the same structure as self.dataset),
        and each leaf is a numpy array representing the data or segment IDs for that packed example.
        """

        pack_doc_ranges = [self._pack_indices[i] for i in indices]

        async def get_data_for_leaf(store, offsets, allowed: int) -> tuple[list[np.ndarray], list[np.ndarray]]:
            out_data = []
            out_segment_ids = []
            # Using ts.Batch to group reads.
            with ts.Batch():
                for dr in pack_doc_ranges:
                    # Compute token boundaries using the store's offsets.
                    token_start = offsets[dr.start] if dr.start > 0 else 0
                    token_end = offsets[dr.stop]
                    token_count = token_end - token_start
                    if token_count > allowed:
                        if self.slice_strategy != "raise":
                            assert len(dr) == 1, "We shouldn't have packed two examples together if one is too long."
                            if self.slice_strategy == "right":
                                # slice from the right
                                token_start = token_end - allowed
                            else:  # left
                                # slice from the left
                                token_end = token_start + allowed
                        else:
                            raise ValueError(
                                f"Token count {token_count} exceeds allowed maximum {allowed} for documents "
                                f"{list(dr)}. Consider using a different slice_strategy or increasing max_length."
                            )
                    # Read the slice from the underlying data.
                    out_data.append(store.data[token_start:token_end].read())

                    # Create segment IDs for this pack
                    segment_ids = []
                    for doc_idx in range(len(dr)):
                        doc_start = offsets[dr.start + doc_idx] if dr.start + doc_idx > 0 else 0
                        doc_end = offsets[dr.start + doc_idx + 1]
                        doc_length = doc_end - doc_start
                        # Use the global document index as the segment ID
                        global_doc_idx = dr.start + doc_idx
                        # If this is a sliced document, adjust the segment IDs to match the sliced portion
                        if doc_length > allowed and self.slice_strategy != "raise":
                            segment_ids.extend([global_doc_idx] * allowed)
                        else:
                            segment_ids.extend([global_doc_idx] * doc_length)
                    out_segment_ids.append(np.array(segment_ids))

            # Await all reads concurrently.
            out_data = await asyncio.gather(*out_data)

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
