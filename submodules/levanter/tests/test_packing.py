import tempfile
from typing import Generator

import jax.numpy as jnp
import numpy as np
import pytest

import haliax as hax

from levanter.data.packing import (
    GreedyPrepackedDataset,
    PromptCompletion,
    SequencePacker,
    greedy_pack_prompt_completions,
    pack_prompt_completions,
    per_segment_correct,
    per_segment_loss,
)
from levanter.layers.attention import AttentionMask
from levanter.models.lm_model import LmExample
from levanter.store.jagged_array import JaggedArrayStore


def test_per_segment_loss():
    Pos = hax.Axis("pos", size=10)
    packer = SequencePacker(Pos=Pos, max_pack_size=10, pad_token=0)

    # Add two sequences
    packer.add_example(ids=[1, 2, 3], loss_mask=[1, 1, 1], segment_id=None)
    packer.add_example(ids=[4, 5], loss_mask=[1, 1], segment_id=None)

    # Pack into LmExample
    packed = packer.pack()

    losses = hax.named(jnp.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0]), Pos)

    Segments = hax.Axis("segments", size=3)

    unique_ids, segment_losses = per_segment_loss(packed, losses, max_Segments=Segments)

    assert list(unique_ids.array) == [-1, 0, 1]
    assert list(segment_losses.array) == [0.0, 0.6, 0.9]


def test_can_pack_simple_case():
    Pos = hax.Axis("pos", size=10)
    packer = SequencePacker(Pos=Pos, max_pack_size=2, pad_token=0)

    assert packer.can_pack([1, 2, 3]) is True
    packer.add_example(ids=[1, 2, 3], loss_mask=[1, 1, 1])
    assert packer.can_pack([4, 5]) is True
    assert packer.can_pack(list(range(6, 16))) is False  # Exceeds Pos size


def test_add_example_and_pack():
    Pos = hax.Axis("pos", size=10)
    packer = SequencePacker(Pos=Pos, max_pack_size=2, pad_token=0)

    packer.add_example([1, 2, 3], [1, 1, 1])
    packed = packer.pack()

    expected_tokens = [1, 2, 3, 0, 0, 0, 0, 0, 0, 0]
    expected_segment_ids = [0, 0, 0, -1, -1, -1, -1, -1, -1, -1]
    expected_loss_mask = [1, 1, 1, 0, 0, 0, 0, 0, 0, 0]

    np.testing.assert_array_equal(packed.tokens.array, expected_tokens)
    np.testing.assert_array_equal(packed.attn_mask.segment_ids.array, expected_segment_ids)
    np.testing.assert_array_equal(packed.loss_mask.array, expected_loss_mask)


def test_exceed_max_pack_size():
    Pos = hax.Axis("pos", size=10)
    packer = SequencePacker(Pos=Pos, max_pack_size=2, pad_token=0)

    packer.add_example([1, 2, 3], [1, 1, 1])
    packer.add_example([4, 5, 6], [1, 1, 1])

    with pytest.raises(ValueError, match="Too many segments"):
        packer.add_example([7, 8], [1, 1])  # Exceeds max pack size


def test_empty_sequence():
    Pos = hax.Axis("pos", size=10)
    packer = SequencePacker(Pos=Pos, max_pack_size=2, pad_token=0)

    with pytest.raises(ValueError, match="ids and loss_mask must have the same length"):
        packer.add_example([], [1])  # Mismatched lengths

    packer.add_example([], [])  # Adding an empty sequence is allowed but does nothing
    packed = packer.pack()

    expected_tokens = [0] * 10
    expected_segment_ids = [-1] * 10
    expected_loss_mask = [0] * 10

    np.testing.assert_array_equal(packed.tokens.array, expected_tokens)
    np.testing.assert_array_equal(packed.attn_mask.segment_ids.array, expected_segment_ids)
    np.testing.assert_array_equal(packed.loss_mask.array, expected_loss_mask)


def test_packing_multiple_examples():
    Pos = hax.Axis("pos", size=10)
    packer = SequencePacker(Pos=Pos, max_pack_size=2, pad_token=0)

    # First example
    packer.add_example([1, 2], [1, 1])
    # Second example
    packer.add_example([3, 4, 5], [1, 1, 1])

    packed = packer.pack()

    expected_tokens = [1, 2, 3, 4, 5, 0, 0, 0, 0, 0]
    expected_segment_ids = [0, 0, 1, 1, 1, -1, -1, -1, -1, -1]
    expected_loss_mask = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]

    np.testing.assert_array_equal(packed.tokens.array, expected_tokens)
    np.testing.assert_array_equal(packed.attn_mask.segment_ids.array, expected_segment_ids)
    np.testing.assert_array_equal(packed.loss_mask.array, expected_loss_mask)


def test_pack_prompt_completions_simple():
    Pos = hax.Axis("pos", size=10)
    pad_token = 0
    max_pack_size = 2
    max_buffered_examples = 2

    sequences = [
        PromptCompletion(ids=[1, 2, 3], prompt_length=2),
        PromptCompletion(ids=[4, 5], prompt_length=1),
        PromptCompletion(ids=[6, 7, 8], prompt_length=1),
    ]

    results = list(pack_prompt_completions(Pos, sequences, pad_token, max_pack_size, max_buffered_examples))

    assert len(results) == 2  # Expect two packed LmExamples

    # Check the first packed example
    packed_1 = results[0]
    expected_tokens_1 = [1, 2, 3, 4, 5, 0, 0, 0, 0, 0]
    expected_segment_ids_1 = [0, 0, 0, 1, 1, -1, -1, -1, -1, -1]
    expected_loss_mask_1 = [0, 1, 0, 1, 0, 0, 0, 0, 0, 0]

    np.testing.assert_array_equal(packed_1.tokens.array, expected_tokens_1)
    np.testing.assert_array_equal(packed_1.attn_mask.segment_ids.array, expected_segment_ids_1)
    np.testing.assert_array_equal(packed_1.loss_mask.array, expected_loss_mask_1)

    # Check the second packed example
    packed_2 = results[1]
    expected_tokens_2 = [6, 7, 8, 0, 0, 0, 0, 0, 0, 0]
    expected_segment_ids_2 = [0, 0, 0, -1, -1, -1, -1, -1, -1, -1]
    expected_loss_mask_2 = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0]

    np.testing.assert_array_equal(packed_2.tokens.array, expected_tokens_2)
    np.testing.assert_array_equal(packed_2.attn_mask.segment_ids.array, expected_segment_ids_2)
    np.testing.assert_array_equal(packed_2.loss_mask.array, expected_loss_mask_2)


def test_pack_prompt_completions_exceed_max_buffered_examples():
    Pos = hax.Axis("pos", size=10)
    pad_token = 0
    max_pack_size = 1
    max_buffered_examples = 1

    sequences = [
        PromptCompletion(ids=[1, 2, 3], prompt_length=2),
        PromptCompletion(ids=[4, 5], prompt_length=1),
        PromptCompletion(ids=[6, 7, 8], prompt_length=1),
    ]

    results = list(pack_prompt_completions(Pos, sequences, pad_token, max_pack_size, max_buffered_examples))

    assert len(results) == 3

    # Check the first packed example
    packed_1 = results[0]
    expected_tokens_1 = [1, 2, 3, 0, 0, 0, 0, 0, 0, 0]
    expected_segment_ids_1 = [0, 0, 0, -1, -1, -1, -1, -1, -1, -1]
    expected_loss_mask_1 = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]

    np.testing.assert_array_equal(packed_1.tokens.array, expected_tokens_1)
    np.testing.assert_array_equal(packed_1.attn_mask.segment_ids.array, expected_segment_ids_1)
    np.testing.assert_array_equal(packed_1.loss_mask.array, expected_loss_mask_1)

    # Check the second packed example
    packed_2 = results[1]
    expected_tokens_2 = [4, 5, 0, 0, 0, 0, 0, 0, 0, 0]
    expected_segment_ids_2 = [0, 0, -1, -1, -1, -1, -1, -1, -1, -1]
    expected_loss_mask_2 = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    np.testing.assert_array_equal(packed_2.tokens.array, expected_tokens_2)
    np.testing.assert_array_equal(packed_2.attn_mask.segment_ids.array, expected_segment_ids_2)
    np.testing.assert_array_equal(packed_2.loss_mask.array, expected_loss_mask_2)

    # Check the third packed example
    packed_3 = results[2]
    expected_tokens_3 = [6, 7, 8, 0, 0, 0, 0, 0, 0, 0]
    expected_segment_ids_3 = [0, 0, 0, -1, -1, -1, -1, -1, -1, -1]
    expected_loss_mask_3 = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0]

    np.testing.assert_array_equal(packed_3.tokens.array, expected_tokens_3)
    np.testing.assert_array_equal(packed_3.attn_mask.segment_ids.array, expected_segment_ids_3)
    np.testing.assert_array_equal(packed_3.loss_mask.array, expected_loss_mask_3)


def test_segment_correct():
    # Mock segment_ids and loss_mask
    Pos = hax.Axis("pos", size=10)
    tokens = hax.named(jnp.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]), Pos)
    segment_ids = hax.named(jnp.array([0, 0, 1, 1, 1, 2, 2, 2, 2, 2]), Pos)
    loss_mask = hax.named(jnp.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0]), Pos)

    # Create a packed example
    attn_mask = AttentionMask.causal().with_segment_ids(segment_ids=segment_ids)
    packed_example = LmExample(tokens=tokens, loss_mask=loss_mask, attn_mask=attn_mask)

    # Mock correctness array (True for correct, False for incorrect)
    correct = hax.named(jnp.array([True, True, True, False, True, False, True, True, True, True]), Pos)

    max_Segments = hax.Axis("segments", size=4)

    # Call the function
    unique_ids, segment_correct = per_segment_correct(packed_example, correct, max_Segments)

    assert list(unique_ids.array) == [0, 1, 2, -1]
    assert list(segment_correct.array) == [True, False, True, True]


####################################
# tests for GreedyPrepackedDataset #
####################################


# TEST 1: Single-leaf dataset.
@pytest.fixture
def simple_dataset() -> Generator[tuple[dict[str, JaggedArrayStore], dict[str, int], np.ndarray], None, None]:
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a single JaggedArrayStore with four documents.
        # Let document lengths be: 100, 200, 150, 150 tokens.
        offsets = np.array([0, 100, 300, 450, 600])
        store = JaggedArrayStore.open(tmpdir, item_rank=1, dtype=jnp.int64)

        # Fill the store with data
        for i in range(len(offsets) - 1):
            start = offsets[i]
            end = offsets[i + 1]
            data = np.arange(start, end)
            store.append(data)

        # Build dataset as a PyTree; here a dict with one key.
        dataset = {"store": store}
        # Set allowed max tokens (per leaf) as 300.
        max_length = {"store": 300}
        yield dataset, max_length, offsets


@pytest.mark.asyncio
async def test_simple_pack(simple_dataset):
    dataset, max_length, offsets = simple_dataset
    tester = GreedyPrepackedDataset(dataset, max_length)
    packs = tester._pack_indices
    # We expect, given document lengths [100,200,150,150] and budget 300,
    # that the first pack covers docs 0 and 1: token range = [offsets[0], offsets[2]) = [0,300),
    # and the second pack covers docs 2 and 3: [offsets[2], offsets[4]) = [300,600).
    assert len(packs) == 2
    pack0 = packs[0]  # Expect {"store": range(0,2)}
    pack1 = packs[1]  # Expect {"store": range(2,4)}
    assert pack0 == range(0, 2)
    assert pack1 == range(2, 4)

    # now check that we can get the data out
    batch = tester.as_sync_dataset()[0]
    data, segment_ids = batch
    expected_data = np.arange(0, 300)
    expected_segment_ids = np.concatenate([np.full(100, 0), np.full(200, 1)])  # global doc indices 0 and 1
    assert np.array_equal(data["store"], expected_data)
    assert np.array_equal(segment_ids["store"], expected_segment_ids)

    batch = tester.as_sync_dataset()[1]
    data, segment_ids = batch
    expected_data = np.arange(300, 600)
    expected_segment_ids = np.concatenate([np.full(150, 2), np.full(150, 3)])
    assert np.array_equal(data["store"], expected_data)
    assert np.array_equal(segment_ids["store"], expected_segment_ids)


def test_simple_pack_max_examples(simple_dataset):
    dataset, max_length, offsets = simple_dataset
    tester = GreedyPrepackedDataset(dataset, max_length, max_segments_per_example=1, pad_with_zeros=False)
    packs = tester._pack_indices
    # We expect, given document lengths [100,200,150,150] and budget 300,
    # that each pack covers exactly one document
    assert len(packs) == 4
    assert list(packs[0]) == [0]
    assert list(packs[1]) == [1]
    assert list(packs[2]) == [2]
    assert list(packs[3]) == [3]

    # now check that we can get the data out
    batch = tester.as_sync_dataset()[0]
    data, segment_ids = batch
    expected_data = np.arange(0, 100)  # first document
    expected_segment_ids = np.full(100, 0)  # global doc index 0
    assert np.array_equal(data["store"], expected_data)
    assert np.array_equal(segment_ids["store"], expected_segment_ids)

    batch = tester.as_sync_dataset()[1]
    data, segment_ids = batch
    expected_data = np.arange(100, 300)  # second document
    expected_segment_ids = np.full(200, 1)  # global doc index 1
    assert np.array_equal(data["store"], expected_data)
    assert np.array_equal(segment_ids["store"], expected_segment_ids)

    batch = tester.as_sync_dataset()[2]
    data, segment_ids = batch
    expected_data = np.arange(300, 450)  # third document
    expected_segment_ids = np.full(150, 2)  # global doc index 2
    assert np.array_equal(data["store"], expected_data)
    assert np.array_equal(segment_ids["store"], expected_segment_ids)

    batch = tester.as_sync_dataset()[3]
    data, segment_ids = batch
    expected_data = np.arange(450, 600)  # fourth document
    expected_segment_ids = np.full(150, 3)  # global doc index 3
    assert np.array_equal(data["store"], expected_data)
    assert np.array_equal(segment_ids["store"], expected_segment_ids)


def test_simple_pack_max_examples_padded(simple_dataset):
    dataset, max_length, offsets = simple_dataset
    tester = GreedyPrepackedDataset(dataset, max_length, max_segments_per_example=1, pad_with_zeros=True)
    packs = tester._pack_indices
    # We expect, given document lengths [100,200,150,150] and budget 300,
    # that each pack covers exactly one document
    assert len(packs) == 4
    assert list(packs[0]) == [0]
    assert list(packs[1]) == [1]
    assert list(packs[2]) == [2]
    assert list(packs[3]) == [3]

    # now check that we can get the data out, with padding
    batch = tester.as_sync_dataset()[0]
    data, segment_ids = batch
    expected_data = np.pad(np.arange(0, 100), (0, max_length["store"] - 100))  # first document padded to max_length
    expected_segment_ids = np.pad(np.full(100, 0), (0, max_length["store"] - 100), constant_values=-1)
    assert np.array_equal(data["store"], expected_data)
    assert np.array_equal(segment_ids["store"], expected_segment_ids)

    batch = tester.as_sync_dataset()[1]
    data, segment_ids = batch
    expected_data = np.pad(np.arange(100, 300), (0, max_length["store"] - 200))  # second document padded to max_length
    expected_segment_ids = np.pad(np.full(200, 1), (0, max_length["store"] - 200), constant_values=-1)
    assert np.array_equal(data["store"], expected_data)
    assert np.array_equal(segment_ids["store"], expected_segment_ids)

    batch = tester.as_sync_dataset()[2]
    data, segment_ids = batch
    expected_data = np.pad(np.arange(300, 450), (0, max_length["store"] - 150))  # third document padded to max_length
    expected_segment_ids = np.pad(np.full(150, 2), (0, max_length["store"] - 150), constant_values=-1)
    assert np.array_equal(data["store"], expected_data)
    assert np.array_equal(segment_ids["store"], expected_segment_ids)

    batch = tester.as_sync_dataset()[3]
    data, segment_ids = batch
    expected_data = np.pad(np.arange(450, 600), (0, max_length["store"] - 150))  # fourth document padded to max_length
    expected_segment_ids = np.pad(np.full(150, 3), (0, max_length["store"] - 150), constant_values=-1)
    assert np.array_equal(data["store"], expected_data)
    assert np.array_equal(segment_ids["store"], expected_segment_ids)


# TEST 2: Multi-leaf dataset.
@pytest.fixture
def multi_leaf_dataset():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create two leaves.
        # Leaf1: document lengths: [100,200,150,150] => offsets: [0,100,300,450,600]
        # Leaf2: document lengths: [90,190,150,150]  => offsets: [0,90,280,430,580]
        offsets1 = np.array([0, 100, 300, 450, 600])
        offsets2 = np.array([0, 90, 280, 430, 580])

        store1 = JaggedArrayStore.open(tmpdir + "/store1", item_rank=1, dtype=jnp.int64)
        store2 = JaggedArrayStore.open(tmpdir + "/store2", item_rank=1, dtype=jnp.int64)

        # Fill the stores with data
        for i in range(len(offsets1) - 1):
            start = offsets1[i]
            end = offsets1[i + 1]
            data = np.arange(start, end)
            store1.append(data)

        for i in range(len(offsets2) - 1):
            start = offsets2[i]
            end = offsets2[i + 1]
            data = np.arange(start, end)
            store2.append(data)

        dataset = {"store1": store1, "store2": store2}
        # Allowed max per leaf: for store1: 300 tokens, for store2: 250 tokens.
        max_length = {"store1": 300, "store2": 250}
        yield dataset, max_length, (offsets1, offsets2)


def test_multi_leaf_pack(multi_leaf_dataset):
    dataset, max_length, _ = multi_leaf_dataset
    tester = GreedyPrepackedDataset(dataset, max_length, pad_with_zeros=False)
    packs = tester._pack_indices
    # Here the effective allowed max is computed per leaf:
    # For store1: budget = 300, for store2: budget = 250. Thus the pack must satisfy both.
    # Document lengths for store1: [100,200,150,150]; for store2: [90,190,150,150].
    # For pack 1: starting at doc0, adding doc0 yields (100 and 90) which are within allowed.
    # Trying to add doc1: store1 becomes 100+200=300 (okay) but store2 becomes 90+190=280 (>250);
    # so pack 1 covers doc0 only.
    # Similarly, pack 2 covers doc1 only; pack 3 covers doc2 only; pack 4 covers doc3 only.
    assert len(packs) == 4
    expected_docs = [[0], [1], [2], [3]]
    for pack, expected in zip(packs, expected_docs):
        assert list(pack) == expected

    # now check that we can get the data out
    batch = tester.as_sync_dataset()[0]
    data, segment_ids = batch
    expected_data1 = np.arange(0, 100)  # first document in store1
    expected_data2 = np.arange(0, 90)  # first document in store2
    expected_segment_ids1 = np.full(100, 0)  # global doc index 0
    expected_segment_ids2 = np.full(90, 0)  # global doc index 0
    assert np.array_equal(data["store1"], expected_data1)
    assert np.array_equal(data["store2"], expected_data2)
    assert np.array_equal(segment_ids["store1"], expected_segment_ids1)
    assert np.array_equal(segment_ids["store2"], expected_segment_ids2)

    batch = tester.as_sync_dataset()[1]
    data, segment_ids = batch
    expected_data1 = np.arange(100, 300)  # second document in store1
    expected_data2 = np.arange(90, 280)  # second document in store2
    expected_segment_ids1 = np.full(200, 1)  # global doc index 1
    expected_segment_ids2 = np.full(190, 1)  # global doc index 1
    assert np.array_equal(data["store1"], expected_data1)
    assert np.array_equal(data["store2"], expected_data2)
    assert np.array_equal(segment_ids["store1"], expected_segment_ids1)
    assert np.array_equal(segment_ids["store2"], expected_segment_ids2)

    batch = tester.as_sync_dataset()[2]
    data, segment_ids = batch
    expected_data1 = np.arange(300, 450)  # third document in store1
    expected_data2 = np.arange(280, 430)  # third document in store2
    expected_segment_ids1 = np.full(150, 2)  # global doc index 2
    expected_segment_ids2 = np.full(150, 2)  # global doc index 2
    assert np.array_equal(data["store1"], expected_data1)
    assert np.array_equal(data["store2"], expected_data2)
    assert np.array_equal(segment_ids["store1"], expected_segment_ids1)
    assert np.array_equal(segment_ids["store2"], expected_segment_ids2)

    batch = tester.as_sync_dataset()[3]
    data, segment_ids = batch
    expected_data1 = np.arange(450, 600)  # fourth document in store1
    expected_data2 = np.arange(430, 580)  # fourth document in store2
    expected_segment_ids1 = np.full(150, 3)  # global doc index 3
    expected_segment_ids2 = np.full(150, 3)  # global doc index 3
    assert np.array_equal(data["store1"], expected_data1)
    assert np.array_equal(data["store2"], expected_data2)
    assert np.array_equal(segment_ids["store1"], expected_segment_ids1)
    assert np.array_equal(segment_ids["store2"], expected_segment_ids2)


def test_multi_leaf_pack_padded(multi_leaf_dataset):
    dataset, max_length, _ = multi_leaf_dataset
    tester = GreedyPrepackedDataset(dataset, max_length, pad_with_zeros=True)
    packs = tester._pack_indices
    # Here the effective allowed max is computed per leaf:
    # For store1: budget = 300, for store2: budget = 250. Thus the pack must satisfy both.
    # Document lengths for store1: [100,200,150,150]; for store2: [90,190,150,150].
    # For pack 1: starting at doc0, adding doc0 yields (100 and 90) which are within allowed.
    # Trying to add doc1: store1 becomes 100+200=300 (okay) but store2 becomes 90+190=280 (>250);
    # so pack 1 covers doc0 only.
    # Similarly, pack 2 covers doc1 only; pack 3 covers doc2 only; pack 4 covers doc3 only.
    assert len(packs) == 4
    expected_docs = [[0], [1], [2], [3]]
    for pack, expected in zip(packs, expected_docs):
        assert list(pack) == expected

    # now check that we can get the data out, with padding
    batch = tester.as_sync_dataset()[0]
    data, segment_ids = batch
    expected_data1 = np.pad(
        np.arange(0, 100), (0, max_length["store1"] - 100)
    )  # first document in store1 padded to max_length
    expected_data2 = np.pad(
        np.arange(0, 90), (0, max_length["store2"] - 90)
    )  # first document in store2 padded to max_length
    expected_segment_ids1 = np.pad(np.full(100, 0), (0, max_length["store1"] - 100), constant_values=-1)
    expected_segment_ids2 = np.pad(np.full(90, 0), (0, max_length["store2"] - 90), constant_values=-1)
    assert np.array_equal(data["store1"], expected_data1)
    assert np.array_equal(data["store2"], expected_data2)
    assert np.array_equal(segment_ids["store1"], expected_segment_ids1)
    assert np.array_equal(segment_ids["store2"], expected_segment_ids2)

    batch = tester.as_sync_dataset()[1]
    data, segment_ids = batch
    expected_data1 = np.pad(
        np.arange(100, 300), (0, max_length["store1"] - 200)
    )  # second document in store1 padded to max_length
    expected_data2 = np.pad(
        np.arange(90, 280), (0, max_length["store2"] - 190)
    )  # second document in store2 padded to max_length
    expected_segment_ids1 = np.pad(np.full(200, 1), (0, max_length["store1"] - 200), constant_values=-1)
    expected_segment_ids2 = np.pad(np.full(190, 1), (0, max_length["store2"] - 190), constant_values=-1)
    assert np.array_equal(data["store1"], expected_data1)
    assert np.array_equal(data["store2"], expected_data2)
    assert np.array_equal(segment_ids["store1"], expected_segment_ids1)
    assert np.array_equal(segment_ids["store2"], expected_segment_ids2)

    batch = tester.as_sync_dataset()[2]
    data, segment_ids = batch
    expected_data1 = np.pad(
        np.arange(300, 450), (0, max_length["store1"] - 150)
    )  # third document in store1 padded to max_length
    expected_data2 = np.pad(
        np.arange(280, 430), (0, max_length["store2"] - 150)
    )  # third document in store2 padded to max_length
    expected_segment_ids1 = np.pad(np.full(150, 2), (0, max_length["store1"] - 150), constant_values=-1)
    expected_segment_ids2 = np.pad(np.full(150, 2), (0, max_length["store2"] - 150), constant_values=-1)
    assert np.array_equal(data["store1"], expected_data1)
    assert np.array_equal(data["store2"], expected_data2)
    assert np.array_equal(segment_ids["store1"], expected_segment_ids1)
    assert np.array_equal(segment_ids["store2"], expected_segment_ids2)

    batch = tester.as_sync_dataset()[3]
    data, segment_ids = batch
    expected_data1 = np.pad(
        np.arange(450, 600), (0, max_length["store1"] - 150)
    )  # fourth document in store1 padded to max_length
    expected_data2 = np.pad(
        np.arange(430, 580), (0, max_length["store2"] - 150)
    )  # fourth document in store2 padded to max_length
    expected_segment_ids1 = np.pad(np.full(150, 3), (0, max_length["store1"] - 150), constant_values=-1)
    expected_segment_ids2 = np.pad(np.full(150, 3), (0, max_length["store2"] - 150), constant_values=-1)
    assert np.array_equal(data["store1"], expected_data1)
    assert np.array_equal(data["store2"], expected_data2)
    assert np.array_equal(segment_ids["store1"], expected_segment_ids1)
    assert np.array_equal(segment_ids["store2"], expected_segment_ids2)


# TEST 3: max_segments_per_example constraint.
@pytest.fixture
def dataset_with_segments():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a single-leaf dataset of 4 documents (as in test 1) with offsets [0,100,300,450,600].
        offsets = np.array([0, 100, 300, 450, 600])
        store = JaggedArrayStore.open(tmpdir, item_rank=1, dtype=jnp.int64)

        # Fill the store with data
        for i in range(len(offsets) - 1):
            start = offsets[i]
            end = offsets[i + 1]
            data = np.arange(start, end)
            store.append(data)

        dataset = {"store": store}
        max_length = {"store": 1000}  # large enough so that token count doesn't force a break.
        yield dataset, max_length, offsets


def test_max_segments_constraint(dataset_with_segments):
    dataset, max_length, offsets = dataset_with_segments
    # With max_segments_per_example set to 1, each pack must cover exactly one document.
    tester = GreedyPrepackedDataset(dataset, max_length, max_segments_per_example=1, pad_with_zeros=False)
    packs = tester._pack_indices
    # There are 4 documents so expect 4 packs.
    assert len(packs) == 4
    # For each document, the returned range should be exactly one document index
    for i, pack in enumerate(packs):
        assert list(pack) == [i]

    # now check that we can get the data out
    for i in range(4):
        batch = tester.as_sync_dataset()[i]
        data, segment_ids = batch
        start = offsets[i]
        end = offsets[i + 1]
        expected_data = np.arange(start, end)
        expected_segment_ids = np.full(end - start, i)
        assert np.array_equal(data["store"], expected_data)
        assert np.array_equal(segment_ids["store"], expected_segment_ids)


def test_max_segments_constraint_padded(dataset_with_segments):
    dataset, max_length, offsets = dataset_with_segments
    # With max_segments_per_example set to 1, each pack must cover exactly one document.
    tester = GreedyPrepackedDataset(dataset, max_length, max_segments_per_example=1, pad_with_zeros=True)
    packs = tester._pack_indices
    # There are 4 documents so expect 4 packs.
    assert len(packs) == 4
    # For each document, the returned range should be exactly one document index
    for i, pack in enumerate(packs):
        assert list(pack) == [i]

    # now check that we can get the data out, with padding
    for i in range(4):
        batch = tester.as_sync_dataset()[i]
        data, segment_ids = batch
        start = offsets[i]
        end = offsets[i + 1]
        length = end - start
        expected_data = np.pad(np.arange(start, end), (0, max_length["store"] - length))
        expected_segment_ids = np.pad(np.full(length, i), (0, max_length["store"] - length), constant_values=-1)
        assert np.array_equal(data["store"], expected_data)
        assert np.array_equal(segment_ids["store"], expected_segment_ids)


def test_too_long_to_pack(multi_leaf_dataset):
    dataset, _, _ = multi_leaf_dataset
    max_length = {"store1": 10, "store2": 5000}

    with pytest.raises(
        ValueError, match="Document 0 in leaf 'store1' has length 100 which exceeds maximum allowed length 10"
    ):
        GreedyPrepackedDataset(dataset, max_length)

    tester = GreedyPrepackedDataset(dataset, max_length, slice_strategy="right", pad_with_zeros=False)
    pack2 = tester._pack_indices[2]
    assert list(pack2) == [2]

    # now check that we can get the data out
    batch = tester.as_sync_dataset()[2]
    data, segment_ids = batch
    expected_data1 = np.arange(450 - 10, 450)  # third document in store1, sliced to last 10 tokens
    expected_data2 = np.arange(280, 430)  # third document in store2
    expected_segment_ids1 = np.full(10, 2)  # global doc index 2 for the sliced portion
    expected_segment_ids2 = np.full(150, 2)  # global doc index 2 for the full document
    assert np.array_equal(data["store1"], expected_data1)
    assert np.array_equal(data["store2"], expected_data2)
    assert np.array_equal(segment_ids["store1"], expected_segment_ids1)
    assert np.array_equal(segment_ids["store2"], expected_segment_ids2)


def test_too_long_to_pack_padded(multi_leaf_dataset):
    dataset, _, _ = multi_leaf_dataset
    max_length = {"store1": 10, "store2": 5000}

    with pytest.raises(
        ValueError, match="Document 0 in leaf 'store1' has length 100 which exceeds maximum allowed length 10"
    ):
        GreedyPrepackedDataset(dataset, max_length)

    tester = GreedyPrepackedDataset(dataset, max_length, slice_strategy="right", pad_with_zeros=True)
    pack2 = tester._pack_indices[2]
    assert list(pack2) == [2]

    # now check that we can get the data out, with padding
    batch = tester.as_sync_dataset()[2]
    data, segment_ids = batch
    expected_data1 = np.arange(450 - 10, 450)  # third document in store1, sliced to last 10 tokens
    expected_data2 = np.pad(
        np.arange(280, 430), (0, max_length["store2"] - (430 - 280))
    )  # third document in store2 padded
    expected_segment_ids1 = np.full(10, 2)  # global doc index 2 for the sliced portion
    expected_segment_ids2 = np.pad(
        np.full(150, 2), (0, max_length["store2"] - 150), constant_values=-1
    )  # segment IDs for the full document with padding
    assert np.array_equal(data["store1"], expected_data1)
    assert np.array_equal(data["store2"], expected_data2)
    assert np.array_equal(segment_ids["store1"], expected_segment_ids1)
    assert np.array_equal(segment_ids["store2"], expected_segment_ids2)


def test_slicing_strategies():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a dataset with a single document that's too long
        store = JaggedArrayStore.open(tmpdir, item_rank=1, dtype=jnp.int64)
        doc_length = 100
        store.append(np.arange(doc_length))
        dataset = {"store": store}
        max_length = {"store": 50}  # Document is twice as long as allowed

        # Test right slicing (default behavior)
        tester_right = GreedyPrepackedDataset(dataset, max_length, slice_strategy="right", pad_with_zeros=False)
        batch = tester_right.as_sync_dataset()[0]
        data, segment_ids = batch
        # Should take last 50 tokens
        expected_data = np.arange(doc_length - max_length["store"], doc_length)
        expected_segment_ids = np.full(max_length["store"], 0)
        assert np.array_equal(data["store"], expected_data)
        assert np.array_equal(segment_ids["store"], expected_segment_ids)

        # Test left slicing
        tester_left = GreedyPrepackedDataset(dataset, max_length, slice_strategy="left", pad_with_zeros=False)
        batch = tester_left.as_sync_dataset()[0]
        data, segment_ids = batch
        # Should take first 50 tokens
        expected_data = np.arange(0, max_length["store"])
        expected_segment_ids = np.full(max_length["store"], 0)
        assert np.array_equal(data["store"], expected_data)
        assert np.array_equal(segment_ids["store"], expected_segment_ids)

        # Test raise strategy
        with pytest.raises(
            ValueError, match="Document 0 in leaf 'store' has length 100 which exceeds maximum allowed length 50"
        ):
            GreedyPrepackedDataset(dataset, max_length, slice_strategy="raise")

        # Test invalid strategy
        with pytest.raises(ValueError, match="slice_strategy must be one of 'left', 'right', or 'raise'"):
            GreedyPrepackedDataset(dataset, max_length, slice_strategy="invalid")


def test_invalid_max_segments():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = JaggedArrayStore.open(tmpdir, item_rank=1, dtype=jnp.int64)
        store.append(np.arange(100))

        # Test non-positive max_segments
        with pytest.raises(ValueError, match="max_segments_per_example must be a positive integer"):
            GreedyPrepackedDataset({"store": store}, max_length={"store": 300}, max_segments_per_example=0)

        # Test non-integer max_segments
        with pytest.raises(ValueError, match="max_segments_per_example must be a positive integer"):
            GreedyPrepackedDataset({"store": store}, max_length={"store": 300}, max_segments_per_example="1")


def test_mismatched_document_counts():
    with tempfile.TemporaryDirectory() as tmpdir:
        store1 = JaggedArrayStore.open(tmpdir + "/store1", item_rank=1, dtype=jnp.int64)
        store2 = JaggedArrayStore.open(tmpdir + "/store2", item_rank=1, dtype=jnp.int64)

        # Add different numbers of documents to each store
        store1.append(np.arange(100))
        store1.append(np.arange(100))
        store2.append(np.arange(100))

        with pytest.raises(ValueError, match="All leaves must have the same number of documents"):
            GreedyPrepackedDataset({"store1": store1, "store2": store2}, max_length={"store1": 300, "store2": 300})


def test_too_long_document_error():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = JaggedArrayStore.open(tmpdir, item_rank=1, dtype=jnp.int64)
        store.append(np.arange(1000))  # Document longer than max_length

        with pytest.raises(
            ValueError, match="Document 0 in leaf 'store' has length 1000 which exceeds maximum allowed length 300"
        ):
            GreedyPrepackedDataset({"store": store}, max_length={"store": 300}, slice_strategy="raise")

        # Should not raise when slice_strategy is not "raise"
        tester = GreedyPrepackedDataset({"store": store}, max_length={"store": 300}, slice_strategy="right")
        assert len(tester._pack_indices) == 1


def test_greedy_pack_prompt_completions_simple():
    Pos = hax.Axis("pos", size=10)
    pad_token = 0
    max_segments_per_example = 2

    sequences = [
        PromptCompletion(ids=[1, 2, 3], prompt_length=2),
        PromptCompletion(ids=[4, 5], prompt_length=1),
        PromptCompletion(ids=[6, 7, 8], prompt_length=1),
    ]

    results = list(greedy_pack_prompt_completions(Pos, sequences, pad_token, max_segments_per_example))

    assert len(results) == 2  # Expect two packed LmExamples

    # Check the first packed example
    packed_1 = results[0]
    expected_tokens_1 = [1, 2, 3, 4, 5, 0, 0, 0, 0, 0]
    expected_segment_ids_1 = [0, 0, 0, 1, 1, -1, -1, -1, -1, -1]
    expected_loss_mask_1 = [0, 1, 0, 1, 0, 0, 0, 0, 0, 0]

    np.testing.assert_array_equal(packed_1.tokens.array, expected_tokens_1)
    np.testing.assert_array_equal(packed_1.attn_mask.segment_ids.array, expected_segment_ids_1)
    np.testing.assert_array_equal(packed_1.loss_mask.array, expected_loss_mask_1)

    # Check the second packed example
    packed_2 = results[1]
    expected_tokens_2 = [6, 7, 8, 0, 0, 0, 0, 0, 0, 0]
    expected_segment_ids_2 = [2, 2, 2, -1, -1, -1, -1, -1, -1, -1]
    expected_loss_mask_2 = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0]

    np.testing.assert_array_equal(packed_2.tokens.array, expected_tokens_2)
    np.testing.assert_array_equal(packed_2.attn_mask.segment_ids.array, expected_segment_ids_2)
    np.testing.assert_array_equal(packed_2.loss_mask.array, expected_loss_mask_2)


def test_greedy_pack_prompt_completions_max_segments():
    Pos = hax.Axis("pos", size=10)
    pad_token = 0
    max_segments_per_example = 1

    sequences = [
        PromptCompletion(ids=[1, 2, 3], prompt_length=2),
        PromptCompletion(ids=[4, 5], prompt_length=1),
        PromptCompletion(ids=[6, 7, 8], prompt_length=1),
    ]

    results = list(greedy_pack_prompt_completions(Pos, sequences, pad_token, max_segments_per_example))

    assert len(results) == 3  # Each sequence should be in its own pack

    # Check the first packed example
    packed_1 = results[0]
    expected_tokens_1 = [1, 2, 3, 0, 0, 0, 0, 0, 0, 0]
    expected_segment_ids_1 = [0, 0, 0, -1, -1, -1, -1, -1, -1, -1]
    expected_loss_mask_1 = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]

    np.testing.assert_array_equal(packed_1.tokens.array, expected_tokens_1)
    np.testing.assert_array_equal(packed_1.attn_mask.segment_ids.array, expected_segment_ids_1)
    np.testing.assert_array_equal(packed_1.loss_mask.array, expected_loss_mask_1)

    # Check the second packed example
    packed_2 = results[1]
    expected_tokens_2 = [4, 5, 0, 0, 0, 0, 0, 0, 0, 0]
    expected_segment_ids_2 = [1, 1, -1, -1, -1, -1, -1, -1, -1, -1]
    expected_loss_mask_2 = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    np.testing.assert_array_equal(packed_2.tokens.array, expected_tokens_2)
    np.testing.assert_array_equal(packed_2.attn_mask.segment_ids.array, expected_segment_ids_2)
    np.testing.assert_array_equal(packed_2.loss_mask.array, expected_loss_mask_2)

    # Check the third packed example
    packed_3 = results[2]
    expected_tokens_3 = [6, 7, 8, 0, 0, 0, 0, 0, 0, 0]
    expected_segment_ids_3 = [2, 2, 2, -1, -1, -1, -1, -1, -1, -1]
    expected_loss_mask_3 = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0]

    np.testing.assert_array_equal(packed_3.tokens.array, expected_tokens_3)
    np.testing.assert_array_equal(packed_3.attn_mask.segment_ids.array, expected_segment_ids_3)
    np.testing.assert_array_equal(packed_3.loss_mask.array, expected_loss_mask_3)


def test_greedy_pack_prompt_completions_too_long():
    Pos = hax.Axis("pos", size=5)  # Small position size to force slicing
    pad_token = 0
    max_segments_per_example = 2

    sequences = [
        PromptCompletion(ids=[1, 2, 3, 4, 5, 6], prompt_length=2),  # Too long, will be sliced
        PromptCompletion(ids=[7, 8], prompt_length=1),
    ]

    results = greedy_pack_prompt_completions(Pos, sequences, pad_token, max_segments_per_example)

    assert len(results) == 2  # Each sequence should be in its own pack due to length

    # Check the first packed example (sliced)
    packed_1 = results[0]
    expected_tokens_1 = [2, 3, 4, 5, 6]  # Sliced to last 5 tokens
    expected_segment_ids_1 = [0, 0, 0, 0, 0]
    expected_loss_mask_1 = [1, 1, 1, 1, 0]  # Only the last token is not in loss mask

    np.testing.assert_array_equal(packed_1.tokens.array, expected_tokens_1)
    np.testing.assert_array_equal(packed_1.attn_mask.segment_ids.array, expected_segment_ids_1)
    np.testing.assert_array_equal(packed_1.loss_mask.array, expected_loss_mask_1)

    # Check the second packed example
    packed_2 = results[1]
    expected_tokens_2 = [7, 8, 0, 0, 0]
    expected_segment_ids_2 = [1, 1, -1, -1, -1]
    expected_loss_mask_2 = [1, 0, 0, 0, 0]

    np.testing.assert_array_equal(packed_2.tokens.array, expected_tokens_2)
    np.testing.assert_array_equal(packed_2.attn_mask.segment_ids.array, expected_segment_ids_2)
    np.testing.assert_array_equal(packed_2.loss_mask.array, expected_loss_mask_2)


def test_greedy_pack_prompt_completions_empty():
    Pos = hax.Axis("pos", size=10)
    pad_token = 0
    max_segments_per_example = 2

    # Empty sequence list
    results = list(greedy_pack_prompt_completions(Pos, [], pad_token, max_segments_per_example))
    assert len(results) == 0

    # Single empty sequence (should raise error due to PromptCompletion validation)
    with pytest.raises(ValueError, match="PromptCompletion must have at least one token"):
        list(
            greedy_pack_prompt_completions(
                Pos, [PromptCompletion(ids=[], prompt_length=0)], pad_token, max_segments_per_example
            )
        )


if __name__ == "__main__":
    pytest.main()
