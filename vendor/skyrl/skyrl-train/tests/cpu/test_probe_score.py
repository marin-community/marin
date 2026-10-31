# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for the probe score pass's compact-space logit scoring.

The pass computes the reference top-K, the candidate gathers, and the exact
full-vocabulary rows from left-unpadded (compact) hidden states, projecting one
sequence chunk at a time, and scatters them back to the batch's padded layout.
These tests hold it against a brute-force full log-softmax reference on random
inputs, including ragged left-padded batches; most pass the logits themselves
as the hidden states under an identity projection.
"""

from types import SimpleNamespace

import pytest
import torch

from skyrl_train.mismatch_probe.score_math import CompactHiddenStates, score_compacted_hidden

BATCH = 3
PADDED_LEN = 12
COMPACT_LEN = PADDED_LEN  # TP=1: no sequence-alignment rounding
VOCAB = 50
TOP_K = 7
LENGTHS = (PADDED_LEN, 7, 3)


def batch(seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """(compact logits, left-padded attention mask) for a ragged batch."""
    generator = torch.Generator().manual_seed(seed)
    mask = torch.zeros(BATCH, PADDED_LEN, dtype=torch.bool)
    for row, length in enumerate(LENGTHS):
        mask[row, PADDED_LEN - length :] = True
    logits = torch.randn(BATCH, COMPACT_LEN, VOCAB + 8, generator=generator)  # padded vocabulary columns
    return logits, mask


def identity(logits: torch.Tensor) -> CompactHiddenStates:
    return CompactHiddenStates(hidden=logits, project=lambda hidden: hidden)


def reference_log_softmax(logits: torch.Tensor) -> torch.Tensor:
    return torch.log_softmax(logits[..., :VOCAB].float(), dim=-1)


def brute_force_tail(full: torch.Tensor, support_ids: torch.Tensor) -> torch.Tensor:
    """The log mass outside ``support_ids``, summed term by term in float64."""
    keep = torch.ones_like(full, dtype=torch.bool).scatter(-1, support_ids, False)
    return torch.log((full.double().exp() * keep).sum(-1))


def test_topk_equals_torch_topk_of_the_full_log_softmax():
    logits, mask = batch(seed=0)
    scores = score_compacted_hidden(
        identity(logits),
        mask,
        top_k=TOP_K,
        gather_ids=None,
        probe_score_positions=None,
        vocabulary_size=VOCAB,
        chunk_size=5,
    )
    reference_values, reference_ids = torch.topk(reference_log_softmax(logits), TOP_K, dim=-1)
    for row, length in enumerate(LENGTHS):
        torch.testing.assert_close(scores["topk_ids"][row, mask[row]], reference_ids[row, :length])
        torch.testing.assert_close(scores["topk_logprobs"][row, mask[row]], reference_values[row, :length])
        assert torch.count_nonzero(scores["topk_ids"][row, ~mask[row]]) == 0
        tail = brute_force_tail(reference_log_softmax(logits)[row, :length], reference_ids[row, :length])
        torch.testing.assert_close(scores["topk_tail_logmass"][row, mask[row]].double(), tail, rtol=1e-5, atol=1e-6)


def test_gather_reads_the_log_softmax_at_the_given_ids():
    logits, mask = batch(seed=1)
    generator = torch.Generator().manual_seed(2)
    # TOP_K distinct support ids per position, then one extra column (the realized
    # token) that may repeat a support id and stays out of the tail's complement.
    support = torch.rand(BATCH, PADDED_LEN, VOCAB, generator=generator).argsort(-1)[..., :TOP_K]
    realized = torch.randint(0, VOCAB, (BATCH, PADDED_LEN, 1), generator=generator)
    gather_ids = torch.cat((support, realized), dim=-1)
    scores = score_compacted_hidden(
        identity(logits),
        mask,
        top_k=None,
        gather_ids=gather_ids,
        probe_score_positions=None,
        vocabulary_size=VOCAB,
        chunk_size=None,
        support_width=TOP_K,
    )
    reference = reference_log_softmax(logits)
    for row, length in enumerate(LENGTHS):
        expected = torch.gather(reference[row, :length], -1, gather_ids[row, mask[row]].long())
        torch.testing.assert_close(scores["gathered_logprobs"][row, mask[row]], expected)
        tail = brute_force_tail(reference[row, :length], support[row, mask[row]])
        torch.testing.assert_close(scores["gathered_tail_logmass"][row, mask[row]].double(), tail, rtol=1e-5, atol=1e-6)


def test_exact_rows_are_the_full_log_softmax_of_the_selected_positions():
    logits, mask = batch(seed=3)
    # This micro-batch holds rows 40.. of the full batch: each selected position is
    # tagged with its batch-wide row plus one, and the coordinates report that row.
    first_row = 40
    positions = torch.zeros(BATCH, PADDED_LEN, dtype=torch.long)
    positions[0] = mask[0].long() * (first_row + 1)  # every real position of the longest row
    positions[2, PADDED_LEN - 1] = first_row + 3  # one position of the shortest row
    scores = score_compacted_hidden(
        identity(logits),
        mask,
        top_k=None,
        gather_ids=None,
        probe_score_positions=positions,
        vocabulary_size=VOCAB,
        chunk_size=4,
    )
    reference = reference_log_softmax(logits)
    # Values and coordinates are produced in the chunk sweep's order; the contract
    # is their pairing, not the order.
    assert sorted(scores["exact_positions"].tolist()) == sorted(
        [[first_row, position] for position in mask[0].nonzero(as_tuple=True)[0].tolist()]
        + [[first_row + 2, PADDED_LEN - 1]]
    )
    padded_to_compact = {
        row: {int(padded): compact for compact, padded in enumerate(mask[row].nonzero(as_tuple=True)[0].tolist())}
        for row in range(BATCH)
    }
    for value, (row, position) in zip(scores["exact_logprobs"], scores["exact_positions"].tolist(), strict=True):
        local = row - first_row
        torch.testing.assert_close(value, reference[local, padded_to_compact[local][position]])


def test_exactly_one_mode_must_be_set():
    logits, mask = batch(seed=4)
    with pytest.raises(ValueError, match="exactly one"):
        score_compacted_hidden(
            identity(logits),
            mask,
            top_k=None,
            gather_ids=None,
            probe_score_positions=None,
            vocabulary_size=VOCAB,
            chunk_size=None,
        )
    with pytest.raises(ValueError, match="exactly one"):
        score_compacted_hidden(
            identity(logits),
            mask,
            top_k=2,
            gather_ids=torch.zeros(BATCH, PADDED_LEN, 2, dtype=torch.long),
            probe_score_positions=None,
            vocabulary_size=VOCAB,
            chunk_size=None,
        )


def test_shorter_compact_logits_raise():
    logits, mask = batch(seed=5)
    with pytest.raises(ValueError, match="shorter than the longest sequence"):
        score_compacted_hidden(
            identity(logits[:, :-2]),
            mask,
            top_k=TOP_K,
            gather_ids=None,
            probe_score_positions=None,
            vocabulary_size=VOCAB,
            chunk_size=None,
        )


def test_mismatched_gather_shape_raises():
    logits, mask = batch(seed=6)
    with pytest.raises(ValueError, match="do not match the batch"):
        score_compacted_hidden(
            identity(logits),
            mask,
            top_k=None,
            gather_ids=torch.zeros(BATCH, PADDED_LEN + 1, TOP_K, dtype=torch.long),
            probe_score_positions=None,
            vocabulary_size=VOCAB,
            chunk_size=None,
            support_width=TOP_K,
        )


@pytest.mark.parametrize("mode", ["top_k", "gather", "exact"])
def test_chunked_projection_of_hidden_states_matches_full_logits(mode):
    """Projecting hidden states chunk by chunk scores like projecting them all at once.

    The projection only ever sees ``chunk_size`` positions, which is what keeps a
    131k-token sequence's [tokens x vocab] logits from being materialized.
    """
    generator = torch.Generator().manual_seed(7)
    _logits, mask = batch(seed=7)
    hidden = torch.randn(BATCH, COMPACT_LEN, 16, generator=generator)
    weight = torch.randn(VOCAB + 8, 16, generator=generator)
    widths: list[int] = []

    def project(chunk: torch.Tensor) -> torch.Tensor:
        widths.append(chunk.shape[1])
        return chunk @ weight.T

    gather_ids = torch.randint(0, VOCAB, (BATCH, PADDED_LEN, TOP_K + 1), generator=generator)
    gather_ids[..., :TOP_K] = torch.rand(BATCH, PADDED_LEN, VOCAB, generator=generator).argsort(-1)[..., :TOP_K]
    kwargs = {
        "top_k": TOP_K if mode == "top_k" else None,
        "gather_ids": gather_ids if mode == "gather" else None,
        "probe_score_positions": mask.long() * torch.arange(1, BATCH + 1)[:, None] if mode == "exact" else None,
        "support_width": TOP_K if mode == "gather" else None,
        "vocabulary_size": VOCAB,
    }
    chunked = score_compacted_hidden(CompactHiddenStates(hidden=hidden, project=project), mask, chunk_size=5, **kwargs)
    full = score_compacted_hidden(identity(hidden @ weight.T), mask, chunk_size=None, **kwargs)

    assert widths == [5, 5, 2]
    assert chunked.keys() == full.keys()
    if mode == "exact":  # exact rows come in chunk-sweep order; their pairing with coordinates is the contract
        for scores in (chunked, full):
            order = sorted(range(len(scores["exact_positions"])), key=lambda i: scores["exact_positions"][i].tolist())
            scores["exact_positions"] = scores["exact_positions"][order]
            scores["exact_logprobs"] = scores["exact_logprobs"][order]
    for key in full:
        torch.testing.assert_close(chunked[key], full[key])


@pytest.mark.parametrize("micro_batch_size", [1, 2])
def test_megatron_probe_forward_matches_full_log_softmax(megatron_wrapper, monkeypatch, micro_batch_size):
    wrapper = megatron_wrapper(packing=False, controller=None, num_experts=1)
    # The fixture installs stubs for the optional Megatron dependency before this import.
    from skyrl_train.workers.megatron.megatron_model_wrapper import MegatronForwardMicroBatch

    embeddings = torch.tensor([[0.0, 0.5], [1.0, -0.5], [-0.5, 1.0]])
    weights = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.5]])
    sequences = torch.tensor([[0, 1, 2, 1], [0, 0, 1, 2]])
    attention_mask = torch.tensor([[True, True, True, True], [False, False, True, True]])
    expected_logprobs, expected_ids = (embeddings[sequences] @ weights.T).log_softmax(-1).topk(2, dim=-1)

    def project(hidden, *, weight, runtime_gather_output):
        return hidden @ weights.T, None

    def model(input_ids, position_ids, model_attention_mask, *, packed_seq_params, output_processor):
        return output_processor(
            hidden_states=embeddings[input_ids].transpose(0, 1),
            output_layer=project,
            output_weight=None,
            scale_logits=lambda logits: logits,
            config=SimpleNamespace(mtp_num_layers=0),
        )

    def serial_forward(*, forward_step_func, data_iterator, model, num_microbatches, **kwargs):
        # Replace the distributed pipeline scheduler with a single-rank CPU executor.
        outputs = []
        for _ in range(num_microbatches):
            hidden, score = forward_step_func(data_iterator, model[0])
            _, scores = score(hidden)
            outputs.append(scores)
        return outputs

    monkeypatch.setattr(
        "skyrl_train.workers.megatron.megatron_model_wrapper.get_forward_backward_func", lambda: serial_forward
    )
    wrapper.actor_module = [model]
    micro_batches = [
        MegatronForwardMicroBatch(
            sequences=sequences[start : start + micro_batch_size],
            attention_mask=attention_mask[start : start + micro_batch_size],
            position_ids=attention_mask[start : start + micro_batch_size].long().cumsum(-1) - 1,
            num_actions=sequences.shape[1],
        )
        for start in range(0, len(sequences), micro_batch_size)
    ]
    scores = wrapper.probe_score_forward(micro_batches, seq_len=4, micro_batch_size=micro_batch_size, top_k=2)
    torch.testing.assert_close(scores["topk_ids"][attention_mask], expected_ids[attention_mask], rtol=0, atol=0)
    torch.testing.assert_close(scores["topk_logprobs"][attention_mask], expected_logprobs[attention_mask])
