# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import haliax as hax
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from haliax.partitioning import ResourceAxis
from levanter.data.dataset import ListAsyncDataset
from levanter.eval import TaggedEvaluator
from levanter.models.lm_model import LmExample
from levanter.testing.helpers import use_test_mesh

from experiments.domain_phase_mix import evaluate_tpp10_finemath_math as evaluation
from experiments.domain_phase_mix import starcoder_tpp10 as experiment


@pytest.mark.parametrize("length,prompt", [(4, 2), (8, 3), (9, 2), (16, 12), (17, 3), (31, 20)])
def test_solution_tokens_scored_once_across_overlapping_windows(length, prompt):
    ids = list(range(100, 100 + length))
    mask = [int(i >= prompt) for i in range(length)]
    windows = evaluation.completion_windows(ids, mask, context=8, stride=4, pad_id=0)
    scored = []
    for window in windows:
        predicted = np.roll(window.tokens, -1)
        scored.extend(predicted[window.loss_weight.astype(bool)].tolist())
        assert window.loss_weight[-1] == 0
    assert scored == ids[prompt:]


def test_reference_solution_mask_excludes_question_and_padding():
    tokenizer = experiment.verified_tokenizer()
    problem, solution = "What is 2 + 3?", "The result is 5."
    encoded = tokenizer.apply_chat_template_with_masks(
        [[{"role": "user", "content": problem}, {"role": "assistant", "content": solution}]],
        chat_template=evaluation.TEMPLATE,
    )
    ids, mask = encoded["input_ids"][0], encoded["assistant_masks"][0]
    selected = [token for token, weight in zip(ids, mask, strict=True) if weight]
    assert tokenizer.decode(selected) == solution
    assert tokenizer.convert_ids_to_tokens(selected[0]).startswith("▁")
    assert mask[0] == 0
    windows = evaluation.completion_windows(ids, mask, context=64, stride=32, pad_id=int(tokenizer.eos_token_id))
    example = evaluation.make_example(windows[0])
    scored = np.roll(np.asarray(example.tokens.array), -1)[np.asarray(example.loss_weight.array).astype(bool)]
    assert scored.tolist() == selected


def test_math_and_segmented_control_batch_together_with_partial_final_batch():
    position = hax.Axis("position", 8)
    math_examples = [
        evaluation.make_example(
            evaluation.ScoringWindow(
                np.arange(100 + i, 108 + i, dtype=np.int32),
                np.array([0, 0, 1, 1, 1, 0, 0, 0], dtype=np.float32),
            )
        )
        for i in range(5)
    ]
    control = LmExample.causal(
        hax.named(np.arange(110, 118, dtype=np.int32), position),
        segment_ids=hax.named(np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int32), position),
    )
    # One math problem has no internal attention boundary.
    mask = math_examples[0].attn_mask.materialize(position, hax.Axis("key_position", position.size))
    np.testing.assert_array_equal(mask.array, np.tril(np.ones((8, 8), dtype=bool)))

    def loss_fn(_model, batch):
        predicted = hax.roll(batch.tokens, -1, position).array
        return predicted.astype(jnp.float32), batch.loss_weight.array, predicted

    batch_axis = hax.Axis("batch", 4 * len(jax.devices()))
    with use_test_mesh(tensor_parallelism=1) as mesh:
        evaluator = TaggedEvaluator(
            EvalBatch=batch_axis,
            tagged_eval_sets=[
                (ListAsyncDataset(math_examples), ["math"]),
                (ListAsyncDataset([control, control]), ["control"]),
            ],
            loss_fn=loss_fn,
            device_mesh=mesh,
            axis_mapping={batch_axis.name: ResourceAxis.DATA},
            shuffle=False,
        )
        result = evaluator.evaluate(None)
    assert result.tag_micro_losses["math"] == pytest.approx(106.0)
    assert result.tag_micro_losses["control"] == pytest.approx(114.0)
