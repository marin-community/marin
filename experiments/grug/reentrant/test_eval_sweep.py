# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""CPU smoke test for the re-entrant depth-scaling eval plumbing.

No checkpoint is involved: a tiny randomly-initialized re-entrant model is
evaluated at several recurrence depths through the real ``evaluate_at_depths``
helper, asserting one finite macro loss per depth and that different depths
generally produce different losses (i.e. the loop count really does change the
forward). This proves the multi-R eval sweep works without a real E3 checkpoint.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from haliax.partitioning import set_mesh
from jax.sharding import PartitionSpec as P
from levanter.data.dataset import ListAsyncDataset
from levanter.data.text.examples import GrugLmExample
from levanter.eval import TaggedEvaluator
from levanter.grug.sharding import compact_grug_mesh

from experiments.grug.reentrant.eval_sweep import evaluate_at_depths
from experiments.grug.reentrant.model import GrugModelConfig, Transformer

_BATCH_AXES = ("replica_dcn", "data", "expert")


def _tiny_reentrant_config() -> GrugModelConfig:
    """A minimal re-entrant MoE config: 1 prelude + 1 core + 1 coda, loopable core."""
    return GrugModelConfig(
        vocab_size=64,
        hidden_dim=32,
        intermediate_dim=32,
        shared_expert_intermediate_dim=32,
        num_experts=4,
        num_experts_per_token=2,
        num_layers=3,
        num_prelude_layers=1,
        num_coda_layers=1,
        recurrence_steps=2,
        num_heads=2,
        num_kv_heads=2,
        max_seq_len=16,
        sliding_window=16,
    )


def _make_grug_dataset(cfg: GrugModelConfig, *, num_examples: int, seed: int) -> ListAsyncDataset:
    """Deterministic synthetic GrugLmExamples with host (numpy) token arrays.

    Built outside any explicit grug mesh so the example arrays carry no committed
    sharding; the DataLoader then places them onto the grug mesh at stack time.
    """
    rng = np.random.default_rng(seed)
    examples = [
        GrugLmExample.causal(np.asarray(rng.integers(0, cfg.vocab_size, size=cfg.max_seq_len), dtype=np.int32))
        for _ in range(num_examples)
    ]
    return ListAsyncDataset(examples)


def _tiny_evaluator(cfg: GrugModelConfig, mesh, datasets: list[ListAsyncDataset]) -> TaggedEvaluator:
    """A 2-tag grug evaluator over pre-built synthetic datasets."""
    eval_batch_size = jax.device_count()
    eval_array_sharding = jax.sharding.NamedSharding(mesh, P(_BATCH_AXES, None))

    def eval_loss_fn(model: Transformer, batch: GrugLmExample):
        per_pos_loss = model.next_token_loss(
            batch.tokens,
            batch.loss_weight,
            mask=batch.attn_mask,
            reduction="none",
            logsumexp_weight=None,
        )
        per_pos_loss = jax.sharding.reshard(per_pos_loss, eval_array_sharding)
        per_pos_weight = jax.sharding.reshard(batch.loss_weight, eval_array_sharding)
        per_pos_token_id = jnp.roll(batch.tokens, -1, axis=-1)
        return per_pos_loss, per_pos_weight, per_pos_token_id

    return TaggedEvaluator(
        EvalBatch=eval_batch_size,
        tagged_eval_sets=[(datasets[0], ["a"]), (datasets[1], ["b"])],
        loss_fn=eval_loss_fn,
        tokenizer=None,
        device_mesh=mesh,
        axis_mapping={"batch": _BATCH_AXES},
    )


def test_evaluate_at_depths_returns_finite_loss_per_depth():
    cfg = _tiny_reentrant_config()
    recurrence_values = (1, 2, 4, 8)
    num_examples = 2 * jax.device_count()

    # Build datasets outside the grug mesh so their example arrays carry the
    # default (Auto) mesh aval that the DataLoader uses when it stacks them.
    datasets = [
        _make_grug_dataset(cfg, num_examples=num_examples, seed=0),
        _make_grug_dataset(cfg, num_examples=num_examples, seed=1),
    ]

    mesh = compact_grug_mesh()
    with set_mesh(mesh):
        model = Transformer.init(cfg, key=jax.random.PRNGKey(0))
        evaluator = _tiny_evaluator(cfg, mesh, datasets)
        results = evaluate_at_depths(model, evaluator, recurrence_values)

    assert set(results) == set(recurrence_values)
    macro_losses = {}
    for recurrence_steps, result in results.items():
        assert np.isfinite(result.macro_avg_loss), f"non-finite macro loss at R={recurrence_steps}"
        assert np.isfinite(result.micro_avg_loss), f"non-finite micro loss at R={recurrence_steps}"
        macro_losses[recurrence_steps] = result.macro_avg_loss

    # Looping the core a different number of times must change the forward, so the
    # depths should not all collapse onto a single loss value.
    distinct = {round(loss, 5) for loss in macro_losses.values()}
    assert len(distinct) > 1, f"all depths gave the same loss {macro_losses}; loop count had no effect"


def test_evaluate_at_depths_does_not_mutate_input_model():
    cfg = _tiny_reentrant_config()
    num_examples = 2 * jax.device_count()
    datasets = [
        _make_grug_dataset(cfg, num_examples=num_examples, seed=0),
        _make_grug_dataset(cfg, num_examples=num_examples, seed=1),
    ]
    mesh = compact_grug_mesh()
    with set_mesh(mesh):
        model = Transformer.init(cfg, key=jax.random.PRNGKey(1))
        evaluator = _tiny_evaluator(cfg, mesh, datasets)
        evaluate_at_depths(model, evaluator, (2, 4))

    # The depth swap is per-variant (dataclasses.replace), so the passed-in model
    # keeps its original config recurrence_steps.
    assert model.config.recurrence_steps == cfg.recurrence_steps


if __name__ == "__main__":
    pytest.main([__file__, "-x", "-q"])
