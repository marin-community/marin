# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The AMD June training loop, which dispatches step N+1 before it checks and logs step N, on four CPU devices."""

import collections
import dataclasses

import jax
import jax.numpy as jnp
import jmp
import numpy as np
import pytest
from levanter.callbacks.state_adapter import StateCallbackRunner
from levanter.checkpoint import Checkpointer, CheckpointInterval
from levanter.grug.sharding import compact_grug_mesh
from levanter.optim.config import AdamConfig
from levanter.testing.devices import use_test_devices
from levanter.utils.logging import LoadingTimeTrackerIterator

from experiments.amd.june_67b_a2b import train
from experiments.june_tpu_67b_a2b.moe.model import GrugModelConfig

CONFIG = GrugModelConfig(
    vocab_size=32,
    hidden_dim=128,
    intermediate_dim=16,
    shared_expert_intermediate_dim=16,
    num_experts=4,
    num_experts_per_token=2,
    num_layers=2,
    num_heads=2,
    num_kv_heads=1,
    head_dim=64,
    max_seq_len=8,
    sliding_window=4,
    capacity_factor=8.0,
    pooled_transport_capacity_factor=4.0,
    attention_implementation="reference",
    use_array_stacked_blocks=True,
    disable_pko=True,
)
NUM_STEPS = 4
POLICY = jmp.get_policy("f32")
Batch = collections.namedtuple("Batch", "tokens loss_weight attn_mask")


@dataclasses.dataclass
class LoopRun:
    state: object
    hooks: list[tuple[int, float, bool]]
    model_hooks: list[tuple[int, float]]
    checkpoints: list[tuple[int, bool, float]] | None
    dispatched: int
    error: str | None


class RecordingCheckpointer:
    """What a checkpointer reads: the step it is given and the state, which must still be live."""

    def __init__(self):
        self.saved = []

    def on_step(self, *, tree, step, force=False):
        self.saved.append((step, force, float(jnp.sum(tree.params.token_embed))))

    def save_final_checkpoint(self, *, tree, step):
        self.on_step(tree=tree, step=step, force=True)

    def wait_until_finished(self):
        pass


def _batches(nan_at_step=None):
    tokens = np.asarray(jax.random.randint(jax.random.key(1), (4, CONFIG.max_seq_len), 0, CONFIG.vocab_size))
    for step in range(NUM_STEPS):
        weight = np.ones(tokens.shape, np.float32)
        if step == nan_at_step:
            weight[0, 0] = np.nan
        yield Batch(np.roll(tokens, step, axis=0), weight, None)


@pytest.fixture(scope="module")
def trainer():
    """One compiled train step and the initial state, on four CPU devices."""
    optimizer = AdamConfig(learning_rate=1e-2, warmup=0.0).build(NUM_STEPS)
    with use_test_devices(4), jax.set_mesh(compact_grug_mesh(expert_axis_size=2)):
        step = train._make_train_step(optimizer, POLICY, z_loss_weight=0.0, ema_beta=None)
        init = jax.jit(lambda key: train.initial_state(CONFIG, optimizer=optimizer, mp=POLICY, key=key, ema_beta=None))
        yield step, lambda: init(jax.random.key(0))


def _run_loop(trainer, *, nan_at_step=None, checkpoint=True, checkpointer=None):
    train_step, initial = trainer
    dispatched = []

    def counting_step(state, batch, *, compute_watch):
        dispatched.append(state.step)
        return train_step(state, batch, compute_watch=compute_watch)

    hooks, model_hooks = [], []
    runner = StateCallbackRunner(
        step_getter=lambda s: s.step,
        model_getter=lambda s: s.params,
        eval_model_getter=lambda s: s.params,
        opt_state_getter=lambda s: s.opt_state,
    )
    runner.add_hook(lambda info, force=False: hooks.append((info.step, info.loss, force)), every=1)
    # A hook that reads the model, as eval does: the loop finishes its steps before dispatching the next.
    runner.add_hook(
        lambda info, force=False: model_hooks.append((info.step, float(jnp.sum(info.model.token_embed)))), every=2
    )
    if checkpointer is None and checkpoint:
        checkpointer = RecordingCheckpointer()
    state, error = None, None
    try:
        state = train._train_loop(
            counting_step,
            initial(),
            LoadingTimeTrackerIterator(_batches(nan_at_step)),
            num_train_steps=NUM_STEPS,
            state_callbacks=runner,
            checkpointer=checkpointer,
            watch_interval=0,
            step_timeout=None,
            diagnose_numerics=False,
            hooks_read_state=lambda step: (step + 1) % 2 == 0,
        )
    except FloatingPointError as e:
        error = str(e)
    return LoopRun(
        state=state,
        hooks=hooks,
        model_hooks=model_hooks,
        checkpoints=getattr(checkpointer, "saved", None),
        dispatched=len(dispatched),
        error=error,
    )


def test_pipelined_loop_trains_and_hooks_like_stepping_one_at_a_time(trainer):
    train_step, initial = trainer
    state, losses, embed_sums = initial(), [], []
    for batch in _batches():
        state, metrics, _ = train_step(state, batch, compute_watch=False)
        losses.append(float(metrics["train/loss"]))
        embed_sums.append(float(jnp.sum(state.params.token_embed)))

    run = _run_loop(trainer)

    for got, want in zip(jax.tree.leaves(run.state), jax.tree.leaves(state), strict=True):
        np.testing.assert_array_equal(got, want)
    # Every step hooked once in order with its own loss, then the forced call after the last.
    assert run.hooks == [(s, losses[s], False) for s in range(NUM_STEPS)] + [(NUM_STEPS - 1, losses[-1], True)]
    # The hook that reads the model saw the real state of the steps it is due on.
    assert run.model_hooks == [(1, embed_sums[1]), (3, embed_sums[3]), (3, embed_sums[3])]
    assert run.checkpoints == [(s + 1, False, embed_sums[s]) for s in range(NUM_STEPS)] + [
        (NUM_STEPS, True, embed_sums[-1])
    ]


def test_nonfinite_loss_raises_one_step_late_and_is_never_checkpointed(trainer):
    unchecked = _run_loop(trainer, nan_at_step=2, checkpoint=False)
    checked = _run_loop(trainer, nan_at_step=2)

    for run in (unchecked, checked):
        assert run.error.startswith("Non-finite Grug loss at step 3"), run.error
        # Steps 0 and 1 were hooked; step 2 was not.
        assert [s for s, *_ in run.hooks] == [0, 1]
    # Without a checkpointer, step 3 was dispatched before step 2's loss was fetched.
    assert unchecked.dispatched == 4
    # With one, step 2 is checked before its state can be saved: no state after the non-finite step is saved.
    assert checked.checkpoints is not None and [s for s, *_ in checked.checkpoints] == [1, 2]
    assert checked.dispatched == 3


@pytest.mark.parametrize(("every", "saved_steps"), [(2, [2, 4]), (3, [3, 4])])
def test_final_save_writes_the_last_step_once(trainer, tmp_path, every, saved_steps):
    """A last step the step policy saved is not saved again after the loop; one it did not save still is."""
    checkpointer = Checkpointer(tmp_path, save_interval=None, step_policies=[CheckpointInterval(every=every)])
    saves = []
    save = checkpointer.save_checkpoint

    def recording_save(tree, step, destination, commit_callback=None, **kwargs):
        saves.append(step)
        save(tree, step, destination, commit_callback, **kwargs)

    checkpointer.save_checkpoint = recording_save
    run = _run_loop(trainer, checkpointer=checkpointer)

    assert run.error is None
    assert saves == saved_steps
    assert sorted(p.name for p in tmp_path.iterdir()) == [f"step-{s}" for s in saved_steps]
