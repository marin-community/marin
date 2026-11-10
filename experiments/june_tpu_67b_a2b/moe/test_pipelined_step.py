# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The June training loop with ``GrugTrainerConfig.pipelined_step_logging``, on eight CPU devices.

Pipelined, the loop dispatches step N+1 before fetching, checking and logging step N. Over a few steps of the tiny
June model it trains, hooks, logs, watches and checkpoints exactly as the synchronous loop: the same final state, the
same logged metrics per step, the same step info in every hook, the real state in the hooks of the steps that read
it and the same states handed to the checkpointer. A non-finite loss still raises, one step late when pipelined, and
no state after it reaches the checkpointer.
"""

import collections

import jax
import jax.numpy as jnp
import jmp
import levanter.tracker
import numpy as np
import pytest
from levanter.callbacks import log_step_info
from levanter.callbacks.state_adapter import StateCallbackRunner
from levanter.callbacks.watch import WatchConfig
from levanter.optim.config import AdamConfig
from levanter.testing.devices import use_cpu_devices
from levanter.tracker.tracker import Tracker
from levanter.utils.logging import LoadingTimeTrackerIterator

from experiments.june_tpu_67b_a2b.moe import train as june_train
from experiments.june_tpu_67b_a2b.moe.train_test_support import BASE, LOSS_WEIGHT, TOKENS, Batch, _meshes, leaves

NUM_STEPS = 5


def _mesh():
    return _meshes()["replica2_data2_expert2"]


POLICY = jmp.get_policy("p=f32,c=bfloat16")
WATCH = WatchConfig(interval=3)
STATE_HOOK_EVERY = 2
# Wall-clock metrics, which differ between any two runs.
TIMING_KEYS = {"throughput/duration", "throughput/hook_time", "throughput/loading_time"}


class RecordingTracker(Tracker):
    name = "recording"

    def __init__(self):
        self.steps = collections.defaultdict(dict)

    def log_hyperparameters(self, hparams):
        pass

    def log(self, metrics, *, step, commit=None):
        flat = {
            key: np.asarray(value).tolist()
            for key, value in jax.tree_util.tree_flatten_with_path(jax.device_get(dict(metrics)))[0]
        }
        self.steps[step].update({jax.tree_util.keystr(key): value for key, value in flat.items()})

    def log_summary(self, metrics):
        pass

    def log_artifact(self, artifact_path, *, name=None, type=None):  # noqa: A002
        pass

    def finish(self):
        pass


class RecordingCheckpointer:
    # What a checkpointer reads: the step it is given and the state, which must still be live.
    def __init__(self):
        self.saved = []

    def on_step(self, *, tree, step, force=False):
        self.saved.append((step, force, float(jnp.sum(tree.params.token_embed))))

    def wait_until_finished(self):
        pass


def param_sum(model):
    return float(jnp.sum(model.token_embed))


def batches(nan_at_step=None):
    for step in range(NUM_STEPS):
        weight = LOSS_WEIGHT.copy()
        if step == nan_at_step:
            weight[0, 0] = np.nan
        tokens = np.roll(TOKENS, step, axis=0)
        yield Batch(tokens, weight, None)


def train(*, pipelined, nan_at_step=None, checkpoint=True):
    optimizer = AdamConfig(learning_rate=1e-2, warmup=0.0, weight_decay=0.1).build(NUM_STEPS)
    train_step = june_train._make_train_step(
        optimizer,
        POLICY,
        z_loss_weight=0.0,
        ema_beta=None,
        watch_config=WATCH,
    )
    dispatched = []

    def counting_step(state, batch, *, compute_watch):
        dispatched.append(len(dispatched))
        return train_step(state, batch, compute_watch=compute_watch)

    hook_calls = []
    state_hook_calls = []
    runner = StateCallbackRunner(
        step_getter=lambda s: s.step,
        model_getter=lambda s: s.params,
        eval_model_getter=lambda s: s.params,
        opt_state_getter=lambda s: s.opt_state,
    )
    runner.add_hook(log_step_info(NUM_STEPS), every=1)
    runner.add_hook(lambda info, force=False: hook_calls.append((info.step, info.next_step, info.loss, force)), every=1)
    runner.add_hook(
        lambda info, force=False: state_hook_calls.append((info.step, force, param_sum(info.model))),
        every=STATE_HOOK_EVERY,
    )
    tracker = RecordingTracker()
    checkpointer = RecordingCheckpointer() if checkpoint else None
    error = None
    with levanter.tracker.current_tracker(tracker), jax.set_mesh(_mesh()):
        state = jax.jit(
            lambda key: june_train.initial_state(BASE, optimizer=optimizer, mp=POLICY, key=key, ema_beta=None)
        )(jax.random.key(0))
        try:
            state = june_train._train_loop(
                counting_step,
                state,
                LoadingTimeTrackerIterator(batches(nan_at_step)),
                num_train_steps=NUM_STEPS,
                state_callbacks=runner,
                checkpointer=checkpointer,
                watch_interval=WATCH.interval,
                step_timeout=None,
                diagnose_numerics=False,
                pipelined=pipelined,
                hooks_read_state=lambda step: (step + 1) % STATE_HOOK_EVERY == 0,
            )
        except FloatingPointError as e:
            error = str(e)
            state = None
    logged = {
        step: {key: value for key, value in metrics.items() if key.strip("[]'") not in TIMING_KEYS}
        for step, metrics in tracker.steps.items()
    }
    final = None if state is None else leaves(state)
    return dict(
        final=final,
        logged=logged,
        hooks=hook_calls,
        state_hooks=state_hook_calls,
        checkpoints=checkpointer.saved if checkpoint else None,
        dispatched=len(dispatched),
        error=error,
    )


@pytest.mark.timeout(600)
@pytest.mark.multidevice
def test_pipelined_loop_matches_the_synchronous_loop():
    with use_cpu_devices(8):
        sync = train(pipelined=False)
        pipelined = train(pipelined=True)
        # Without a checkpointer no step is fetched before the next dispatch except those whose hooks read the state.
        unchecked = train(pipelined=True, checkpoint=False)

        for run in (pipelined, unchecked):
            for a, b in zip(run["final"], sync["final"], strict=True):
                np.testing.assert_array_equal(a, b)
            assert run["logged"] == sync["logged"]
            assert run["hooks"] == sync["hooks"]
            assert run["state_hooks"] == sync["state_hooks"]
            assert run["dispatched"] == NUM_STEPS
        assert sorted(pipelined["logged"]) == sorted(sync["logged"]) == list(range(NUM_STEPS))
        assert pipelined["logged"] == sync["logged"]
        # The learning rate from the optimizer state, the router metrics, and the watch stats of steps 0 and 3.
        assert all("['optim/learning_rate']" in sync["logged"][step] for step in range(NUM_STEPS))
        assert any(key.startswith("['train/router/") for key in sync["logged"][0])
        watched = [step for step in range(NUM_STEPS) if any("grad" in key for key in sync["logged"][step])]
        assert watched == [0, 3]
        assert pipelined["hooks"] == sync["hooks"]
        assert [(s, n) for s, n, _, force in sync["hooks"] if not force] == [(s, s + 1) for s in range(NUM_STEPS)]
        # The hook that reads the model ran on the steps it is due (and the forced call), with their real states.
        assert pipelined["state_hooks"] == sync["state_hooks"]
        assert [(s, force) for s, force, _ in sync["state_hooks"]] == [(1, False), (3, False), (4, True)]
        assert pipelined["checkpoints"] == sync["checkpoints"]
        assert [(s, force) for s, force, _ in sync["checkpoints"]] == [(s, False) for s in range(1, 6)] + [(5, True)]
        assert pipelined["dispatched"] == sync["dispatched"] == NUM_STEPS


@pytest.mark.timeout(600)
@pytest.mark.multidevice
def test_nonfinite_loss_raises_one_step_late_when_pipelined():
    with use_cpu_devices(8):
        sync = train(pipelined=False, nan_at_step=2)
        pipelined = train(pipelined=True, nan_at_step=2, checkpoint=False)
        # With a checkpointer the pipelined loop checks each step before its state can be saved.
        checked = train(pipelined=True, nan_at_step=2)

        for run in (sync, pipelined, checked):
            assert run["error"].startswith("Non-finite Grug loss at step 3"), run["error"]
            # Steps 0 and 1 were hooked and logged; step 2 was not.
            assert [s for s, *_ in run["hooks"]] == [0, 1]
            assert sorted(run["logged"]) == [0, 1]
        for run in (sync, checked):
            # No state after the non-finite step reached the checkpointer.
            assert [s for s, *_ in run["checkpoints"]] == [1, 2]
            assert run["dispatched"] == 3
        # Step 3 was dispatched before step 2's loss was fetched.
        assert pipelined["dispatched"] == 4
