# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The AMD June trainer on four CPU devices: microbatching and ZeRO-1 against the whole batch, and the training loop,
which dispatches step N+1 before it checks and logs step N."""

import collections
import concurrent.futures
import dataclasses
import re

import jax
import jax.numpy as jnp
import jmp
import numpy as np
import optax
import pytest
from jax.sharding import PartitionSpec as P
from levanter.callbacks.state_adapter import StateCallbackRunner
from levanter.checkpoint import Checkpointer, CheckpointInterval, load_checkpoint, save_checkpoint
from levanter.grug.attention import AttentionMask
from levanter.grug.grug_moe import MOE_DROPPED_ASSIGNMENTS_METRIC, MOE_SKIPPED_PADDING_ASSIGNMENTS_METRIC
from levanter.grug.sharding import compact_grug_mesh
from levanter.optim.config import AdamConfig
from levanter.testing.devices import use_test_devices
from levanter.utils.logging import LoadingTimeTrackerIterator

from experiments.amd.june_67b_a2b import train
from experiments.amd.june_67b_a2b.parallel import mark_batch_reduced
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
# Float32 parameters and a bfloat16 compute copy, as the MI355X runs train.
MIXED_POLICY = jmp.get_policy("p=f32,c=bfloat16")
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


# A float32 written over int32 words, as the MI355X stray writes do: 1.0f read as an int32.
FLOAT_BITS = np.float32(1.0).view(np.int32)


def _damage_tokens(batch):
    tokens = batch.tokens.copy()
    tokens[0, :2] = FLOAT_BITS
    return batch._replace(tokens=tokens)


def _damage_loss_weight(batch):
    weight = batch.loss_weight.copy()
    weight[1, 3] = np.nan
    return batch._replace(loss_weight=weight)


def _damage_segment_ids(batch):
    q_seg, kv_seg = batch.attn_mask.segment_ids
    kv_seg = kv_seg.copy()
    kv_seg[2, 4:6] = FLOAT_BITS
    return batch._replace(attn_mask=AttentionMask(is_causal=True, segment_ids=(q_seg, kv_seg)))


def _batches(damage_at_steps=(), damage=_damage_tokens):
    tokens = np.asarray(jax.random.randint(jax.random.key(1), (4, CONFIG.max_seq_len), 0, CONFIG.vocab_size))
    segment_ids = np.zeros(tokens.shape, np.int32)
    segment_ids[:, CONFIG.max_seq_len // 2 :] = 1
    for step in range(NUM_STEPS):
        weight = np.ones(tokens.shape, np.float32)
        mask = AttentionMask(is_causal=True, segment_ids=(segment_ids, segment_ids.copy()))
        batch = Batch(np.roll(tokens, step, axis=0), weight, mask)
        yield damage(batch) if step in damage_at_steps else batch


@pytest.fixture(scope="module")
def trainer():
    """One compiled train step and the initial state, on four CPU devices."""
    optimizer = AdamConfig(learning_rate=1e-2, warmup=0.0).build(NUM_STEPS)
    with use_test_devices(4), jax.set_mesh(compact_grug_mesh(expert_axis_size=2)):
        step = train._make_train_step(
            optimizer,
            POLICY,
            vocab_size=CONFIG.vocab_size,
            z_loss_weight=0.0,
            ema_beta=None,
            microbatches=1,
            zero1=False,
        )
        init = jax.jit(
            lambda key: train.initial_state(CONFIG, optimizer=optimizer, mp=POLICY, key=key, ema_beta=None, zero1=False)
        )
        yield step, lambda: init(jax.random.key(0))


def _poison_opt_state(state):
    """Non-finite optimizer moments: the next update makes the parameters non-finite while its loss stays finite."""
    nan_moments = jax.tree.map(lambda x: x * jnp.nan if jnp.issubdtype(x.dtype, jnp.floating) else x, state.opt_state)
    return dataclasses.replace(state, opt_state=nan_moments)


def _run_loop(trainer, *, poison_at_step=None, batches=None, checkpoint=True, checkpointer=None):
    train_step, initial = trainer
    dispatched = []

    def counting_step(state, batch, *, compute_watch):
        dispatched.append(state.step)
        if int(state.step) == poison_at_step:
            state = _poison_opt_state(state)
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
            LoadingTimeTrackerIterator(_batches() if batches is None else batches),
            num_train_steps=NUM_STEPS,
            state_callbacks=runner,
            checkpointer=checkpointer,
            watch_interval=0,
            step_timeout=None,
            diagnose_numerics=False,
            hooks_read_state=lambda step: (step + 1) % 2 == 0,
        )
    except (FloatingPointError, ValueError) as e:
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


def test_nonfinite_state_raises_one_step_late_and_is_never_checkpointed(trainer):
    unchecked = _run_loop(trainer, poison_at_step=2, checkpoint=False)
    checked = _run_loop(trainer, poison_at_step=2)

    for run in (unchecked, checked):
        assert run.error.startswith("Non-finite Grug state at step 3"), run.error
        # Steps 0 and 1 were hooked; step 2 was not.
        assert [s for s, *_ in run.hooks] == [0, 1]
    # Without a checkpointer, step 3 was dispatched before step 2's loss was fetched.
    assert unchecked.dispatched == 4
    # With one, step 2 is checked before its state can be saved: no state after the non-finite step is saved.
    assert checked.checkpoints is not None and [s for s, *_ in checked.checkpoints] == [1, 2]
    assert checked.dispatched == 3


@pytest.mark.parametrize("damage", [_damage_tokens, _damage_loss_weight, _damage_segment_ids])
def test_step_with_damaged_inputs_keeps_the_state_and_training_continues(trainer, damage):
    train_step, initial = trainer
    clean = list(_batches())
    state = initial()
    for step in (0, 1, 3):
        state, _, _ = train_step(state, clean[step], compute_watch=False)

    run = _run_loop(trainer, batches=_batches(damage_at_steps={2}, damage=damage))

    assert run.error is None
    # The damaged batch changed nothing but the step count: the state is that of training on the other three batches.
    assert int(run.state.step) == NUM_STEPS
    for got, want in zip(
        jax.tree.leaves((run.state.params, run.state.opt_state)),
        jax.tree.leaves((state.params, state.opt_state)),
        strict=True,
    ):
        np.testing.assert_array_equal(got, want)
    assert [s for s, _, force in run.hooks if not force] == list(range(NUM_STEPS))
    assert run.checkpoints is not None and [s for s, *_ in run.checkpoints] == [1, 2, 3, 4, 4]


def test_damaged_inputs_on_every_step_raise(trainer):
    run = _run_loop(trainer, batches=_batches(damage_at_steps=set(range(NUM_STEPS))))

    assert run.error is not None and run.error.startswith("4 consecutive steps up to 3 failed the input check")


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


# Microbatching and ZeRO-1. Eight rows, four batch shards (replica_dcn=2 x expert=2): K=2 is one row per device per
# microbatch. Rows 0, 3 and 6 end in three padding positions, so the microbatches route unequal numbers of tokens, and
# the router z-loss is on, so its token-share weighting matters.
ACCUMULATION_CONFIG = dataclasses.replace(CONFIG, router_z_loss_coef=0.01)
_SEGMENT_IDS = np.zeros((8, CONFIG.max_seq_len), np.int32)
_SEGMENT_IDS[::3, -3:] = -1
_LOSS_WEIGHT = np.where(_SEGMENT_IDS >= 0, 1.0, 0.0).astype(np.float32)
_LOSS_WEIGHT[:, -1] = 0
PADDED_BATCH = Batch(
    np.asarray(jax.random.randint(jax.random.key(1), _SEGMENT_IDS.shape, 0, CONFIG.vocab_size)),
    _LOSS_WEIGHT,
    AttentionMask(is_causal=True, segment_ids=(_SEGMENT_IDS, _SEGMENT_IDS)),
)
# Plain momentum SGD: the first update is the gradient times the learning rate, and its state has the weights' shapes.
OPTIMIZER = optax.sgd(0.5, momentum=0.9)
# Compiled without the collective combiners, so every reduction and gather keeps its own shape and op_name.
NO_COMBINERS = {"xla_disable_hlo_passes": "cpu-all-reduce-combiner,cpu-all-gather-combiner"}


@dataclasses.dataclass
class AccumulationRun:
    states: list  # The state after each step, on the host.
    losses: list[float]
    metrics: dict  # The first step's metrics, on the host.
    hlo: str
    train_step: object  # The compiled step.
    initial: object


def _accumulation_mesh():
    return compact_grug_mesh(expert_axis_size=2, replica_axis_size=2)


@pytest.fixture(scope="module")
def accumulation_runs():
    """Two steps of the train step for each ``(microbatches, zero1, policy)``, compiled once without the combiners;
    cached. The steps a call needs compile concurrently (XLA releases the GIL). The four devices stay up for the
    module, so the compiled steps can run again."""
    runs = {}

    def run(*keys: tuple[int, bool, jmp.Policy]) -> list[AccumulationRun]:
        with jax.set_mesh(_accumulation_mesh()):
            lowered = {}
            for key in dict.fromkeys(keys):
                if key in runs:
                    continue
                microbatches, zero1, policy = key
                initial = jax.jit(
                    lambda k, policy=policy, zero1=zero1: train.initial_state(
                        ACCUMULATION_CONFIG, optimizer=OPTIMIZER, mp=policy, key=k, ema_beta=None, zero1=zero1
                    )
                )
                train_step = train._make_train_step(
                    OPTIMIZER,
                    policy,
                    vocab_size=CONFIG.vocab_size,
                    z_loss_weight=1e-3,
                    ema_beta=None,
                    microbatches=microbatches,
                    zero1=zero1,
                )
                state = initial(jax.random.key(0))
                lowered[key] = (initial, state, train_step.lower(state, PADDED_BATCH))
            with concurrent.futures.ThreadPoolExecutor() as pool:
                compiled = dict(
                    zip(
                        lowered,
                        pool.map(lambda entry: entry[2].compile(compiler_options=NO_COMBINERS), lowered.values()),
                        strict=True,
                    )
                )
            for key, (initial, state, _) in lowered.items():
                step = compiled[key]
                states, losses, first_metrics = [], [], None
                for _ in range(2):
                    state, metrics, _ = step(state, PADDED_BATCH)
                    states.append(jax.tree.map(np.asarray, state))
                    losses.append(float(metrics["train/loss"]))
                    first_metrics = first_metrics or jax.tree.map(np.asarray, metrics)
                runs[key] = AccumulationRun(
                    states,
                    losses,
                    first_metrics,
                    step.as_text(),
                    step,
                    lambda initial=initial: initial(jax.random.key(0)),
                )
        return [runs[key] for key in keys]

    with use_test_devices(4):
        yield run


def _relative_errors(tree, reference) -> dict[str, float]:
    return {
        jax.tree_util.keystr(path): float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))
        for (path, a), b in zip(jax.tree_util.tree_leaves_with_path(tree), jax.tree.leaves(reference), strict=True)
    }


def _specs(tree) -> dict[str, object]:
    return {jax.tree_util.keystr(path): x.sharding.spec for path, x in jax.tree_util.tree_leaves_with_path(tree)}


def _is_expert(path) -> bool:
    return "expert_mlp" in jax.tree_util.keystr(path)


Collective = collections.namedtuple("Collective", "kind elements in_loop op_name")


def _replica_collectives(hlo: str, mesh) -> list[Collective]:
    """The all-reduces, reduce-scatters and all-gathers of a compiled CPU module whose device groups cross
    ``replica_dcn``, with non-scalar operands: their kind, the elements of their result, whether they run in a while
    loop (the layer scans) and their source op. Parses XLA's HLO text (the replica_groups forms) and JAX's op_name
    metadata, so a JAX or XLA upgrade that changes either format needs this updated."""
    computations, current = {}, None
    for line in hlo.splitlines():
        header = re.match(r"(ENTRY )?%?([\w.\-]+) .*\{$", line)
        if header and not line.startswith(" "):
            current = header.group(2)
            computations[current] = (bool(header.group(1)), [])
        elif current and line.startswith(" "):
            computations[current][1].append(line.strip())
    in_loop, stack = {}, [(name, False) for name, (entry, _) in computations.items() if entry]
    while stack:
        name, looped = stack.pop()
        if name in in_loop and in_loop[name] >= looped:
            continue
        in_loop[name] = looped
        for line in computations[name][1]:
            body = re.search(r"\bbody=%?([\w.\-]+)", line)
            for callee in re.findall(r"%([\w.\-]+)", line):
                if callee in computations:
                    stack.append((callee, looped or (body is not None and callee == body.group(1))))
    replica = np.unravel_index(np.arange(mesh.devices.size), mesh.devices.shape)[mesh.axis_names.index("replica_dcn")]
    out = []
    for name, (_, lines) in computations.items():
        for line in lines:
            m = re.match(r"(?:ROOT )?%?[\w.\-]+ = (\w+)\[([\d,]*)\]\S* ([\w\-]+)\((.*)$", line)
            kinds = ("all-reduce", "all-reduce-start", "reduce-scatter", "all-gather", "all-gather-start")
            if not m or m.group(3) not in kinds or not m.group(2):
                continue
            attrs = m.group(4)
            named = re.search(
                r"replica_groups=mesh\[([^\]]*)\](?:, device_ids=\(\[([\d,]+)\](?:T\(([\d,]+)\))?\))?\s*\{([^}]*)\}",
                attrs,
            )
            iota = re.search(r"replica_groups=\[(\d+),(\d+)\]<=\[([\d,]+)\](?:T\(([\d,]+)\))?", attrs)
            if named:
                sizes = [(axis, int(size)) for axis, size in re.findall(r"'(\w+)'=(\d+)", named.group(1))]
                names, axes = [axis for axis, _ in sizes], re.findall(r"'(\w+)'", named.group(4))
                ids = np.arange(mesh.devices.size)
                if named.group(2):
                    ids = ids.reshape([int(d) for d in named.group(2).split(",")])
                    if named.group(3):
                        ids = ids.transpose([int(d) for d in named.group(3).split(",")])
                ids = np.moveaxis(
                    ids.reshape([size for _, size in sizes]), [names.index(a) for a in axes], range(-len(axes), 0)
                )
                groups = ids.reshape(-1, int(np.prod([size for axis, size in sizes if axis in axes]))).tolist()
            elif iota:
                count, size, dims, permutation = iota.groups()
                ids = np.arange(mesh.devices.size).reshape([int(d) for d in dims.split(",")])
                if permutation:
                    ids = ids.transpose([int(d) for d in permutation.split(",")])
                groups = ids.reshape(int(count), int(size)).tolist()
            else:
                listed = re.search(r"replica_groups=\{(.*?)\}\}", attrs)
                assert listed, line
                groups = [[int(d) for d in g.split(",")] for g in re.findall(r"\{([\d,]+)", listed.group(0))]
            if any(len({replica[d] for d in group}) > 1 for group in groups):
                op_name = re.search(r'op_name="([^"]*)"', attrs)
                elements = int(np.prod([int(d) for d in m.group(2).split(",")]))
                out.append(Collective(m.group(3).removesuffix("-start"), elements, in_loop[name], op_name.group(1)))
    return out


def test_microbatched_step_matches_the_whole_batch(accumulation_runs):
    # Two microbatches of unequal routed tokens give the whole batch's loss, metrics and (through SGD) gradients:
    # each microbatch's loss share and the once-per-step reduction of the float32 sums reorder sums only (8.6e-8
    # relative observed). The QB thresholds are estimated per microbatch, so only the first step is comparable.
    whole, microbatched = accumulation_runs((1, False, POLICY), (2, False, POLICY))

    np.testing.assert_allclose(microbatched.losses[0], whole.losses[0], rtol=1e-6)
    for key in ("train/cross_entropy_loss", "train/router/router_z_loss", "train/router/aux_loss_weighted"):
        np.testing.assert_allclose(microbatched.metrics[key], whole.metrics[key], rtol=1e-6, err_msg=key)
    for key in (MOE_DROPPED_ASSIGNMENTS_METRIC, MOE_SKIPPED_PADDING_ASSIGNMENTS_METRIC):
        np.testing.assert_array_equal(microbatched.metrics[key], whole.metrics[key], err_msg=key)
    errors = _relative_errors(microbatched.states[0].params, whole.states[0].params)
    assert max(errors.values()) <= 1e-6, errors


def test_zero1_matches_microbatching_and_crosses_the_replicas_once(accumulation_runs):
    # In mixed precision: float32 parameters, optimizer state and sums, a bfloat16 compute copy.
    microbatched, zero1 = accumulation_runs((2, False, MIXED_POLICY), (2, True, MIXED_POLICY))
    mesh = _accumulation_mesh()

    # The compute copy is the updated float32 parameters' bfloat16 cast.
    for copy, params in zip(
        jax.tree.leaves(microbatched.states[0].compute_params),
        jax.tree.leaves(microbatched.states[0].params),
        strict=True,
    ):
        assert copy.dtype == jnp.bfloat16 and params.dtype == jnp.float32
        np.testing.assert_array_equal(copy, params.astype(jnp.bfloat16))
    # The same arithmetic in the same order: two steps match bit for bit, the second gathering the first's update.
    assert zero1.losses == microbatched.losses
    for got, want in zip(zero1.states, microbatched.states, strict=True):
        for a, b in zip(jax.tree.leaves(got.params), jax.tree.leaves(want.params), strict=True):
            np.testing.assert_array_equal(a, b)
    # The expert weights' parameters and optimizer state are sharded over the replicas as well; the rest, and the
    # compute copy, keep the model's sharding.
    with jax.set_mesh(mesh):
        zero1_state, state = zero1.initial(), microbatched.initial()
        for tree in ("params", "opt_state"):
            for (path, spec), want in zip(
                _specs(getattr(zero1_state, tree)).items(), _specs(getattr(state, tree)).values(), strict=True
            ):
                expected = want
                if "expert_mlp" in path:
                    expected = P(*(("expert", "replica_dcn") if entry == "expert" else entry for entry in want))
                assert spec == expected, (tree, path, spec, want)
        assert _specs(zero1_state.compute_params) == _specs(state.compute_params)
        marked = jax.eval_shape(mark_batch_reduced, state.compute_params)

    def local_elements(x, *, layer):
        shape = x.sharding.shard_shape(x.shape)
        return int(np.prod(shape[1:] if layer else shape))

    expert_slices = sorted(
        local_elements(x, layer=True) for path, x in jax.tree_util.tree_leaves_with_path(marked) if _is_expert(path)
    )
    other_grads = collections.Counter(
        local_elements(x, layer=False)
        for path, x in jax.tree_util.tree_leaves_with_path(marked)
        if not _is_expert(path) and getattr(path[-1], "name", None) != "router_bias"
    )
    collectives = _replica_collectives(zero1.hlo, mesh)
    reductions = [c for c in collectives if c.kind != "all-gather"]

    # Against microbatching alone, each expert weight is gathered once per step, one layer's slice per iteration of
    # the first microbatch's forward layer scan, and nowhere else: not in a backward, not after the update.
    # Matched by place and op_name, without the microbatch loop's prefix: here the first microbatch runs before it.
    def gathers(listed):
        return collections.Counter(
            c._replace(op_name=c.op_name.replace("jit(train_step)/while/body/eval_jaxpr/", "jit(train_step)/", 1))
            for c in listed
            if c.kind == "all-gather"
        )

    added = gathers(collectives)
    added.subtract(gathers(_replica_collectives(microbatched.hlo, mesh)))
    assert not -added, -added
    added = list((+added).elements())
    assert sorted(c.elements for c in added) == expert_slices, added
    assert all(c.in_loop and "transpose(" not in c.op_name for c in added), added
    # Its gradient is reduce-scattered once per step, layer by layer in the last microbatch's backward (on CPU an
    # all-reduce and a slice); the first microbatch's backward reduces nothing. Every other gradient is reduced once,
    # after the layer scans; the forward's routing statistics are not gradients.
    in_loop = [c for c in reductions if c.in_loop]
    assert sorted(c.elements for c in in_loop) == expert_slices and all("transpose(" in c.op_name for c in in_loop)
    after = collections.Counter(c.elements for c in reductions if not c.in_loop and "jvp()" not in c.op_name)
    assert not other_grads - after and all(size < min(other_grads) for size in after - other_grads), after


def test_zero1_checkpoint_restores_without_zero1(tmp_path):
    # Checkpoints hold the global shapes without the compute copy: a state saved with ZeRO-1 restores without it,
    # and its compute copy is rebuilt in the model's sharding.
    def initial(zero1):
        return jax.jit(
            lambda k: train.initial_state(
                ACCUMULATION_CONFIG, optimizer=OPTIMIZER, mp=MIXED_POLICY, key=k, ema_beta=None, zero1=zero1
            )
        )(jax.random.key(0))

    with use_test_devices(4), jax.set_mesh(_accumulation_mesh()):
        saved = initial(True)
        save_checkpoint(train._checkpointed(saved), step=1, checkpoint_path=str(tmp_path))
        exemplar = initial(False)
        restored = load_checkpoint(train._checkpointed(exemplar), str(tmp_path), mesh=_accumulation_mesh())
        restored = train.rebuilt_compute_params(restored, MIXED_POLICY, train.compute_shardings(exemplar))
        assert _specs(restored.params) == _specs(exemplar.params)
        assert _specs(restored.compute_params) == _specs(exemplar.compute_params)
        for a, b in zip(jax.tree.leaves(restored.params), jax.tree.leaves(saved.params), strict=True):
            np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
        for a, b in zip(jax.tree.leaves(restored.opt_state), jax.tree.leaves(saved.opt_state), strict=True):
            np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
        for copy, params in zip(jax.tree.leaves(restored.compute_params), jax.tree.leaves(saved.params), strict=True):
            np.testing.assert_array_equal(np.asarray(copy), np.asarray(params).astype(jnp.bfloat16))
