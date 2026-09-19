# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses

import equinox as eqx
import haliax as hax
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from haliax.partitioning import ResourceAxis

from levanter.grad_accum import microbatched
from levanter.grpo import GrpoConfig, KlGradient, grpo_advantages, grpo_loss, grpo_objective_weights
from levanter.testing.helpers import skip_if_no_torch, use_test_mesh


@dataclasses.dataclass(frozen=True)
class Rollout:
    Batch: hax.Axis
    Position: hax.Axis
    old: np.ndarray
    current: np.ndarray
    reference: np.ndarray
    mask: np.ndarray
    groups: np.ndarray
    rewards: np.ndarray
    partitions: np.ndarray


CONFIG = GrpoConfig(eps_clip_low=0.2, eps_clip_high=0.2, kl_loss_coef=0.001, kl_gradient=KlGradient.DETACHED)


@pytest.fixture
def rollout():
    Batch, Position = hax.Axis("batch", 8 * len(jax.devices())), hax.Axis("position", 5)
    rng = np.random.default_rng(18)
    old = rng.normal(-3, 0.3, (Batch.size, Position.size)).astype(np.float32)
    current = old + rng.uniform(-0.6, 0.6, old.shape).astype(np.float32)
    reference = old + rng.normal(0, 0.2, old.shape).astype(np.float32)
    # Empty responses, unequal lengths, and interleaved groups are intentional.
    lengths = np.resize([0, 1, 2, 5, 4, 1, 3, 5], Batch.size)
    mask = (np.arange(Position.size)[None, :] < lengths[:, None]).astype(np.float32)
    groups = np.resize([0, 1, 0, 1, 2, 2, 3, 4], Batch.size).astype(np.int32)
    rewards = np.zeros_like(old)
    rewards[:, -1] = np.resize([1, -2, 3, 4, 1, 1, 2, 0], Batch.size)
    partitions = np.arange(Batch.size, dtype=np.int32) // 2
    return Rollout(Batch, Position, old, current, reference, mask, groups, rewards, partitions)


def torch_advantages(rewards, mask, groups, normalize):
    import torch  # noqa: PLC0415  # optional dependency

    scores = torch.tensor(rewards).sum(-1)
    result = torch.zeros_like(scores)
    for group in np.unique(groups):
        rows = groups == group
        selected = scores[rows]
        mean = selected.mean() if len(selected) > 1 else 0
        std = selected.std() if len(selected) > 1 else 1
        result[rows] = (selected - mean) / (std + 1e-6) if normalize else selected - mean
    return result[:, None] * torch.tensor(mask)


def torch_objective(current, old, reference, advantages, mask, partitions, config):
    """Independent Torch reduction over the reference's actual objective partitions.

    Equations follow MarinSkyRL 8e33e017 policy_losses.py and policy_math.py; unlike the
    JAX implementation this executes one locally normalized loss per partition.
    """
    import torch  # noqa: PLC0415  # optional dependency

    objectives = []
    for partition in np.unique(partitions):
        rows = partitions == partition
        lp, prev, ref, adv, m = (x[rows] for x in (current, old, reference, advantages, mask))
        ratio = (lp - prev).float().clamp(-20, 20).exp().to(lp.dtype)
        policy = -torch.minimum(ratio * adv, ratio.clamp(0.8, 1.2) * adv)
        policy = (policy * m).sum() / m.sum().clamp(min=1)
        delta = (ref - lp).clamp(-20, 20)
        kl = (delta.exp() - delta - 1).clamp(-10, 10)
        if config.kl_gradient == KlGradient.DETACHED:
            kl = kl.detach()
        kl = ((kl * m).sum(-1) / m.sum(-1).clamp(min=1)).mean()
        objectives.append(policy + config.kl_loss_coef * kl)
    return torch.stack(objectives).mean()


@skip_if_no_torch
@pytest.mark.parametrize("normalize", [False, True])
def test_grpo_advantages_matches_torch_groups(rollout, normalize):
    B = rollout.Batch
    P = rollout.Position
    mask = rollout.mask
    groups = rollout.groups
    rewards = rollout.rewards
    actual = eqx.filter_jit(grpo_advantages)(
        hax.named(rewards, (B, P)),
        hax.named(mask, (B, P)),
        hax.named(groups, B),
        Batch=B,
        Position=P,
        num_groups=5,
        normalize_by_std=normalize,
    )
    expected = torch_advantages(rewards, mask, groups, normalize)
    np.testing.assert_allclose(actual.array, expected.numpy(), atol=1e-5, rtol=1e-5)


@skip_if_no_torch
@pytest.mark.parametrize(
    "kl_gradient, kl_coef",
    [(KlGradient.DETACHED, 0.0), (KlGradient.DETACHED, 0.001), (KlGradient.DIFFERENTIABLE, 0.001)],
)
@pytest.mark.parametrize("microbatch_size", [1, 2, 4])
def test_grpo_value_gradient_and_execution_partition_parity(rollout, kl_gradient, kl_coef, microbatch_size):
    import torch  # noqa: PLC0415  # optional dependency

    B = rollout.Batch
    P = rollout.Position
    old = rollout.old
    current = rollout.current
    reference = rollout.reference
    mask = rollout.mask
    groups = rollout.groups
    rewards = rollout.rewards
    partitions = rollout.partitions
    microbatch_size *= len(jax.devices())
    config = dataclasses.replace(CONFIG, kl_gradient=kl_gradient, kl_loss_coef=kl_coef)
    adv = torch_advantages(rewards, mask, groups, True)
    lp = torch.tensor(current, requires_grad=True)
    expected = torch_objective(
        lp, torch.tensor(old), torch.tensor(reference), adv, torch.tensor(mask), partitions, config
    )
    expected.backward()
    policy_weights, kl_weights = grpo_objective_weights(
        hax.named(mask, (B, P)),
        hax.named(partitions, B),
        Batch=B,
        Position=P,
        num_partitions=int(partitions.max()) + 1,
    )
    inputs = tuple(hax.named(x, (B, P)) for x in (old, reference, adv.numpy()))

    # A shared scalar perturbation permits real microbatched parameter gradients;
    # per-token gradients are also checked below against Torch autograd.
    def loss(offset, current, old, reference, advantages, pw, kw):
        return grpo_loss(
            current + offset,
            old,
            reference if kl_coef else None,
            advantages,
            pw,
            kw,
            config=config,
            accumulation_steps=B.size // microbatch_size,
        )

    mapping = {B.name: ResourceAxis.DATA}
    with use_test_mesh():
        fn = microbatched(
            eqx.filter_value_and_grad(loss, has_aux=True), B, microbatch_size, mapping, mapping, patch_in_rng_key=None
        )
        (value, metrics), grad = eqx.filter_jit(fn)(
            jnp.array(0.0),
            hax.named(current, (B, P)),
            *inputs,
            policy_weights,
            kl_weights,
        )
    np.testing.assert_allclose(value, expected.detach().numpy(), atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(grad, lp.grad.numpy().sum(), atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(metrics["loss"].value(), value, atol=1e-5, rtol=1e-5)

    # Compare the observable diagnostics against independently normalized Torch
    # partitions, including partitions with unequal numbers of response tokens.
    ratio = (lp.detach() - torch.tensor(old)).clamp(-20, 20).exp()
    selected = ratio.clamp(0.8, 1.2) * adv < ratio * adv
    for name, condition in (
        ("ppo_clip_ratio", selected),
        ("ppo_clip_ratio_low", selected & (ratio < 0.8)),
        ("ppo_clip_ratio_high", selected & (ratio > 1.2)),
        ("ppo_clip_pressure_low", ratio < 0.8),
        ("ppo_clip_pressure_high", ratio > 1.2),
        ("ppo_ratio_exact_unit_fraction", ratio == 1),
    ):
        fractions = []
        for partition in np.unique(partitions):
            rows = partitions == partition
            m = torch.tensor(mask[rows])
            fractions.append((condition[rows] * m).sum() / m.sum().clamp(min=1))
        expected_metric = torch.stack(fractions).mean().numpy()
        np.testing.assert_allclose(metrics[name].value(), expected_metric, atol=1e-5, rtol=1e-5)
    delta = (torch.tensor(reference) - lp.detach()).clamp(-20, 20)
    token_kl = (delta.exp() - delta - 1).clamp(-10, 10)
    expected_kl = ((token_kl * torch.tensor(mask)).sum(-1) / torch.tensor(mask).sum(-1).clamp(min=1)).mean()
    np.testing.assert_allclose(
        metrics["policy_kl"].value(), expected_kl.numpy() if kl_coef else 0, atol=1e-5, rtol=1e-5
    )
    np.testing.assert_allclose(
        metrics["log_ratio_abs_max"].value(), np.abs(current - old)[mask > 0].max(), atol=1e-5, rtol=1e-5
    )

    def full_loss(values):
        return grpo_loss(
            hax.named(values, (B, P)), *inputs, policy_weights, kl_weights, config=config, accumulation_steps=1
        )[0]

    gradient = jax.jit(jax.grad(full_loss))(jnp.array(current))
    np.testing.assert_allclose(gradient, lp.grad.numpy(), atol=1e-5, rtol=1e-5)
    assert np.max(np.abs(np.asarray(gradient)[mask == 0])) == 0


@skip_if_no_torch
def test_grpo_tiny_model_adamw_update_matches_torch(rollout):
    import torch  # noqa: PLC0415  # optional dependency

    B = rollout.Batch
    P = rollout.Position
    old = rollout.old
    reference = rollout.reference
    mask = rollout.mask
    groups = rollout.groups
    rewards = rollout.rewards
    partitions = rollout.partitions
    rng = np.random.default_rng(5)
    features = rng.normal(size=(B.size, P.size, 3)).astype(np.float32) * 1000
    initial = rng.normal(size=(3, 7)).astype(np.float32) * 1e-4
    labels = rng.integers(0, 7, size=(B.size, P.size))
    advantages = torch_advantages(rewards, mask, groups, True)
    param = torch.tensor(initial, requires_grad=True)
    torch_optimizer = torch.optim.AdamW([param], lr=1e-6, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
    log_probs = (
        torch.log_softmax(torch.tensor(features) @ param, dim=-1)
        .gather(-1, torch.tensor(labels)[..., None])
        .squeeze(-1)
    )
    # Keep ratios around one so the update exercises unclipped token gradients.
    old = log_probs.detach().numpy() + 0.05
    loss = torch_objective(
        log_probs, torch.tensor(old), torch.tensor(reference), advantages, torch.tensor(mask), partitions, CONFIG
    )
    loss.backward()
    torch_grad_norm = torch.nn.utils.clip_grad_norm_([param], max_norm=1.0)
    assert torch_grad_norm > 1.0
    torch_optimizer.step()

    pw, kw = grpo_objective_weights(
        hax.named(mask, (B, P)),
        hax.named(partitions, B),
        Batch=B,
        Position=P,
        num_partitions=int(partitions.max()) + 1,
    )
    inputs = tuple(hax.named(x, (B, P)) for x in (old, reference, advantages.numpy()))

    def jax_loss(weights):
        logits = jnp.asarray(features) @ weights
        lp = jnp.take_along_axis(jax.nn.log_softmax(logits), jnp.asarray(labels)[..., None], axis=-1)[..., 0]
        return grpo_loss(hax.named(lp, (B, P)), *inputs, pw, kw, config=CONFIG, accumulation_steps=1)[0]

    # Match Torch's clip_grad_norm_ epsilon explicitly rather than silently
    # changing either optimizer's clipping convention.
    optimizer = optax.adamw(1e-6, b1=0.9, b2=0.999, eps=1e-8, weight_decay=0.01)
    gradient = jax.grad(jax_loss)(jnp.asarray(initial))
    gradient *= jnp.minimum(1.0, 1.0 / (optax.global_norm(gradient) + 1e-6))
    updates, state = optimizer.update(gradient, optimizer.init(jnp.asarray(initial)), jnp.asarray(initial))
    actual = optax.apply_updates(jnp.asarray(initial), updates)
    actual_delta = np.asarray(actual) - initial
    expected_delta = param.detach().numpy() - initial
    np.testing.assert_allclose(actual_delta, expected_delta, atol=1e-10, rtol=1e-5)
    assert np.max(np.abs(actual_delta)) > 5e-7
    # First-step Adam largely cancels uniform gradient scaling in the update.
    # Moment parity therefore verifies that clipping actually happened.
    np.testing.assert_allclose(state[0].mu, torch_optimizer.state[param]["exp_avg"].numpy(), atol=1e-8, rtol=1e-5)
    np.testing.assert_allclose(state[0].nu, torch_optimizer.state[param]["exp_avg_sq"].numpy(), atol=1e-10, rtol=1e-5)


def test_grpo_zero_advantages_detached_kl_has_zero_surrogate_gradient(rollout):
    B = rollout.Batch
    P = rollout.Position
    old = rollout.old
    current = rollout.current
    reference = rollout.reference
    mask = rollout.mask
    partitions = rollout.partitions
    pw, kw = grpo_objective_weights(
        hax.named(mask, (B, P)),
        hax.named(partitions, B),
        Batch=B,
        Position=P,
        num_partitions=int(partitions.max()) + 1,
    )

    def loss(values, config):
        return grpo_loss(
            hax.named(values, (B, P)),
            hax.named(old, (B, P)),
            hax.named(reference, (B, P)),
            hax.zeros((B, P)),
            pw,
            kw,
            config=config,
            accumulation_steps=1,
        )[0]

    detached = jax.grad(lambda x: loss(x, CONFIG))(jnp.asarray(current))
    differentiable = jax.grad(lambda x: loss(x, dataclasses.replace(CONFIG, kl_gradient=KlGradient.DIFFERENTIABLE)))(
        jnp.asarray(current)
    )
    np.testing.assert_array_equal(detached, np.zeros_like(current))
    assert np.max(np.abs(differentiable)) > 0


def test_grpo_clamp_boundaries_preserve_pinned_torch_gradient():
    B, P = hax.Axis("batch", 2), hax.Axis("position", 4)
    # Torch 2.11.0 (MarinSkyRL 8e33e017's lock) passes the full clamp
    # derivative at endpoints. Torch 2.14 changed it to zero. These recorded
    # 2.11 gradients protect the migration contract independently of installed
    # Torch. Zero-width PPO bounds force an exact surrogate tie at ratio=1.
    current = np.array([[0, 0, 20, -20], [0, 0, 20, -20]], dtype=np.float32)
    adv = np.array([[1, -1, -1, 1], [-1, 1, 1, -1]], dtype=np.float32)
    expected = np.array(
        [
            [-0.125, 0.125, 60645648.0, -2.576442115e-10],
            [0.125, -0.125, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    config = GrpoConfig(0.0, 0.0, 0.0, KlGradient.DETACHED)

    def jax_loss(values):
        return grpo_loss(
            hax.named(values, (B, P)),
            hax.zeros((B, P)),
            None,
            hax.named(adv, (B, P)),
            hax.ones((B, P)) / 8,
            hax.ones((B, P)) / 8,
            config=config,
            accumulation_steps=1,
        )[0]

    actual = jax.grad(jax_loss)(jnp.asarray(current))
    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)
