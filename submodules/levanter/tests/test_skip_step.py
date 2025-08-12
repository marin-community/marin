import tempfile

import chex
import jax
import numpy as np
import optax
import pytest
from jax import numpy as jnp

import haliax

from levanter.optim import AdamConfig
from levanter.optim.skipstep import SkipStepConfig, SkipStepState
from levanter.tensorstore_serialization import tree_deserialize_leaves_tensorstore, tree_serialize_leaves_tensorstore


@pytest.fixture
def initialized_wrapped_optimizer():
    mock_opt = mock_optimizer_transform()
    skip_conf = SkipStepConfig(rolling_interval_length=10)  # smaller interval for tests
    wrapped_opt = skip_conf.wrap(mock_opt)
    dummy_params = {"w": jnp.array([1.0, 2.0, 3.0]), "b": jnp.array([0.5])}
    initial_state = wrapped_opt.init(dummy_params)
    return wrapped_opt, initial_state, dummy_params, skip_conf


# Helper function to run update_fn for a number of steps to populate history
def populate_history(wrapped_optimizer, initial_state, params, num_steps, base_loss, base_grads):
    current_state = initial_state
    for i in range(num_steps):
        # Vary loss and grads slightly to have some variance
        loss = base_loss + jnp.sin(i * 0.1)
        grads = jax.tree_util.tree_map(lambda g: g * (1 + 0.1 * jnp.cos(i * 0.1)), base_grads)
        _, current_state = wrapped_optimizer.update(grads, current_state, params, loss=loss)
    return current_state


@pytest.mark.parametrize("trigger_high_loss", [True, False])
def test_update_fn_skip_logic(initialized_wrapped_optimizer, trigger_high_loss):
    wrapped_optimizer, initial_state, params, config = initialized_wrapped_optimizer

    # Populate history until skipping can occur
    min_data_points = max(2, config.rolling_interval_length // 2)

    base_loss = jnp.array(5.0)
    # Gradients that result in global_norm of sqrt(0.1^2 * 3 + 0.05^2) = sqrt(0.03 + 0.0025) = sqrt(0.0325) approx 0.18
    base_grads = {"w": jnp.array([0.1, 0.1, 0.1]), "b": jnp.array([0.05])}

    # Populate history with "normal" values
    state_after_population = populate_history(
        wrapped_optimizer, initial_state, params, min_data_points, base_loss, base_grads
    )

    # Capture inner state before potentially skipping
    pre_skip_inner_opt_state = state_after_population.inner_opt_state

    # Calculate current mean/std to determine what "high" is
    valid_losses = state_after_population.losses[: state_after_population.count]
    loss_mean = jnp.nanmean(valid_losses)
    loss_std = jnp.nanstd(valid_losses)
    loss_std_safe = jnp.maximum(loss_std, 1e-6)  # as in implementation

    valid_gnorms = state_after_population.grad_norms[: state_after_population.count]
    gnorm_mean = jnp.nanmean(valid_gnorms)
    gnorm_std = jnp.nanstd(valid_gnorms)
    gnorm_std_safe = jnp.maximum(gnorm_std, 1e-6)  # as in implementation

    if trigger_high_loss:
        # Trigger skip due to high loss
        trigger_loss = loss_mean + config.sigma_factor * loss_std_safe + 1.0  # Ensure it's above threshold
        trigger_grads = base_grads
    else:
        # Trigger skip due to high grad norm
        trigger_loss = base_loss
        # Calculate grads that would result in a high norm
        # Target norm: gnorm_mean + config.sigma_factor * gnorm_std_safe + 1.0
        target_high_gnorm = gnorm_mean + config.sigma_factor * gnorm_std_safe + 1.0
        # Scale base_grads to achieve this norm. Current norm is optax.global_norm(base_grads)
        current_gnorm_val = optax.global_norm(base_grads)
        scaling_factor = target_high_gnorm / jnp.maximum(current_gnorm_val, 1e-6)
        trigger_grads = jax.tree_util.tree_map(lambda g: g * scaling_factor, base_grads)
        # Verify gnorm is high
        actual_trigger_gnorm = optax.global_norm(trigger_grads)
        chex.assert_trees_all_close(actual_trigger_gnorm, target_high_gnorm, atol=1e-3)

    updates, new_state = wrapped_optimizer.update(trigger_grads, state_after_population, params, loss=trigger_loss)

    # Assert step_factor was 0.0 (gradients passed to mock optimizer are zero)
    # Mock optimizer negates gradients, so if grads were zeroed, output is zero
    expected_zero_updates = jax.tree_util.tree_map(jnp.zeros_like, trigger_grads)
    chex.assert_trees_all_close(
        updates, expected_zero_updates, atol=1e-5, custom_message="Updates were not zeroed out when step was skipped."
    )

    # Assert inner optimizer state is preserved
    chex.assert_trees_all_close(
        new_state.inner_opt_state,
        pre_skip_inner_opt_state,
        custom_message="Inner optimizer state was modified during a skipped step.",
    )

    # Verify state updates (loss and grad_norm are recorded, count/idx update)
    last_idx = state_after_population.current_idx  # This was the index before the update call
    assert np.isclose(new_state.losses[last_idx], trigger_loss)
    assert np.isclose(new_state.grad_norms[last_idx], optax.global_norm(trigger_grads))
    assert new_state.count == state_after_population.count + 1
    assert new_state.current_idx == (state_after_population.current_idx + 1) % (config.rolling_interval_length)


def test_update_fn_proceed_normally(initialized_wrapped_optimizer):
    wrapped_optimizer, initial_state, params, config = initialized_wrapped_optimizer
    min_data_points = max(2, config.rolling_interval_length // 2)

    base_loss = jnp.array(5.0)
    base_grads = {"w": jnp.array([0.1, 0.1, 0.1]), "b": jnp.array([0.05])}

    state_after_population = populate_history(
        wrapped_optimizer, initial_state, params, min_data_points, base_loss, base_grads
    )

    # Values that should not trigger skipping
    normal_loss = base_loss
    normal_grads = base_grads

    updates, new_state = wrapped_optimizer.update(normal_grads, state_after_population, params, loss=normal_loss)

    # Assert step_factor was 1.0
    expected_inner_updates = jax.tree_util.tree_map(lambda g: -g, normal_grads)  # Mock optimizer negates
    chex.assert_trees_all_close(updates, expected_inner_updates, atol=1e-5)

    # Verify state updates
    last_idx = state_after_population.current_idx
    assert np.isclose(new_state.losses[last_idx], normal_loss)
    assert np.isclose(new_state.grad_norms[last_idx], optax.global_norm(normal_grads))


def test_update_fn_circular_buffer_and_count_cap(initialized_wrapped_optimizer):
    wrapped_optimizer, initial_state, params, config = initialized_wrapped_optimizer

    num_updates = config.rolling_interval_length + 5  # Exceed buffer length

    dummy_grads = {"w": jnp.array([0.1, 0.2, 0.3]), "b": jnp.array([0.05])}

    current_state = initial_state
    recorded_losses = []
    recorded_gnorms = []

    for i in range(num_updates):
        loss_val = jnp.array(float(i))
        # Ensure grads are not zero, otherwise global_norm might be zero, which could interact poorly with std calculation if not careful
        # (though current skip logic should handle std=0 via jnp.maximum(std, 1e-6))
        current_grads = jax.tree_util.tree_map(lambda g: g + i * 0.01, dummy_grads)

        _, current_state = wrapped_optimizer.update(current_grads, current_state, params, loss=loss_val)
        recorded_losses.append(loss_val)
        recorded_gnorms.append(optax.global_norm(current_grads))

    # Verify count caps at rolling_interval_length
    assert current_state.count == config.rolling_interval_length

    # Verify current_idx wraps around
    # current_idx is where the *next* write will happen.
    # After N updates, current_idx should be N % (length+1)
    # So after num_updates updates, current_idx should be num_updates % (length+1)
    expected_idx = num_updates % (config.rolling_interval_length)
    assert current_state.current_idx == expected_idx

    # Verify that losses and grad_norms arrays store the most recent values
    # The history arrays store (rolling_interval_length) values.
    # The values in the state.losses/grad_norms should be the last (rolling_interval_length) recorded values.
    expected_stored_losses = np.array(recorded_losses[-(config.rolling_interval_length) :])
    expected_stored_gnorms = np.array(recorded_gnorms[-(config.rolling_interval_length) :])

    # The order in state.losses/grad_norms might be wrapped. We need to reconstruct it.
    # current_idx points to the oldest value + 1 (or where the next value will be written)
    # So, the values are from current_idx to end, then from start to current_idx-1
    actual_losses_ordered = jnp.concatenate(
        [current_state.losses[current_state.current_idx :], current_state.losses[: current_state.current_idx]]
    )
    actual_gnorms_ordered = jnp.concatenate(
        [current_state.grad_norms[current_state.current_idx :], current_state.grad_norms[: current_state.current_idx]]
    )

    chex.assert_trees_all_close(actual_losses_ordered, expected_stored_losses, atol=1e-5)
    chex.assert_trees_all_close(actual_gnorms_ordered, expected_stored_gnorms, atol=1e-5)


def test_integration_with_optimizer_config_build():
    skip_config = SkipStepConfig(rolling_interval_length=5, sigma_factor=1.0)  # Aggressive skipping for test
    adam_config = AdamConfig(learning_rate=1e-3, skip_bad_steps=skip_config)

    optimizer = adam_config.build(num_train_steps=100)

    # Check that the built optimizer is indeed wrapped
    # This is a bit of an indirect check. We look for the structure of SkipStepState.
    dummy_params = {"w": jnp.array([1.0, 2.0]), "b": jnp.array([0.5])}
    initial_state = optimizer.init(dummy_params)

    initial_skip_state = initial_state.inner_state  # Should be SkipStepState
    assert isinstance(initial_skip_state, SkipStepState), "Optimizer was not wrapped by SkipStep"
    assert initial_skip_state.losses.shape == (skip_config.rolling_interval_length,)

    # Try a simple update sequence to see if it can skip
    # Populate history
    current_state = initial_state
    base_loss = jnp.array(1.0)
    base_grads = {"w": jnp.array([0.01, 0.01]), "b": jnp.array([0.01])}  # Small, consistent grads

    min_data_points = max(2, skip_config.rolling_interval_length // 2)
    for i in range(min_data_points):
        _, current_state = optimizer.update(base_grads, current_state, dummy_params, loss=base_loss + i * 0.1)

    # Now, a high loss update
    high_loss = base_loss + 100.0  # Should be >> mean + 1.0 * std

    # Need to get the actual learning rate for this step to compare updates
    # The AdamConfig optimizer has an injected hyperparam for LR.
    # The _optimizer inside AdamConfig returns a chain.
    # The last step is scale(-lr). If step is skipped, updates are 0.
    # If not skipped, updates are -lr * (adam_scaled_grads).

    # Since we can't easily get the lr for a specific step from the outside here without
    # re-implementing lr_scheduler logic, we'll check if the updates are zero.
    # If they are zero, it means the step was skipped.

    # Capture inner state before potentially skipping
    pre_skip_inner_opt_state = current_state.inner_state.inner_opt_state

    updates, final_state = optimizer.update(base_grads, current_state, dummy_params, loss=high_loss)

    # If skipped, updates passed to the *inner* optimizer are zero.
    # The Adam optimizer itself has state (m, v). Here, we are checking the *parameter updates*.
    # If step_factor is 0, then scaled_updates are 0.
    # Then inner_optimizer.update(0, ...) will produce some updates based on its own logic
    # (e.g. for Adam, if updates are 0, then m and v decay, but param change is 0).
    # So, the *returned* updates should be zero.
    expected_zero_updates = jax.tree_util.tree_map(jnp.zeros_like, base_grads)
    chex.assert_trees_all_close(
        updates,
        expected_zero_updates,
        atol=1e-7,
        custom_message="Updates were not zeroed out when step was skipped in integration test.",
    )

    # Assert inner optimizer state is preserved
    chex.assert_trees_all_close(
        final_state.inner_state.inner_opt_state,
        pre_skip_inner_opt_state,
        custom_message="Inner optimizer state was modified during a skipped step in integration test.",
    )


# Mock optimizer for testing SkipStepConfig.wrap
# A simple optimizer that just scales gradients by -1 and maintains a count state
def mock_optimizer_transform() -> optax.GradientTransformation:
    def init_fn(params):
        return {"count": jnp.array(0, dtype=jnp.int32)}

    def update_fn(updates, state, params=None):
        new_state = {"count": state["count"] + 1}
        # simple transformation: negate gradients
        transformed_updates = jax.tree_util.tree_map(lambda g: -g, updates)
        return transformed_updates, new_state

    return optax.GradientTransformation(init_fn, update_fn)


def test_skip_step_config_wrap_basic_setup():
    mock_optimizer = mock_optimizer_transform()
    skip_config = SkipStepConfig()

    wrapped_optimizer = skip_config.wrap(mock_optimizer)

    assert isinstance(wrapped_optimizer, optax.GradientTransformation)
    # Check if it has init and update attributes
    assert hasattr(wrapped_optimizer, "init")
    assert hasattr(wrapped_optimizer, "update")


def test_skip_step_init_fn():
    mock_optimizer = mock_optimizer_transform()
    skip_config = SkipStepConfig(rolling_interval_length=10)  # Use a smaller interval for easier testing
    wrapped_optimizer = skip_config.wrap(mock_optimizer)

    dummy_params = {"w": jnp.array([1.0, 2.0])}
    state: SkipStepState = wrapped_optimizer.init(dummy_params)

    # Check inner_opt_state
    expected_inner_state = mock_optimizer.init(dummy_params)
    chex.assert_trees_all_close(state.inner_opt_state, expected_inner_state)

    # Check losses and grad_norms arrays
    assert state.losses.shape == (skip_config.rolling_interval_length,)
    assert state.grad_norms.shape == (skip_config.rolling_interval_length,)
    assert jnp.all(state.losses == 0)
    assert jnp.all(state.grad_norms == 0)

    # Check current_idx and count
    chex.assert_trees_all_close(state.current_idx, jnp.array(0, dtype=jnp.int32))
    chex.assert_trees_all_close(state.count, jnp.array(0, dtype=jnp.int32))


def test_skip_step_update_fn_loss_kwarg_requirement():
    mock_optimizer = mock_optimizer_transform()
    skip_config = SkipStepConfig()
    wrapped_optimizer = skip_config.wrap(mock_optimizer)

    dummy_params = {"w": jnp.array([1.0, 2.0])}
    dummy_grads = {"w": jnp.array([-0.1, 0.1])}
    initial_state = wrapped_optimizer.init(dummy_params)

    with pytest.raises(ValueError, match="Loss must be provided"):
        wrapped_optimizer.update(dummy_grads, initial_state, dummy_params)

    # Test that it works if loss is provided
    try:
        wrapped_optimizer.update(dummy_grads, initial_state, dummy_params, loss=jnp.array(1.0))
    except ValueError:
        pytest.fail("ValueError raised unexpectedly when loss was provided.")


def test_update_fn_initial_phase_not_enough_data(initialized_wrapped_optimizer):
    wrapped_optimizer, state, params, config = initialized_wrapped_optimizer

    num_initial_steps = config.rolling_interval_length // 2 - 1
    if num_initial_steps < 1:  # ensure at least one step for the test logic
        num_initial_steps = 1

    dummy_updates = {"w": jnp.array([0.1, 0.2, 0.3]), "b": jnp.array([0.05])}
    loss_val = jnp.array(10.0)

    current_state = state
    for i in range(num_initial_steps):
        updates, new_state = wrapped_optimizer.update(dummy_updates, current_state, params, loss=loss_val + i)

        # Assert step_factor was 1.0 by checking if inner optimizer was applied
        # Mock optimizer negates gradients
        expected_inner_updates = jax.tree_util.tree_map(lambda g: -g, dummy_updates)
        chex.assert_trees_all_close(updates, expected_inner_updates, atol=1e-5)

        # Check state updates
        assert new_state.count == i + 1
        assert new_state.current_idx == (i + 1) % (config.rolling_interval_length)
        assert np.isclose(new_state.losses[i], loss_val + i)
        expected_gnorm = optax.global_norm(dummy_updates)
        assert np.isclose(new_state.grad_norms[i], expected_gnorm)

        current_state = new_state

    # Check that after these initial steps, count is num_initial_steps
    assert current_state.count == num_initial_steps
    # And current_idx has advanced
    assert current_state.current_idx == num_initial_steps % (config.rolling_interval_length)


def test_flatten_unflatten_skip_step_state():
    # Create a SkipStepState instance
    mock_opt = mock_optimizer_transform()
    skip_conf = SkipStepConfig(rolling_interval_length=5)
    wrapped_opt = skip_conf.wrap(mock_opt)
    dummy_params = {"w": jnp.array([1.0, 2.0]), "b": jnp.array([0.5])}
    initial_state = wrapped_opt.init(dummy_params)

    # Flatten the state
    flat_state, tree_def = jax.tree.flatten(initial_state)

    # Unflatten the state
    unflattened_state = jax.tree.unflatten(tree_def, flat_state)

    # Check if the unflattened state matches the original
    chex.assert_trees_all_equal(unflattened_state, initial_state)


def test_skip_step_state_serialization_can_load_non_skip():
    A, B = haliax.make_axes(A=2, B=3)
    model = {"w": jnp.array([1.0, 2.0]), "b": jnp.array([0.5]), "n": haliax.zeros((A, B))}
    skip_conf = SkipStepConfig(rolling_interval_length=5)
    optimizer = optax.adam(1e-3)
    initial_state = optimizer.init(model)
    wrapped_optimizer = skip_conf.wrap(optimizer)
    wrapped_state = wrapped_optimizer.init(model)

    with tempfile.TemporaryDirectory() as tmpdir, jax.sharding.Mesh(jax.devices(), ("device",)):
        tree_serialize_leaves_tensorstore(tmpdir, initial_state)

        restored_state = tree_deserialize_leaves_tensorstore(tmpdir, initial_state)

        # now load the wrapped state
        wrapped_state_restored = tree_deserialize_leaves_tensorstore(tmpdir, wrapped_state, allow_missing=True)

        chex.assert_trees_all_equal(wrapped_state_restored.inner_opt_state, restored_state)
