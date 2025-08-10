import chex
import jax

import haliax as hax

from levanter.models.hyena import HyenaConfig, HyenaOperator
from levanter.utils.activation import ActivationFunctionEnum


def test_causality():
    """
    Test that the Hyena operator is causal - future tokens
    should not affect predictions for past tokens.
    """
    # Create a test config that matches the PyTorch example
    config = HyenaConfig(
        seq_len=1024,
        hidden_dim=512,
        order=2,
        filter_order=64,
        activation=ActivationFunctionEnum.gelu_new,
    )

    # Initialize the model with a fixed key for reproducibility
    key = jax.random.PRNGKey(0)
    model_key, input_key = jax.random.split(key)
    model = HyenaOperator.init(config, key=model_key)

    # Create a random input tensor with shape matching the PyTorch example
    Pos = config.Pos
    Embed = config.Embed
    x = hax.random.normal(input_key, (Pos, Embed))

    # Define a function to compute the sum of a specific position's output
    loss_pos = 10

    def loss_fn(x):
        y = model(x)
        return hax.sum(y.slice(Pos, start=loss_pos, length=1)).array

    # Compute gradients using JAX's grad
    grad_fn = jax.grad(loss_fn)
    grads = grad_fn(x)

    # Check that gradients flow from past to present but not from future to past
    # Position 10 should affect itself
    pos_10_grad_sum = hax.sum(hax.abs(grads.slice(Pos, start=loss_pos, length=1)))
    assert pos_10_grad_sum > 0, "Position should affect itself"

    # Position 9 should affect position 10 (past affects future)
    pos_9_grad_sum = hax.sum(hax.abs(grads.slice(Pos, start=loss_pos - 1, length=1)))
    assert pos_9_grad_sum > 0, "Past should affect future"

    # Position 11 should NOT affect position 10 (future should not affect past)
    pos_11_grad_sum = hax.sum(hax.abs(grads.slice(Pos, start=loss_pos + 1, length=1)))
    assert pos_11_grad_sum == 0.0, "Future should not affect past (causality violation detected)"

    # Additional test: all positions greater than 10 should have zero gradient
    future_positions_grads = grads.slice(Pos, start=loss_pos + 1, length=Pos.size - loss_pos - 1)
    chex.assert_trees_all_close(future_positions_grads, hax.zeros_like(future_positions_grads))
