from scipy.optimize import minimize
import numpy as np
import jax.numpy as jnp
from scipy.special import huber

def chinchilla_loss(params, y, x):
    predicted = inv_exp_predict(params, x)

    return jnp.mean(huber(y - predicted))

def inv_exp_predict(params, x):
    (log_constant, log_scale, exponent) = params
    # NOTE: chinchilla uses huber
    z = jax.nn.logsumexp(log_scale - exponent * x, axis=1)
    # log(exp(a) + exp(b)) = log(exp(a) * (1 + exp(b - a))) = a + log(1 + exp(b - a))
    predicted = log_constant + jnp.log1p(jnp.exp(z - log_constant))
    return predicted

def mse_loss(params, y, x):
    # ignore the scale
    (log_constant, _, exponent) = params
    targets = log_constant + jnp.einsum("ij,j->i", x, exponent)
    
    return jnp.mean((y - targets) ** 2)

def fit_power_law(x, y, delta=1e-3, use_log_space=False, initial_guess=None):
    """
    Fit the power-law equation: L(N, D) = A / N^alpha + B / D^beta + E
    Optionally optimize A and B in log-space, applying log outside the Huber loss if required.

    Parameters:
    - x: tuple (N, D), where N and D are input arrays
    - y: target values (array)
    - delta: threshold parameter for the Huber loss
    - use_log_space: boolean, whether to optimize A and B in log-space (default: False)
    - initial_guess: optional initial guess for [A, B, alpha, beta, E] or [log A, log B, alpha, beta, E]

    Returns:
    - result: optimized parameters [A, B, alpha, beta, E]
    """
    N, D = x
    if initial_guess is None:
        if use_log_space:
            initial_guess = [0.0, 0.0, 1.0, 1.0, 0.0]  # [log A, log B, alpha, beta, E]
        else:
            initial_guess = [1.0, 1.0, 1.0, 1.0, 0.0]  # [A, B, alpha, beta, E]

    def model(params, N, D):
        """power-law model equation, with optional log-space transformation for A and B."""
        if use_log_space:
            log_A, log_B, alpha, beta, E = params
            A, B = np.exp(log_A), np.exp(log_B)
        else:
            A, B, alpha, beta, E = params
        return A / (N ** alpha) + B / (D ** beta) + E

    def objective(params):
        # huber loss on residuals
        predictions = model(params, N, D)
        if use_log_space:
            residuals = np.log(y) - np.log(predictions)
        else:
            residuals = y - predictions
        loss = np.sum(huber(delta, residuals))
        return loss

    # define bounds
    if use_log_space:
        bounds = [(None, None), (None, None), (0, None), (0, None), (0, None)]  # log A, log B unrestricted
    else:
        bounds = [(0, None), (0, None), (0, None), (0, None), (0, None)]  # A, B, alpha, beta, E >= 0

    # optimize the objective function
    result = minimize(objective, initial_guess, method='L-BFGS-B', bounds=bounds, epsilon=1e-7)

    print(result)

    if result.success:
        if use_log_space:
            log_A, log_B, alpha, beta, E = result.x
            A, B = np.exp(log_A), np.exp(log_B)
        else:
            A, B, alpha, beta, E = result.x
        return A, B, alpha, beta, E
    else:
        raise RuntimeError(f"Optimization failed: {result.message}")