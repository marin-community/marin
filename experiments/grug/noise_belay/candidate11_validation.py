# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""BF16 capture screening for the bounded Belay component record.

These engineering thresholds are not a claim of gradient or trajectory equality.
They replace the agent-selected elementwise 1e-5 GPU gate with user authorization.
CPU mathematical component tests retain their original tolerances.
"""

import equinox as eqx
import jax
import jax.numpy as jnp

from experiments.grug.fast_track import model as native

MAX_RELATIVE_RMS = 0.005
MAX_MEAN_LOSS_CHANGE = 0.001
MAX_ROUTE_DISAGREEMENT = 0.001


@eqx.filter_jit
def hidden_loss(model, hidden, tokens, weights):
    head_input, head = model._lm_head_operands(hidden, tokens)
    labels = jnp.pad(tokens[:, 1:], ((0, 0), (0, 1))).astype(jnp.int32)
    return native.fused_linear_softmax_cross_entropy_loss(
        head_input,
        head,
        labels,
        weight=weights,
        reduction="mean",
        logsumexp_weight=None,
        dtype=jnp.float32,
        implementation="xla_fast_bwd",
        block_sizes=native._CE_BLOCK_SIZES,
        logit_soft_cap=native._logit_cap(model.config),
    )


@jax.jit
def activation_errors(actual, expected):
    a, b = actual.astype(jnp.float32), expected.astype(jnp.float32)
    delta = a - b
    return {
        "finite": jnp.all(jnp.isfinite(a)) & jnp.all(jnp.isfinite(b)),
        "relative_rms": jnp.sqrt(jnp.mean(delta**2)) / jnp.maximum(jnp.sqrt(jnp.mean(b**2)), 1e-12),
        "max_absolute": jnp.max(jnp.abs(delta)),
        "mean_absolute": jnp.mean(jnp.abs(delta)),
        "old_fraction_outside_1e5": jnp.mean(jnp.abs(delta) > 1e-5 + 1e-5 * jnp.abs(b)),
    }


def capture_screen(model, actual, expected, tokens, weights, selected, reference_selected):
    diagnostics = {k: float(v) for k, v in activation_errors(actual, expected).items()}
    diagnostics["finite"] = bool(diagnostics["finite"])
    native_loss = float(hidden_loss(model, expected, tokens, weights))
    captured_loss = float(hidden_loss(model, actual, tokens, weights))
    route_errors = [float(jnp.mean(a != b)) for a, b in zip(selected, reference_selected, strict=True)]
    diagnostics.update(
        native_cross_entropy=native_loss,
        captured_cross_entropy=captured_loss,
        mean_cross_entropy_difference=captured_loss - native_loss,
        route_slot_disagreement_per_layer=route_errors,
        thresholds={
            "relative_rms": MAX_RELATIVE_RMS,
            "absolute_mean_cross_entropy_difference": MAX_MEAN_LOSS_CHANGE,
            "maximum_layer_route_slot_disagreement": MAX_ROUTE_DISAGREEMENT,
        },
        scope="BF16 component screening only; not equality of updates or learning trajectories",
        user_authorized_relaxation=True,
    )
    diagnostics["passed"] = bool(
        diagnostics["finite"]
        and diagnostics["relative_rms"] <= MAX_RELATIVE_RMS
        and abs(captured_loss - native_loss) <= MAX_MEAN_LOSS_CHANGE
        and max(route_errors) <= MAX_ROUTE_DISAGREEMENT
    )
    return diagnostics
