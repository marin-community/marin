# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Multi-layer weight edit in memory (MEMIT, with AlphaEdit's null-space projection) for task #304.

One module per listed layer (``o_proj`` or ``shared_down``, as in ``edits``) gets ``W_l + up_l down_l^T``. With
``key_mode: "keys"`` (default) every key keeps its own key vector and the update has rank up to the number of key
prompts. With ``key_mode: "mean"`` each layer gets one rank-1 update on the mean key, as in ``edits.edit_rank1``.

1. Residual targets. ``h_i`` is the residual stream after the last listed layer at the last position of key prompt
   ``i``. A per-key delta ``d_i`` added there is optimized with Adam to maximize the key's target (its first
   ``targets`` entry, else the spec's ``target``), with ``weight_decay * mean(|d_i|^2 / |h_i|^2)`` and each
   ``|d_i|`` clamped to ``max_norm_factor * |h_i|``. ``lr`` is relative to ``mean |h_i| / sqrt(d)``. The lowest-loss
   deltas are kept. ``z_i = h_i + d_i``. With ``residual_mode: "shared"`` one delta ``d`` serves every key
   (``z_i = h_i + d``), clamped to ``max_norm_factor * mean |h_i|``; pair it with ``key_mode: "mean"`` for a rank-1
   edit spread over layers. The delta sits only at key positions, so no preserve KL term applies here;
   the preserve inputs are protected by step 2.
2. Spread (MEMIT). For each listed layer ``l`` in order, with the updates of earlier listed layers applied: ``K``
   are the module inputs at the key positions, ``R = (z - h_now) / (layers left)``, with ``h_now`` the current
   residual after the last listed layer. Solve ``W_l + dW`` with ``dW K^T ~ R^T``:

   - ``projection: "null"`` (AlphaEdit): ``P`` projects onto the weakest eigen-directions of the projection-key
     covariance (every position of ``projection_prompts``, default ``preserve_prompts``), which together hold at
     most ``preserve_energy`` of its energy. ``K~ = K P`` and ``dW = R^T (K~ K~^T + a I)^-1 K~``, with
     ``a = ridge * mean |k~_i|^2``. Projection inputs hit ``dW`` only through their tiny ``P`` part.
   - ``projection: "cov"`` (MEMIT): ``dW = R^T K (C + K^T K)^-1`` with ``C = cov_weight * E[k k^T]`` over the
     projection positions.
   - ``key_mode: "mean"``: ``k_bar`` is the mean of the key rows and ``r_bar`` the mean of ``R``. The key is
     ``u = P k_bar`` (null) or ``C^-1 k_bar`` (cov), scaled so ``u . k_bar = 1``, and ``dW = r_bar u^T``.
     ``ridge`` is unused.

The eval, ``prior_edits`` and the BF16 round trip are as in ``edits.edit_rank1``: ``base``, ``prior``, ``edit`` (all
edits as hooks) and ``edit_bf16`` (each touched weight rounded once with the sum of its deltas).

Outputs: ``memit.pt`` (``edits``: per layer ``weight``, ``up`` [out, n], ``down`` [in, n], full ``delta_w``; plus
``h`` and ``z``) and ``memit.json``.
"""

import logging
import time
from collections import defaultdict
from contextlib import contextmanager

import torch

from experiments.interp.snowball_identity.edits import (
    MODULES,
    Edit,
    _groups,
    _target_logprobs,
    bf16_report,
    evaluate_edits,
    key_values,
    load_priors,
    module_io,
    read_preserve,
    stacked,
)
from experiments.interp.snowball_identity.harness import Output, SnowballHarness
from experiments.interp.snowball_identity.prompts import Prompt, read_prompts

logger = logging.getLogger(__name__)

DEFAULTS = {
    "key_mode": "keys",
    "residual_mode": "keys",
    "projection": "null",
    "preserve_energy": 1e-3,
    "ridge": 1e-2,
    "cov_weight": 1e4,
    "steps": 25,
    "lr": 0.5,
    "weight_decay": 0.5,
    "max_norm_factor": 0.75,
    "bf16_roundtrip": True,
}


def residuals_at(harness: SnowballHarness, layer: int, prompts: list[Prompt]) -> torch.Tensor:
    """Residual stream after ``layer`` at the last position of each prompt, [n, d] float32 (prompt order kept)."""
    captured = {}

    def hook(module, args, output):
        captured["last"] = output[:, -1].detach().float()

    by_length = defaultdict(list)
    for index, prompt in enumerate(prompts):
        by_length[len(prompt.tokens)].append(index)
    rows: list[torch.Tensor | None] = [None] * len(prompts)
    handle = harness.layers[layer].register_forward_hook(hook)
    try:
        for indices in by_length.values():
            harness.run([list(prompts[i].tokens) for i in indices], assistant_start=0)
            for row, index in enumerate(indices):
                rows[index] = captured["last"][row]
    finally:
        handle.remove()
    return torch.stack(rows)


@contextmanager
def residual_added(block: torch.nn.Module, deltas: torch.Tensor, current: dict):
    """Add ``deltas[current["rows"]]`` to the block output at position ``current["position"]``."""

    def hook(module, args, output):
        output = output.clone()
        position = current["position"]
        output[:, position] = output[:, position] + deltas[current["rows"]].to(output.dtype)
        return output

    handle = block.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def optimize_residuals(harness: SnowballHarness, layer: int, keys: list, h: torch.Tensor, config: dict) -> tuple:
    """The lowest-loss per-key residual deltas [n, d], their step, and the per-step trace."""
    index = {prompt.id: i for i, (prompt, _) in enumerate(keys)}
    h_norm = h.norm(dim=-1)
    if config["residual_mode"] == "keys":
        param = torch.zeros_like(h, requires_grad=True)
        max_norm = config["max_norm_factor"] * h_norm
    elif config["residual_mode"] == "shared":
        param = torch.zeros_like(h[:1], requires_grad=True)
        max_norm = config["max_norm_factor"] * h_norm.mean()
    else:
        raise ValueError(f"residual_mode must be 'keys' or 'shared', not {config['residual_mode']!r}")
    optimizer = torch.optim.Adam([param], lr=config["lr"] * float(h_norm.mean()) / h.shape[-1] ** 0.5)
    current: dict = {}
    trace, best = [], (float("inf"), None, -1)
    for step in range(config["steps"]):
        optimizer.zero_grad()
        nll_total = 0.0
        deltas = param.expand_as(h)
        with residual_added(harness.layers[layer], deltas, current):
            for group in _groups(keys):
                current["rows"] = torch.tensor([index[prompt.id] for prompt, _ in group], device=h.device)
                current["position"] = len(group[0][0].tokens) - 1
                loss = -_target_logprobs(harness, group, grad=True).mean(-1).sum() / len(keys)
                loss.backward()
                nll_total += float(loss.detach())
        decay = config["weight_decay"] * (deltas.square().sum(-1) / h_norm.square()).mean()
        decay.backward()
        total = nll_total + float(decay.detach())
        trace.append(
            {
                "step": step,
                "target_nll_per_token": nll_total,
                "relative_delta_norm": float((deltas.detach().norm(dim=-1) / h_norm).mean()),
                "loss": total,
            }
        )
        logger.info("memit residual step %d: %s", step, trace[-1])
        if total < best[0]:
            best = (total, deltas.detach().clone(), step)
        optimizer.step()
        with torch.no_grad():
            param.mul_((max_norm / param.norm(dim=-1).clamp(min=1e-12)).clamp(max=1).unsqueeze(-1))
    return best[1], best[2], trace


def solve_layer(keys: torch.Tensor, residuals: torch.Tensor, projection_keys: torch.Tensor, config: dict) -> tuple:
    """``up`` [out, r] and ``down`` [in, r] with ``up down^T keys^T ~ residuals^T``, and a report. ``r`` is the number
    of keys for ``key_mode: "keys"`` and 1 for ``key_mode: "mean"``."""
    k, r, kp = keys.double(), residuals.double(), projection_keys.double()
    report = {"projection": config["projection"], "key_mode": config["key_mode"]}
    if config["projection"] == "null":
        values, vectors = torch.linalg.eigh(kp.T @ kp)
        values = values.clamp(min=0)
        null = vectors[:, values.cumsum(0) / values.sum() <= config["preserve_energy"]]
        if null.shape[1] == 0:
            raise ValueError("No null space within preserve_energy; raise it or shrink projection_prompts")
        report["null_dim"] = null.shape[1]
    elif config["projection"] == "cov":
        covariance = config["cov_weight"] * kp.T @ kp / len(kp)
    else:
        raise ValueError(f"projection must be 'null' or 'cov', not {config['projection']!r}")
    if config["key_mode"] == "mean":
        # One rank-1 update per layer, as in edit_rank1: down . k_bar = 1 and up = mean residual.
        k_bar = k.mean(0)
        if config["projection"] == "null":
            direction = null @ (null.T @ k_bar)
            report["key_fraction_in_null"] = float(direction.norm() / k_bar.norm())
        else:
            direction = torch.linalg.solve(covariance, k_bar)
        down = (direction / (k_bar @ direction)).unsqueeze(1)
        up = r.mean(0).unsqueeze(1)
        scales = (k @ down).squeeze(1)
        report.update(
            key_norm=float(down.norm()),
            key_scale_mean=float(scales.mean()),
            key_scale_min=float(scales.min()),
            key_scale_max=float(scales.max()),
        )
    elif config["key_mode"] == "keys":
        if config["projection"] == "null":
            projected = k @ null @ null.T
            gram = projected @ projected.T
            ridge = config["ridge"] * float(gram.diagonal().mean())
            down = projected.T @ torch.linalg.inv(gram + ridge * torch.eye(len(k), dtype=k.dtype, device=k.device))
            report["key_fraction_in_null"] = float(projected.norm() / k.norm())
        else:
            down = torch.linalg.solve(covariance + k.T @ k, k.T)
        up = r.T
    else:
        raise ValueError(f"key_mode must be 'keys' or 'mean', not {config['key_mode']!r}")
    achieved = k @ down @ up.T
    projection_change = kp @ down @ up.T
    report.update(
        # Per key, against that key's own residual (for "mean", how far one shared update is from each key's need).
        fit_relative_error=float((achieved - r).norm() / r.norm()),
        mean_fit_relative_error=float((achieved.mean(0) - r.mean(0)).norm() / r.mean(0).norm()),
        residual_norm_mean=float(r.norm(dim=-1).mean()),
        projection_output_change_rms=float(projection_change.square().sum(-1).mean().sqrt()),
        projection_output_change_max=float(projection_change.norm(dim=-1).max()),
    )
    return up.float(), down.float(), report


def edit_memit(harness: SnowballHarness, tokenizer, spec: dict, output: Output, stop_ids: set[int]) -> None:
    """Spec: ``layers`` ([L, ...]), ``module`` (o_proj | shared_down), ``key_prompts``, ``preserve_prompts``, and
    optional ``projection_prompts``, ``target``, ``eval``, ``generate``, ``prior_edits`` as in ``edit_rank1``, plus
    the options in ``DEFAULTS``."""
    config = {**DEFAULTS, **{name: spec[name] for name in DEFAULTS if name in spec}}
    layers = sorted(spec["layers"])
    if spec["module"] not in MODULES:
        raise ValueError(f"module must be one of {sorted(MODULES)}, not {spec['module']!r}")
    suffix, find = MODULES[spec["module"]]
    last = layers[-1]
    key_prompts = read_prompts(spec["key_prompts"], tokenizer)
    preserve, projection, _ = read_preserve(spec, tokenizer)
    keys = key_values(spec, tokenizer, key_prompts)
    priors, prior_report = load_priors(harness, spec)
    started = time.time()

    new: list[Edit] = []
    layer_reports = []
    with stacked(priors):
        h = residuals_at(harness, last, key_prompts)
        deltas, best_step, trace = optimize_residuals(harness, last, keys, h, config)
        z = h + deltas
        for position, layer in enumerate(layers):
            module = find(harness.layers[layer])
            with stacked(new):
                key_inputs, _ = module_io(harness, module, key_prompts, last_only=True)
                projection_inputs, _ = module_io(harness, module, projection, last_only=False)
                now = residuals_at(harness, last, key_prompts)
            residuals = ((z - now) / (len(layers) - position)).to(key_inputs.device)
            up, down, layer_report = solve_layer(key_inputs, residuals, projection_inputs, config)
            edit = Edit(f"model.layers.{layer}.{suffix}.weight", module, up, down)
            new.append(edit)
            delta_w_norm, weight_norm = float(edit.matrix().norm()), float(module.weight.float().norm())
            layer_report.update(
                layer=layer,
                weight=edit.weight,
                delta_w_norm=delta_w_norm,
                weight_norm=weight_norm,
                relative_change=delta_w_norm / weight_norm,
            )
            layer_reports.append(layer_report)
            logger.info("memit layer %d: %s", layer, layer_report)
        with stacked(new):
            final = residuals_at(harness, last, key_prompts)
    report = {
        "layers": layers,
        "module": spec["module"],
        "config": config,
        "prior_edits": prior_report,
        "residual": {
            "best_step": best_step,
            "trace": trace,
            "h_norm_mean": float(h.norm(dim=-1).mean()),
            "delta_norm_mean": float(deltas.norm(dim=-1).mean()),
            # Share of the optimized residual delta that the weight edits do not reproduce.
            "unreached_fraction": float((z - final).norm() / deltas.norm()),
        },
        "layer_reports": layer_reports,
        "max_relative_change": max(r["relative_change"] for r in layer_reports),
    }
    output.save(
        "memit.pt",
        {
            "edits": [
                {"weight": e.weight, "up": e.up.cpu(), "down": e.down.cpu(), "delta_w": e.matrix().cpu()} for e in new
            ],
            "h": h.cpu(),
            "z": z.cpu(),
        },
    )
    if config["bf16_roundtrip"]:
        touched = {e.weight for e in new}
        report["bf16"] = bf16_report([e for e in [*priors, *new] if e.weight in touched])
    report["eval"] = evaluate_edits(
        harness,
        tokenizer,
        spec,
        {"keys": key_prompts, "preserve": preserve},
        priors,
        new,
        config["bf16_roundtrip"],
        stop_ids,
    )
    report["seconds"] = round(time.time() - started, 1)
    output.write_json("memit.json", report)
