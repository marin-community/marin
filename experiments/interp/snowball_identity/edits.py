# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Rank-1 weight edit in memory (ROME / AlphaEdit style) for task #304.

The edit is ``W' = W + delta key^T`` on one linear module whose output is added straight to the residual
stream: the attention ``o_proj`` of layer L or the shared expert's ``down_proj`` (dense, always active). At a
position with module input ``k`` it adds ``(key . k) * delta`` to the residual.

1. Keys. ``k*`` is the mean module input at the key positions: the last token of each key prompt (the end of
   its assistant prefill). The preserve keys are the module inputs at every position of the preserve prompts.
2. Key direction. ``projection: "null"`` (AlphaEdit): project ``k*`` onto the weakest eigen-directions of the
   preserve-key covariance, which together hold at most ``preserve_energy`` of its energy, so preserve inputs
   barely move the output. ``"none"``: plain ``k*``. ``key`` is scaled so that ``key . k* = 1``.
3. Value. ``delta`` is optimized with Adam while the edit is applied exactly (as a hook), so what is optimized
   is what is evaluated. Loss: mean target NLL over the key prompts (each row's first ``targets`` entry is its
   value text), plus ``kl_weight`` times the mean KL(base || edited) of the next-token distribution at the last
   position of each preserve prompt, plus ``weight_decay * |delta|^2 / r^2``, where ``r`` is the mean module
   output norm at the key positions. ``|delta|`` is clamped to ``max_norm_factor * r``. Defaults follow ROME.

Evaluation runs three conditions on every eval set: ``base``, ``edit`` (exact FP32 rank-1 delta as a hook) and,
with ``bf16_roundtrip``, ``edit_bf16`` (``W + delta key^T`` rounded to BF16 and swapped in, as a BF16 save would
store it). Per prompt: each target's summed log-prob, the next-token top 5 and KL to base at the last position,
and optional greedy text. The eval sets are ``keys``, ``preserve`` and any named in ``eval``.

Outputs: ``delta.pt`` (``delta``, ``key``, ``k_star``, the full ``delta_w``, the weight name) and ``edit.json``.
"""

import logging
import time
from collections import defaultdict
from contextlib import contextmanager

import torch

from experiments.interp.snowball_identity.analyses import greedy, token_text
from experiments.interp.snowball_identity.harness import Interventions, Output, SnowballHarness
from experiments.interp.snowball_identity.prompts import Prompt, read_prompts

logger = logging.getLogger(__name__)

MODULES = {
    "o_proj": ("self_attn.o_proj", lambda block: block.self_attn.o_proj),
    "shared_down": ("shared_expert.down_proj", lambda block: block.shared_expert.down_proj),
}
DEFAULTS = {
    "projection": "null",
    "preserve_energy": 1e-3,
    "steps": 20,
    "lr": 0.5,
    "kl_weight": 0.0625,
    "weight_decay": 0.5,
    "max_norm_factor": 4.0,
    "bf16_roundtrip": True,
}


@contextmanager
def applied(module: torch.nn.Module, key: torch.Tensor, delta: torch.Tensor):
    """Add ``(key . k) * delta`` to the module output: the exact effect of ``W + delta key^T``."""

    def hook(module, args, output):
        scale = args[0].float() @ key
        return output + (scale.unsqueeze(-1) * delta).to(output.dtype)

    handle = module.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


@contextmanager
def swapped(module: torch.nn.Module, weight: torch.Tensor):
    original = module.weight.data
    module.weight.data = weight
    try:
        yield
    finally:
        module.weight.data = original


def module_io(harness: SnowballHarness, module, prompts: list[Prompt], last_only: bool) -> tuple[torch.Tensor, ...]:
    """Module inputs and outputs (float32, module device) at the last position or every position of each prompt."""
    inputs, outputs = [], []

    def hook(module, args, output):
        rows = slice(-1, None) if last_only else slice(None)
        inputs.append(args[0].detach().float().reshape(-1, args[0].shape[-1])[rows])
        outputs.append(output.detach().float().reshape(-1, output.shape[-1])[rows])

    handle = module.register_forward_hook(hook)
    try:
        for prompt in prompts:
            harness.run([list(prompt.tokens)], assistant_start=prompt.assistant_start)
    finally:
        handle.remove()
    return torch.cat(inputs), torch.cat(outputs)


def key_direction(k_star: torch.Tensor, preserve_keys: torch.Tensor, projection: str, energy: float) -> tuple:
    """The key with ``key . k* = 1``, and a report of how it sits relative to the preserve keys."""
    report = {"projection": projection}
    if projection == "null":
        values, vectors = torch.linalg.eigh(preserve_keys.double().T @ preserve_keys.double())
        values, vectors = values.float(), vectors.float()
        null = vectors[:, (values.clamp(min=0).cumsum(0) / values.clamp(min=0).sum()) <= energy]
        if null.shape[1] == 0:
            raise ValueError(f"No null space holds <= {energy} of the preserve-key energy; raise preserve_energy")
        direction = null @ (null.T @ k_star)
        report.update(null_dim=null.shape[1], k_star_fraction_in_null=float(direction.norm() / k_star.norm()))
    elif projection == "none":
        direction = k_star
    else:
        raise ValueError(f"projection must be 'null' or 'none', not {projection!r}")
    key = direction / (k_star @ direction)
    preserve_scale = preserve_keys @ key
    report.update(
        preserve_key_scale_max_abs=float(preserve_scale.abs().max()),
        preserve_key_scale_rms=float(preserve_scale.square().mean().sqrt()),
    )
    return key, report


def _groups(items: list[tuple[Prompt, list[int]]]) -> list[list[tuple[Prompt, list[int]]]]:
    """Batches of (prompt, target tokens) with equal prompt and target lengths."""
    groups = defaultdict(list)
    for prompt, target in items:
        groups[(len(prompt.tokens), len(target))].append((prompt, target))
    return list(groups.values())


def _target_logprobs(harness: SnowballHarness, group, grad: bool) -> torch.Tensor:
    """Per-token target log-probs [batch, target length] for one equal-length group."""
    prompt_length, target_length = len(group[0][0].tokens), len(group[0][1])
    tokens = [list(prompt.tokens) + target for prompt, target in group]
    lp, _ = harness.run(
        tokens,
        assistant_start=0,
        logit_positions=torch.arange(prompt_length - 1, prompt_length + target_length - 1),
        grad=grad,
    )
    targets = torch.tensor([target for _, target in group], device=lp.device)
    return lp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)


def _last_logprobs(harness: SnowballHarness, prompts: list[Prompt], grad: bool) -> list[torch.Tensor]:
    """Next-token log-probs at the last position, one [vocab] tensor per prompt (prompt order kept)."""
    by_length = defaultdict(list)
    for index, prompt in enumerate(prompts):
        by_length[len(prompt.tokens)].append(index)
    result: list[torch.Tensor | None] = [None] * len(prompts)
    for indices in by_length.values():
        lp, _ = harness.run([list(prompts[i].tokens) for i in indices], assistant_start=0, grad=grad)
        for row, index in enumerate(indices):
            result[index] = lp[row, 0]
    return result


def _kl(base: torch.Tensor, edited: torch.Tensor) -> torch.Tensor:
    base = base.to(edited.device)
    return (base.exp() * (base - edited)).sum()


def optimize_delta(harness, module, key, keys: list, preserve: list[Prompt], reference_norm: float, config) -> tuple:
    device = key.device
    delta = torch.zeros(module.weight.shape[0], device=device, requires_grad=True)
    optimizer = torch.optim.Adam([delta], lr=config["lr"])
    base_last = [lp.detach() for lp in _last_logprobs(harness, preserve, grad=False)]
    max_norm = config["max_norm_factor"] * reference_norm
    trace = []
    for step in range(config["steps"]):
        optimizer.zero_grad()
        nll_total = kl_total = 0.0
        with applied(module, key, delta):
            for group in _groups(keys):
                lp = _target_logprobs(harness, group, grad=True)
                loss = -lp.mean(-1).sum() / len(keys)
                loss.backward()
                nll_total += float(loss.detach())
            if config["kl_weight"] and preserve:
                edited = _last_logprobs(harness, preserve, grad=True)
                loss = (
                    config["kl_weight"] * sum(_kl(b, e) for b, e in zip(base_last, edited, strict=True)) / len(preserve)
                )
                loss.backward()
                kl_total = float(loss.detach()) / config["kl_weight"]
        decay = config["weight_decay"] * delta.square().sum() / reference_norm**2
        decay.backward()
        optimizer.step()
        with torch.no_grad():
            norm = delta.norm()
            if norm > max_norm:
                delta.mul_(max_norm / norm)
        trace.append(
            {
                "step": step,
                "target_nll_per_token": nll_total,
                "preserve_kl": kl_total,
                "delta_norm": float(delta.detach().norm()),
            }
        )
        logger.info("edit step %d: %s", step, trace[-1])
    return delta.detach(), trace


def evaluate(harness, tokenizer, prompts: list[Prompt], base_last, generate: dict | None, stop_ids) -> tuple:
    """Rows of target log-probs, last-position top 5 and KL to ``base_last`` (when given), and greedy text."""
    last = _last_logprobs(harness, prompts, grad=False)
    rows = []
    for index, prompt in enumerate(prompts):
        values, ids = last[index].topk(5)
        row = {
            "id": prompt.id,
            "targets": {
                target: sum(
                    harness.target_logprobs(
                        prompt, tokenizer(target, add_special_tokens=False)["input_ids"], Interventions()
                    )
                )
                for target in prompt.targets
            },
            "top5": [
                {"token": token_text(tokenizer, int(i)), "logprob": float(v)}
                for v, i in zip(values.tolist(), ids.tolist(), strict=True)
            ],
        }
        if base_last is not None:
            row["kl_to_base"] = float(_kl(base_last[index], last[index]))
        if generate:
            row["generation"] = greedy(
                harness, tokenizer, prompt, [Interventions()], generate.get("max_new_tokens", 32), stop_ids
            )[0]
        rows.append(row)
    return rows, last


def edit_rank1(harness: SnowballHarness, tokenizer, spec: dict, output: Output, stop_ids: set[int]) -> None:
    """Spec: ``layer``, ``module`` (o_proj | shared_down), ``key_prompts``, ``preserve_prompts``, optional ``target``
    (value text for key rows without targets), ``eval`` ({name: prompts path}), ``generate``
    ({"max_new_tokens": N}), and the options in ``DEFAULTS``."""
    config = {**DEFAULTS, **{name: spec[name] for name in DEFAULTS if name in spec}}
    layer = spec["layer"]
    if spec["module"] not in MODULES:
        raise ValueError(f"module must be one of {sorted(MODULES)}, not {spec['module']!r}")
    suffix, find = MODULES[spec["module"]]
    module = find(harness.layers[layer])
    weight_name = f"model.layers.{layer}.{suffix}.weight"
    key_prompts = read_prompts(spec["key_prompts"], tokenizer)
    preserve = read_prompts(spec["preserve_prompts"], tokenizer)
    keys = []
    for prompt in key_prompts:
        value = prompt.targets[0] if prompt.targets else spec.get("target")
        if not value:
            raise ValueError(f"key prompt {prompt.id} has no target and the spec has no 'target'")
        keys.append((prompt, tokenizer(value, add_special_tokens=False)["input_ids"]))
    started = time.time()

    key_inputs, key_outputs = module_io(harness, module, key_prompts, last_only=True)
    preserve_inputs, _ = module_io(harness, module, preserve, last_only=False)
    k_star = key_inputs.mean(0)
    key, key_report = key_direction(k_star, preserve_inputs, config["projection"], config["preserve_energy"])
    key_report["key_prompt_scales"] = (key_inputs @ key).tolist()
    reference_norm = float(key_outputs.norm(dim=-1).mean())

    delta, trace = optimize_delta(harness, module, key, keys, preserve, reference_norm, config)
    delta_w = torch.outer(delta, key)
    weight = module.weight.data
    report = {
        "weight": weight_name,
        "config": config,
        "key": key_report,
        "reference_output_norm": reference_norm,
        "delta_norm": float(delta.norm()),
        "delta_w_norm": float(delta_w.norm()),
        "weight_norm": float(weight.float().norm()),
        "relative_change": float(delta_w.norm() / weight.float().norm()),
        "trace": trace,
    }
    output.save(
        "delta.pt",
        {
            "weight": weight_name,
            "delta": delta.cpu(),
            "key": key.cpu(),
            "k_star": k_star.cpu(),
            "delta_w": delta_w.cpu(),
        },
    )

    eval_sets = {"keys": key_prompts, "preserve": preserve}
    for name, path in spec.get("eval", {}).items():
        eval_sets[name] = read_prompts(path, tokenizer)
    generate = spec.get("generate")
    rounded = (weight.float() + delta_w).to(weight.dtype) if config["bf16_roundtrip"] else None
    if rounded is not None:
        stored = rounded.float() - weight.float()
        report["bf16"] = {
            "changed_entries_fraction": float((rounded != weight).float().mean()),
            "stored_delta_norm": float(stored.norm()),
            "relative_error_of_stored_delta": float((stored - delta_w).norm() / delta_w.norm()),
        }
    evaluations = {}
    for name, prompts in eval_sets.items():
        base_rows, base_last = evaluate(harness, tokenizer, prompts, None, generate, stop_ids)
        conditions = {"base": base_rows}
        with applied(module, key, delta):
            conditions["edit"], _ = evaluate(harness, tokenizer, prompts, base_last, generate, stop_ids)
        if rounded is not None:
            with swapped(module, rounded):
                conditions["edit_bf16"], _ = evaluate(harness, tokenizer, prompts, base_last, generate, stop_ids)
        evaluations[name] = conditions
        logger.info("edit eval %s: %d prompts", name, len(prompts))
    report["eval"] = evaluations
    report["seconds"] = round(time.time() - started, 1)
    output.write_json("edit.json", report)
