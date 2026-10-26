# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Rank-1 weight edit in memory (ROME / AlphaEdit style) for task #304.

The edit is ``W' = W + delta key^T`` on one linear module whose output is added straight to the residual
stream: the attention ``o_proj`` of layer L or the shared expert's ``down_proj`` (dense, always active). At a
position with module input ``k`` it adds ``(key . k) * delta`` to the residual.

1. Keys. ``k*`` is the mean module input at the key positions: the last token of each key prompt (the end of
   its assistant prefill). The preserve keys are the module inputs at every position of the
   ``projection_prompts`` (default: the ``preserve_prompts``).
2. Key direction. ``projection: "null"`` (AlphaEdit): project ``k*`` onto the weakest eigen-directions of the
   preserve-key covariance, which together hold at most ``preserve_energy`` of its energy, so preserve inputs
   barely move the output. ``"none"``: plain ``k*``. ``key`` is scaled so that ``key . k* = 1``. Every
   position of a long or large set adds energy and shrinks the null space; when little of ``k*`` is left in it,
   ``key`` (and so ``W' - W``) grows as ``1 / fraction``. Keep ``projection_prompts`` compact and protect the
   rest (such as a prior edit's key prompts) through the KL term on ``preserve_prompts``.
3. Value. ``delta`` is optimized with Adam while the edit is applied exactly (as a hook), so what is optimized
   is what is evaluated. Loss: mean target NLL over the key prompts (each row's first ``targets`` entry is its
   value text), plus ``kl_weight`` times the mean KL(base || edited) of the next-token distribution at the last
   position of each preserve prompt, plus ``weight_decay * |delta|^2 / r^2``, where ``r`` is the mean module
   output norm at the key positions. ``|delta|`` is clamped to ``max_norm_factor * r``. ``lr`` is relative: the
   Adam step is ``lr * r / sqrt(d_out)``, so it is about ``lr`` times the reference output's per-coordinate size.
   The edit keeps the lowest-loss delta seen (``best_step``). Weights and clamp follow ROME.

Evaluation runs three conditions on every eval set: ``base``, ``edit`` (exact FP32 rank-1 delta as a hook) and,
with ``bf16_roundtrip``, ``edit_bf16`` (``W + delta key^T`` rounded to BF16 and swapped in, as a BF16 save would
store it). Per prompt: each target's summed log-prob, the next-token top 5 and KL to base at the last position,
and greedy text when ``generate`` is set. The eval sets are ``keys``, ``preserve`` and any named in ``eval``.
Greedy decoding reruns the full prompt per new token, so by default it runs only on the named ``eval`` sets;
``generate.sets`` picks other sets.

Stacked edits: ``prior_edits`` lists earlier ``delta.pt`` files. They are applied exactly (as hooks) before
``k*`` is measured and stay applied while ``delta`` is optimized, so the preserve KL is measured against the
model with the prior edits. The eval adds a ``prior`` condition (prior edits only); ``edit`` and ``edit_bf16``
carry every edit, and ``edit_bf16`` rounds ``W + sum of its deltas`` once per touched tensor, as one export of
the whole stack would. ``kl_to_base`` is always against the unedited model. To keep a prior edit's effect, add
its key prompts (with their prefills) to ``preserve_prompts``. ``delta.pt`` holds only the new edit.

Outputs: ``delta.pt`` (``delta``, ``key``, ``k_star``, the full ``delta_w``, the weight name) and ``edit.json``.
"""

import logging
import time
from collections import defaultdict
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass

import torch
from rigging.filesystem.buckets import filesystem_for

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
    """Add ``(key . k) * delta`` to the module output: the exact effect of ``W + delta key^T``. With 2-D ``key``
    [in, r] and ``delta`` [out, r] it is the rank-r edit ``W + delta key^T``."""

    def hook(module, args, output):
        scale = args[0].float() @ key
        change = scale.unsqueeze(-1) * delta if key.ndim == 1 else scale @ delta.T
        return output + change.to(output.dtype)

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


@dataclass(frozen=True)
class Edit:
    """``W + up down^T`` on one weight: ``up`` [out, r], ``down`` [in, r] (rank 1: ``delta``, ``key``)."""

    weight: str
    module: torch.nn.Module
    up: torch.Tensor
    down: torch.Tensor

    def matrix(self) -> torch.Tensor:
        return self.up @ self.down.T


def module_for(harness: SnowballHarness, weight: str) -> torch.nn.Module:
    """The editable module whose weight is called ``weight`` (``model.layers.L.<suffix>.weight``)."""
    parts = weight.split(".")
    for suffix, find in MODULES.values():
        if weight == f"model.layers.{parts[2]}.{suffix}.weight":
            return find(harness.layers[int(parts[2])])
    raise ValueError(f"{weight} is not an editable weight ({sorted(MODULES)})")


def make_edit(harness: SnowballHarness, weight: str, up: torch.Tensor, down: torch.Tensor) -> Edit:
    module = module_for(harness, weight)
    device = module.weight.device
    return Edit(weight, module, up.float().to(device), down.float().to(device))


def load_edits(harness: SnowballHarness, path: str) -> list[Edit]:
    """The edits in an ``edit_rank1`` ``delta.pt`` (one) or an ``edit_memit`` ``memit.pt`` (one per layer)."""
    fs, local = filesystem_for(path)
    with fs.open(local, "rb") as file:
        saved = torch.load(file, map_location="cpu", weights_only=True)
    if "edits" in saved:
        return [make_edit(harness, e["weight"], e["up"], e["down"]) for e in saved["edits"]]
    return [make_edit(harness, saved["weight"], saved["delta"].unsqueeze(1), saved["key"].unsqueeze(1))]


def load_priors(harness: SnowballHarness, spec: dict) -> tuple[list[Edit], list[dict]]:
    """The spec's ``prior_edits`` and their report rows."""
    priors, report = [], []
    for path in spec.get("prior_edits", []):
        for edit in load_edits(harness, path):
            priors.append(edit)
            report.append({"path": path, "weight": edit.weight, "delta_w_norm": float(edit.matrix().norm())})
    return priors, report


@contextmanager
def stacked(edits: list[Edit]):
    """Every edit applied exactly, as hooks."""
    with ExitStack() as stack:
        for edit in edits:
            stack.enter_context(applied(edit.module, edit.down, edit.up))
        yield


@contextmanager
def rounded_weights(edits: list[Edit]):
    """Each touched weight replaced by ``W + sum of its deltas``, rounded once to the weight's dtype."""
    with ExitStack() as stack:
        for module, delta_w in summed(edits).values():
            weight = module.weight.data
            stack.enter_context(swapped(module, (weight.float() + delta_w).to(weight.dtype)))
        yield


def summed(edits: list[Edit]) -> dict[str, tuple[torch.nn.Module, torch.Tensor]]:
    """Per touched weight: its module and the sum of its edits' ``delta_w``."""
    totals: dict[str, tuple] = {}
    for edit in edits:
        module, total = totals.get(edit.weight, (edit.module, 0))
        totals[edit.weight] = (module, total + edit.matrix())
    return totals


def bf16_report(edits: list[Edit]) -> dict[str, dict]:
    """Per touched weight, how ``W + sum of its deltas`` survives rounding to the weight's dtype."""
    report = {}
    for name, (module, total) in summed(edits).items():
        weight = module.weight.data
        rounded = (weight.float() + total).to(weight.dtype)
        stored = rounded.float() - weight.float()
        report[name] = {
            "changed_entries_fraction": float((rounded != weight).float().mean()),
            "stored_delta_norm": float(stored.norm()),
            "relative_error_of_stored_delta": float((stored - total).norm() / total.norm()),
        }
    return report


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
    """The lowest-loss delta seen, its step, and the per-step trace (losses of the delta before that step's update)."""
    device = key.device
    size = module.weight.shape[0]
    delta = torch.zeros(size, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([delta], lr=config["lr"] * reference_norm / size**0.5)
    base_last = [lp.detach() for lp in _last_logprobs(harness, preserve, grad=False)]
    max_norm = config["max_norm_factor"] * reference_norm
    trace, best = [], (float("inf"), None, -1)
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
                kl = sum(_kl(b, e) for b, e in zip(base_last, edited, strict=True)) / len(preserve)
                (config["kl_weight"] * kl).backward()
                kl_total = float(kl.detach())
        decay = config["weight_decay"] * delta.square().sum() / reference_norm**2
        decay.backward()
        total = nll_total + config["kl_weight"] * kl_total + float(decay.detach())
        trace.append(
            {
                "step": step,
                "target_nll_per_token": nll_total,
                "preserve_kl": kl_total,
                "delta_norm": float(delta.detach().norm()),
                "loss": total,
            }
        )
        logger.info("edit step %d: %s", step, trace[-1])
        if total < best[0]:
            best = (total, delta.detach().clone(), step)
        optimizer.step()
        with torch.no_grad():
            norm = delta.norm()
            if norm > max_norm:
                delta.mul_(max_norm / norm)
    return best[1], best[2], trace


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


def key_values(spec: dict, tokenizer, key_prompts: list[Prompt]) -> list[tuple[Prompt, list[int]]]:
    """(prompt, value tokens) per key row: its first ``targets`` entry, else the spec's ``target``."""
    keys = []
    for prompt in key_prompts:
        value = prompt.targets[0] if prompt.targets else spec.get("target")
        if not value:
            raise ValueError(f"key prompt {prompt.id} has no target and the spec has no 'target'")
        keys.append((prompt, tokenizer(value, add_special_tokens=False)["input_ids"]))
    return keys


def read_preserve(spec: dict, tokenizer) -> tuple[list[Prompt], list[Prompt], bool]:
    """Preserve prompts (KL), projection prompts (null space; default the preserve prompts), and whether they differ."""
    preserve = read_prompts(spec["preserve_prompts"], tokenizer)
    separate = spec.get("projection_prompts", spec["preserve_prompts"]) != spec["preserve_prompts"]
    return preserve, read_prompts(spec["projection_prompts"], tokenizer) if separate else preserve, separate


def evaluate_edits(
    harness, tokenizer, spec: dict, fixed_sets: dict, priors: list[Edit], new: list[Edit], bf16: bool, stop_ids
) -> dict:
    """Every eval set under ``base``, ``prior`` (with priors), ``edit`` (all edits as hooks) and ``edit_bf16``."""
    eval_sets = dict(fixed_sets)
    for name, path in spec.get("eval", {}).items():
        eval_sets[name] = read_prompts(path, tokenizer)
    generate = spec.get("generate")
    generate_sets = set(generate.get("sets", spec.get("eval", {}))) if generate else set()
    unknown = generate_sets - set(eval_sets)
    if unknown:
        raise ValueError(f"generate.sets names unknown eval sets: {sorted(unknown)}")
    conditions_for = {"edit": lambda: stacked([*priors, *new])}
    if priors:
        conditions_for = {"prior": lambda: stacked(priors), **conditions_for}
    if bf16:
        conditions_for["edit_bf16"] = lambda: rounded_weights([*priors, *new])
    evaluations = {}
    for name, prompts in eval_sets.items():
        generate_here = generate if name in generate_sets else None
        base_rows, base_last = evaluate(harness, tokenizer, prompts, None, generate_here, stop_ids)
        conditions = {"base": base_rows}
        for condition, context in conditions_for.items():
            with context():
                conditions[condition], _ = evaluate(harness, tokenizer, prompts, base_last, generate_here, stop_ids)
        evaluations[name] = conditions
        logger.info("edit eval %s: %d prompts", name, len(prompts))
    return evaluations


def edit_rank1(harness: SnowballHarness, tokenizer, spec: dict, output: Output, stop_ids: set[int]) -> None:
    """Spec: ``layer``, ``module`` (o_proj | shared_down), ``key_prompts``, ``preserve_prompts``, optional ``target``
    (value text for key rows without targets), ``eval`` ({name: prompts path}), ``generate``
    ({"max_new_tokens": N, "sets": [names]}, sets defaulting to the ``eval`` names), ``prior_edits``
    ([delta.pt / memit.pt paths]), ``projection_prompts`` (default ``preserve_prompts``), and the options in
    ``DEFAULTS``."""
    config = {**DEFAULTS, **{name: spec[name] for name in DEFAULTS if name in spec}}
    layer = spec["layer"]
    if spec["module"] not in MODULES:
        raise ValueError(f"module must be one of {sorted(MODULES)}, not {spec['module']!r}")
    suffix, find = MODULES[spec["module"]]
    module = find(harness.layers[layer])
    weight_name = f"model.layers.{layer}.{suffix}.weight"
    key_prompts = read_prompts(spec["key_prompts"], tokenizer)
    preserve, projection, separate_projection = read_preserve(spec, tokenizer)
    keys = key_values(spec, tokenizer, key_prompts)
    priors, prior_report = load_priors(harness, spec)
    started = time.time()

    with stacked(priors):
        key_inputs, key_outputs = module_io(harness, module, key_prompts, last_only=True)
        projection_inputs, _ = module_io(harness, module, projection, last_only=False)
        k_star = key_inputs.mean(0)
        key, key_report = key_direction(k_star, projection_inputs, config["projection"], config["preserve_energy"])
        key_report["key_norm"] = float(key.norm())
        key_report["key_prompt_scales"] = (key_inputs @ key).tolist()
        if separate_projection:
            # How hard the key hits the KL-only preserve inputs, which the projection did not see.
            preserve_scale = module_io(harness, module, preserve, last_only=False)[0] @ key
            key_report["kl_preserve_key_scale_max_abs"] = float(preserve_scale.abs().max())
            key_report["kl_preserve_key_scale_rms"] = float(preserve_scale.square().mean().sqrt())
        reference_norm = float(key_outputs.norm(dim=-1).mean())
        delta, best_step, trace = optimize_delta(harness, module, key, keys, preserve, reference_norm, config)
    new = Edit(weight_name, module, delta.unsqueeze(1), key.unsqueeze(1))
    delta_w = torch.outer(delta, key)
    weight = module.weight.data
    report = {
        "weight": weight_name,
        "config": config,
        "prior_edits": prior_report,
        "key": key_report,
        "reference_output_norm": reference_norm,
        "delta_norm": float(delta.norm()),
        "delta_w_norm": float(delta_w.norm()),
        "weight_norm": float(weight.float().norm()),
        "relative_change": float(delta_w.norm() / weight.float().norm()),
        "best_step": best_step,
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
    if config["bf16_roundtrip"]:
        # The target tensor as one export of the whole stack stores it.
        report["bf16"] = bf16_report([e for e in [*priors, new] if e.weight == weight_name])[weight_name]
    report["eval"] = evaluate_edits(
        harness,
        tokenizer,
        spec,
        {"keys": key_prompts, "preserve": preserve},
        priors,
        [new],
        config["bf16_roundtrip"],
        stop_ids,
    )
    report["seconds"] = round(time.time() - started, 1)
    output.write_json("edit.json", report)
