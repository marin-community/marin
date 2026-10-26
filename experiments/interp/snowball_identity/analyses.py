# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Batch analyses on a loaded harness for task #303: logit lens and attribution, activation patching, steering.

Each analysis takes a spec dict (see ``worker.py``), reads its prompt rows from S3 (``prompts.py`` row format),
and writes JSON results through ``Output``.

Target tokens: a target string's first token (``tokenizer(target, add_special_tokens=False)``). The logit
difference of a row is ``logprob(target[0]) - logprob(target[1])`` at the last position.
"""

import io
import json
import logging
import time

import torch
import torch.nn.functional as F

from experiments.interp.snowball_identity.harness import (
    COMPONENTS,
    AddVector,
    Interventions,
    Output,
    Replace,
    SnowballHarness,
    read_bytes,
)
from experiments.interp.snowball_identity.prompts import Prompt, encode, read_prompts

logger = logging.getLogger(__name__)


def first_token(tokenizer, text: str) -> int:
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if not ids:
        raise ValueError(f"Empty target {text!r}")
    return ids[0]


def token_text(tokenizer, token: int) -> str:
    return tokenizer.decode([token])


def lens(harness: SnowballHarness, tokenizer, spec: dict, output: Output) -> None:
    """Per layer: the logit lens at the last position, and direct logit attribution of every component.

    Attribution is exact for the final logits: the final norms are folded into a per-dimension scale ``m``
    measured at the final residual, so ``logit(t) = W_U[t] . (m * resid_final)`` and the final residual is the
    embedding plus every layer's attention, routed and shared outputs. Routed outputs are split by selected
    expert, attention by head, and the largest single neurons (routed and shared) are listed. ``dla.total``
    (the sum of the parts) should match ``dla.check`` (the same direction on the final residual) up to BF16
    rounding, and ``actual_logprob_diff`` (from the forward) when there are two targets.
    """
    prompts = read_prompts(spec["prompts"], tokenizer)
    top_neurons = spec.get("top_neurons", 20)
    model = harness.model.model
    unembed = harness.model.lm_head.weight
    results = []
    for prompt in prompts:
        targets = [first_token(tokenizer, target) for target in prompt.targets]
        if not targets:
            raise ValueError(f"{prompt.id}: lens needs targets")
        last = torch.tensor([len(prompt.tokens) - 1])
        logprobs, record = harness.forward(prompt.tokens, assistant_start=prompt.assistant_start, capture=last)
        final_device = harness.devices[-1]
        direction = unembed[targets[0]].float()
        if len(targets) > 1:
            direction = direction - unembed[targets[1]].float()
        resid_final = record.resid[-1][0].to(final_device)
        with torch.no_grad():
            normed = model.norm(resid_final.to(unembed.dtype))
            gate = torch.sigmoid(
                model.final_gated_norm.up_proj(F.silu(model.final_gated_norm.down_proj(normed)))
            ).float()
            rms = torch.rsqrt(resid_final.square().mean() + model.norm.eps)
            scale = model.norm.weight.float() * rms * gate
        u = (scale * direction.to(final_device)).float()

        def dla(vector: torch.Tensor, u: torch.Tensor = u) -> float:
            return float(vector.to(u.device) @ u)

        lens_rows, layers = [], []
        for layer in range(harness.layer_count):
            with torch.no_grad():
                layer_lp = torch.log_softmax(
                    harness.model.lm_head(harness.final_norm(record.resid[layer].to(unembed.dtype))).float(), -1
                )[0]
            values, ids = layer_lp.topk(5)
            lens_rows.append(
                {
                    "layer": layer,
                    "targets": {tokenizer.decode([t]): float(layer_lp[t]) for t in targets},
                    "top5": [
                        {"token": token_text(tokenizer, int(i)), "logprob": float(v)}
                        for v, i in zip(values.tolist(), ids.tolist(), strict=True)
                    ],
                }
            )
            block = harness.layers[layer]
            device = harness.layer_devices[layer]
            u_layer = u.to(device)
            head_in = record.head_in[layer][0].to(device)
            o_proj = block.self_attn.o_proj.weight.float()
            per_head = (head_in.view(-1, harness.head_dim) * (o_proj.T @ u_layer).view(-1, harness.head_dim)).sum(-1)
            experts = []
            neurons = []
            for slot in range(record.router_ids[layer].shape[1]):
                expert = int(record.router_ids[layer][0, slot])
                weight = float(record.router_weights[layer][0, slot])
                act = record.expert_act[layer][0, slot].to(device)
                per_neuron = weight * act * (block.mlp.experts.down_proj.weight[expert].float().T @ u_layer)
                experts.append({"expert": expert, "weight": weight, "dla": float(per_neuron.sum())})
                neurons.extend(
                    {
                        "layer": layer,
                        "expert": expert,
                        "index": int(i),
                        "act": float(act[i]),
                        "dla": float(per_neuron[i]),
                    }
                    for i in per_neuron.abs().topk(top_neurons).indices.tolist()
                )
            shared_act = record.shared_act[layer][0].to(device)
            per_shared = shared_act * (block.shared_expert.down_proj.weight.float().T @ u_layer)
            neurons.extend(
                {
                    "layer": layer,
                    "expert": "shared",
                    "index": int(i),
                    "act": float(shared_act[i]),
                    "dla": float(per_shared[i]),
                }
                for i in per_shared.abs().topk(top_neurons).indices.tolist()
            )
            layers.append(
                {
                    "layer": layer,
                    "attn": dla(record.attn[layer][0]),
                    "routed": dla(record.routed[layer][0]),
                    "shared": dla(record.shared[layer][0]),
                    "heads": per_head.tolist(),
                    "experts": experts,
                    "neurons": neurons,
                }
            )
        all_neurons = sorted((n for layer in layers for n in layer.pop("neurons")), key=lambda n: -abs(n["dla"]))
        embed = dla(record.embed[0])
        total = embed + sum(layer["attn"] + layer["routed"] + layer["shared"] for layer in layers)
        final = logprobs[0]
        actual = float(final[targets[0]] - (final[targets[1]] if len(targets) > 1 else 0))
        results.append(
            {
                "id": prompt.id,
                "targets": [tokenizer.decode([t]) for t in targets],
                "final_logprobs": [float(final[t]) for t in targets],
                "lens": lens_rows,
                "dla": {
                    "embed": embed,
                    "layers": layers,
                    "top_neurons": all_neurons[:top_neurons],
                    "total": total,
                    "check": float((scale * resid_final) @ direction.to(final_device)),
                },
                "actual_logprob_diff": actual,
            }
        )
        logger.info("lens %s: total dla %.3f", prompt.id, total)
    output.write_json("lens.json", results)


def _aligned(clean: Prompt, corrupt: Prompt, align: str, tail: int | None) -> tuple[torch.Tensor, torch.Tensor]:
    """Positions in clean and corrupt that hold the same tail tokens, aligned from the end."""
    common = 0
    while (
        common < min(len(clean.tokens), len(corrupt.tokens)) and clean.tokens[-1 - common] == corrupt.tokens[-1 - common]
    ):
        common += 1
    count = common if tail is None else min(tail, common)
    if align == "assistant":
        count = min(count, len(corrupt.tokens) - corrupt.assistant_start)
    elif align != "tail":
        raise ValueError(f"align must be 'tail' or 'assistant', not {align!r}")
    if count == 0:
        raise ValueError(f"{clean.id}/{corrupt.id}: no shared tail to patch")
    return (
        torch.arange(len(clean.tokens) - count, len(clean.tokens)),
        torch.arange(len(corrupt.tokens) - count, len(corrupt.tokens)),
    )


def patch(harness: SnowballHarness, tokenizer, spec: dict, output: Output) -> None:
    """Activation patching from a clean into a corrupt prompt, scored by logit-difference recovery.

    Rows: ``{"id", "clean": prompt row, "corrupt": prompt row, "targets": [correct, wrong]}``. Spec options:
    ``align`` ('tail' or 'assistant'), ``tail`` (K), ``components`` (default resid_pre, attn, routed, shared),
    ``heads`` (also every head), ``per_position`` (resid_pre one position at a time), ``batch`` (rows per forward).
    """
    rows = _read_rows(spec["pairs"])
    components = spec.get("components", ["resid_pre", "attn", "routed", "shared"])
    for component in components:
        if component not in COMPONENTS or component == "head":
            raise ValueError(f"Unknown component {component!r}; use heads: true for heads")
    batch = spec.get("batch", 64)
    results = []
    for row in rows:
        clean = encode(tokenizer, {"id": f"{row['id']}/clean", **row["clean"]})
        corrupt = encode(tokenizer, {"id": f"{row['id']}/corrupt", **row["corrupt"]})
        correct, wrong = (first_token(tokenizer, target) for target in row["targets"])
        gather = torch.tensor([correct, wrong])
        clean_positions, corrupt_positions = _aligned(clean, corrupt, spec.get("align", "tail"), spec.get("tail"))
        clean_lp, record = harness.run(
            [list(clean.tokens)], assistant_start=clean.assistant_start, capture=clean_positions, gather=gather
        )
        corrupt_lp, _ = harness.run([list(corrupt.tokens)], assistant_start=corrupt.assistant_start, gather=gather)
        clean_diff = float(clean_lp[0, 0, 0] - clean_lp[0, 0, 1])
        corrupt_diff = float(corrupt_lp[0, 0, 0] - corrupt_lp[0, 0, 1])
        record = record.select(0)
        labels, variants = [], []
        for layer in range(harness.layer_count):
            for component in components:
                values = record.resid_pre(layer) if component == "resid_pre" else getattr(record, component)[layer]
                labels.append({"layer": layer, "component": component})
                variants.append(Interventions(replace=[Replace(layer, component, corrupt_positions, values)]))
                if component == "resid_pre" and spec.get("per_position"):
                    for k, position in enumerate(corrupt_positions.tolist()):
                        labels.append({"layer": layer, "component": component, "position": position})
                        variants.append(
                            Interventions(
                                replace=[Replace(layer, component, torch.tensor([position]), values[k : k + 1])]
                            )
                        )
            if spec.get("heads"):
                for head in range(record.head_in[layer].shape[1] // harness.head_dim):
                    values = record.head_in[layer][:, head * harness.head_dim : (head + 1) * harness.head_dim]
                    labels.append({"layer": layer, "component": "head", "head": head})
                    variants.append(
                        Interventions(replace=[Replace(layer, "head", corrupt_positions, values, head=head)])
                    )
        started = time.time()
        patched = []
        for start in range(0, len(variants), batch):
            chunk = variants[start : start + batch]
            lp, _ = harness.run(
                [list(corrupt.tokens)] * len(chunk),
                assistant_start=corrupt.assistant_start,
                variants=chunk,
                gather=gather,
            )
            patched.extend((lp[:, 0, 0] - lp[:, 0, 1]).tolist())
        span = clean_diff - corrupt_diff
        results.append(
            {
                "id": row["id"],
                "targets": row["targets"],
                "clean_logit_diff": clean_diff,
                "corrupt_logit_diff": corrupt_diff,
                "patched_positions": len(corrupt_positions),
                "results": [
                    {**label, "logit_diff": value, "recovery": (value - corrupt_diff) / span if span else None}
                    for label, value in zip(labels, patched, strict=True)
                ],
            }
        )
        logger.info("patch %s: %d variants in %.0fs", row["id"], len(variants), time.time() - started)
    output.write_json("patch.json", results)


def _read_rows(path: str) -> list[dict]:
    return [json.loads(line) for line in read_bytes(path).decode().splitlines() if line.strip()]


def _mean_last_resid(harness: SnowballHarness, prompts: list[Prompt]) -> torch.Tensor:
    """Mean residual after each layer at the last position: [layers, hidden]."""
    total = None
    for prompt in prompts:
        _, record = harness.forward(
            prompt.tokens, assistant_start=prompt.assistant_start, capture=torch.tensor([len(prompt.tokens) - 1])
        )
        resid = torch.stack([value[0] for value in record.resid])
        total = resid if total is None else total + resid
    return total / len(prompts)


def _mean_head_outputs(harness: SnowballHarness, prompts: list[Prompt], heads: list[tuple[int, int]]) -> dict:
    """Per (layer, head): the head's output (its o_proj columns times its input), mean over assistant positions,
    then over prompts: {(layer, head): [hidden]}."""
    total: dict[tuple[int, int], torch.Tensor] = {}
    width = harness.head_dim
    for prompt in prompts:
        positions = torch.arange(prompt.assistant_start, len(prompt.tokens))
        _, record = harness.forward(prompt.tokens, assistant_start=prompt.assistant_start, capture=positions)
        for layer, head in heads:
            columns = harness.layers[layer].self_attn.o_proj.weight[:, head * width : (head + 1) * width].float().cpu()
            value = (record.head_in[layer][:, head * width : (head + 1) * width] @ columns.T).mean(0)
            total[(layer, head)] = value + total.get((layer, head), 0)
    return {key: value / len(prompts) for key, value in total.items()}


def steering_vectors(harness: SnowballHarness, tokenizer, spec: dict) -> dict[int, torch.Tensor]:
    """One vector per layer, from ``vector``:

    - ``{"path": .pt}``: one 1-D vector for every layer.
    - ``{"mean_diff": {"a": prompts, "b": prompts}}``: mean last-position residual after the layer, A minus B.
    - ``{"head_diff": {"a": prompts, "b": prompts, "heads": [[layer, head], ...]}}``: per layer, the sum over the
      listed heads of the head's mean output over assistant positions, A minus B.

    ``"unit": true`` scales each layer's vector to norm 1.
    """
    source = spec["vector"]
    layers = sorted({layer for entry in spec["layers"] for layer in _layer_set(entry)})
    if "path" in source:
        vector = torch.load(io.BytesIO(read_bytes(source["path"])), map_location="cpu").flatten().float()
        vectors = dict.fromkeys(layers, vector)
    elif "mean_diff" in source:
        a = _mean_last_resid(harness, read_prompts(source["mean_diff"]["a"], tokenizer))
        b = _mean_last_resid(harness, read_prompts(source["mean_diff"]["b"], tokenizer))
        vectors = {layer: a[layer] - b[layer] for layer in layers}
    elif "head_diff" in source:
        heads = [tuple(item) for item in source["head_diff"]["heads"]]
        a = _mean_head_outputs(harness, read_prompts(source["head_diff"]["a"], tokenizer), heads)
        b = _mean_head_outputs(harness, read_prompts(source["head_diff"]["b"], tokenizer), heads)
        vectors = {}
        for layer, head in heads:
            vectors[layer] = a[(layer, head)] - b[(layer, head)] + vectors.get(layer, 0)
        missing = set(layers) - set(vectors)
        if missing:
            raise ValueError(f"layers {sorted(missing)} have no listed heads")
    else:
        raise ValueError('vector needs "path", "mean_diff" or "head_diff"')
    if source.get("unit"):
        vectors = {layer: vector / vector.norm() for layer, vector in vectors.items()}
    return vectors


def _layer_set(entry: int | list[int]) -> list[int]:
    return [entry] if isinstance(entry, int) else list(entry)


def steer(harness: SnowballHarness, tokenizer, spec: dict, output: Output, stop_ids: set[int]) -> None:
    """Add alpha * v after each layer of a layer set, for a grid of (layer set, alpha); report target log-probs
    and optional greedy text.

    Spec: ``prompts``, ``layers`` (each entry a layer or a list of layers steered together), ``alphas``,
    ``vector`` (see ``steering_vectors``), ``positions`` ('all' or 'assistant'), optional
    ``generate: {"max_new_tokens": N}``, ``batch``.
    """
    vectors = steering_vectors(harness, tokenizer, spec)
    output.save("vectors.pt", vectors)
    assistant_only = spec.get("positions", "assistant") == "assistant"
    grid = [(entry, alpha) for entry in spec["layers"] for alpha in spec["alphas"]]
    variants = [
        Interventions(add=[AddVector(layer, vectors[layer], alpha, assistant_only) for layer in _layer_set(entry)])
        for entry, alpha in grid
    ]
    batch = spec.get("batch", 32)
    results = []
    for prompt in read_prompts(spec["prompts"], tokenizer):
        row = {"id": prompt.id, "grid": [{"layer": entry, "alpha": alpha} for entry, alpha in grid]}
        for target in prompt.targets:
            target_tokens = tokenizer(target, add_special_tokens=False)["input_ids"]
            tokens = list(prompt.tokens) + target_tokens
            positions = torch.arange(len(prompt.tokens) - 1, len(tokens) - 1)
            totals = []
            for start in range(0, len(variants), batch):
                chunk = variants[start : start + batch]
                lp, _ = harness.run(
                    [tokens] * len(chunk),
                    assistant_start=prompt.assistant_start,
                    variants=chunk,
                    logit_positions=positions,
                )
                picked = lp[:, torch.arange(len(target_tokens)), torch.tensor(target_tokens)]
                totals.extend(picked.sum(-1).tolist())
            for cell, total in zip(row["grid"], totals, strict=True):
                cell.setdefault("targets", {})[target] = total
        if spec.get("generate"):
            texts = []
            for start in range(0, len(variants), batch):
                chunk = variants[start : start + batch]
                texts.extend(
                    greedy(harness, tokenizer, prompt, chunk, spec["generate"].get("max_new_tokens", 32), stop_ids)
                )
            for cell, text in zip(row["grid"], texts, strict=True):
                cell["generation"] = text
        results.append(row)
        logger.info("steer %s: %d cells", prompt.id, len(grid))
    output.write_json("steer.json", results)


def greedy(
    harness: SnowballHarness, tokenizer, prompt: Prompt, variants: list[Interventions], max_new: int, stop_ids
) -> list[str]:
    """Batched greedy decoding, one row per variant; finished rows keep running but their text stops."""
    tokens = [list(prompt.tokens) for _ in variants]
    done = [False] * len(variants)
    for _ in range(max_new):
        lp, _ = harness.run(tokens, assistant_start=prompt.assistant_start, variants=variants)
        for row, token in enumerate(lp[:, 0].argmax(-1).tolist()):
            tokens[row].append(token)
            done[row] = done[row] or token in stop_ids
        if all(done):
            break
    texts = []
    for row in tokens:
        completion = row[len(prompt.tokens) :]
        stops = [i for i, token in enumerate(completion) if token in stop_ids]
        texts.append(tokenizer.decode(completion[: stops[0] + 1] if stops else completion))
    return texts
