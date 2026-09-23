# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Measure top-k centering error on full Qwen distributions from a matched run.

At the start of a one-pass PPO update, current and stored-old policies are equal.
For one token and TIS cap ``c``, the exact full-vocabulary coefficient is
``min(p_i, c * q_i)``. We compare its score gradient with the implemented
top-k approximation, which models the omitted behavior tail as proportional
to the current policy while preserving its total mass.

Example::

    python -m experiments.post_training.measure_score_centering_tail \
        --behavior-model /tmp/qwen-base --current-model /tmp/qwen-step8 \
        --eval-root s3://marin-us-east-02a/.../global_step_0_evals \
        --output /tmp/score-centering-tail.csv
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import click
import fsspec
import torch
from rigging.filesystem.storage_path import prefix_join
from transformers import AutoModelForCausalLM, AutoTokenizer

POSITIONS = (("prompt", 0), ("answer16", 16), ("answer32", 32), ("answer64", 64), ("answer128", 128), ("answer256", 256))


@dataclass(frozen=True)
class EvalContext:
    suite: str
    prompt_sha16: str
    token_ids: torch.Tensor
    positions: tuple[int, ...]


def _filesystem(uri: str):
    if uri.startswith("s3://"):
        return fsspec.filesystem(
            "s3",
            client_kwargs={"endpoint_url": "https://cwobject.com"},
            config_kwargs={"s3": {"addressing_style": "virtual"}},
        )
    return fsspec.filesystem("file")


def _contexts(eval_root: str, tokenizer) -> list[EvalContext]:
    fs = _filesystem(eval_root)
    contexts = []
    for suite in ("val-gsm8k", "val-math500"):
        path = prefix_join(eval_root, f"{suite}.jsonl")
        with fs.open(path) as stream:
            for index in range(4):
                row = json.loads(stream.readline())
                prompt = row["input_prompt"]
                response = row["output_response"]
                ids = tokenizer.encode(prompt + response, add_special_tokens=False)
                prompt_length = len(tokenizer.encode(prompt, add_special_tokens=False))
                positions = tuple(prompt_length + offset - 1 for _, offset in POSITIONS)
                if positions[-1] >= len(ids):
                    raise ValueError(f"{suite} row {index} has fewer than 256 answer tokens")
                digest = hashlib.sha256(prompt.encode()).hexdigest()[:16]
                contexts.append(EvalContext(suite, digest, torch.tensor(ids[: positions[-1] + 1]), positions))
    return contexts


def _distributions(model_path: Path, contexts: list[EvalContext], *, threads: int) -> list[torch.Tensor]:
    torch.set_num_threads(threads)
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float32, low_cpu_mem_usage=True)
    model.eval()
    vectors = []
    with torch.inference_mode():
        for context in contexts:
            logits = model(context.token_ids.unsqueeze(0)).logits[0, list(context.positions)].double()
            vectors.append(logits.softmax(dim=-1).cpu())
    del model
    return vectors


def _score_gradient(coefficients: torch.Tensor, policy: torch.Tensor) -> torch.Tensor:
    return coefficients - policy * coefficients.sum()


def _calibrated_policy(behavior: torch.Tensor, target_abs_log_ratio: float, seed: int) -> torch.Tensor:
    """Perturb real Qwen logits to a named mismatch magnitude for sensitivity analysis."""
    generator = torch.Generator().manual_seed(seed)
    noise = torch.randn(behavior.shape, generator=generator, dtype=behavior.dtype)
    baseline = behavior.log()

    def mismatch(scale: float) -> float:
        policy = (baseline + scale * noise).softmax(dim=-1)
        return (behavior * (policy.log() - baseline).abs()).sum().item()

    low, high = 0.0, 1.0
    while mismatch(high) < target_abs_log_ratio and high < 32:
        high *= 2
    if mismatch(high) < target_abs_log_ratio:
        raise ValueError("could not reach the requested calibrated mismatch")
    for _ in range(18):
        middle = (low + high) / 2
        if mismatch(middle) < target_abs_log_ratio:
            low = middle
        else:
            high = middle
    return (baseline + ((low + high) / 2) * noise).softmax(dim=-1)


def approximation_metrics(policy: torch.Tensor, behavior: torch.Tensor, width: int, cap: float) -> dict:
    """Compare exact and proportional-tail score gradients for p=o, A=1."""
    if policy.ndim != 1 or behavior.shape != policy.shape or width < 1 or width > policy.numel():
        raise ValueError("policy and behavior must be matching vectors with a valid head width")
    if cap <= 0:
        raise ValueError("TIS cap must be positive")
    head = behavior.topk(width).indices
    omitted = torch.ones_like(policy, dtype=torch.bool)
    omitted[head] = False
    exact = torch.minimum(policy, cap * behavior)
    modeled = exact.clone()
    p_tail = policy[omitted].sum()
    q_tail = behavior[omitted].sum()
    modeled[omitted] = policy[omitted] * min(1.0, (cap * q_tail / p_tail).item()) if p_tail > 0 else 0
    exact_gradient = _score_gradient(exact, policy)
    modeled_gradient = _score_gradient(modeled, policy)
    error = modeled_gradient - exact_gradient
    return {
        "behavior_tail_mass": q_tail.item(),
        "current_tail_mass": p_tail.item(),
        "behavior_cap_mass": behavior[policy > cap * behavior].sum().item(),
        "behavior_log_ratio_abs_mean": (behavior * (policy.log() - behavior.log()).abs()).sum().item(),
        "exact_gradient_l1": exact_gradient.abs().sum().item(),
        "exact_gradient_l2": exact_gradient.norm().item(),
        "error_l1": error.abs().sum().item(),
        "error_l2": error.norm().item(),
        "relative_error_l1": (
            error.abs().sum().item() / exact_gradient.abs().sum().item() if exact_gradient.abs().sum() > 1e-12 else None
        ),
    }


@click.command()
@click.option("--behavior-model", type=click.Path(path_type=Path, exists=True), required=True)
@click.option("--current-model", type=click.Path(path_type=Path, exists=True), required=True)
@click.option("--eval-root", required=True)
@click.option("--output", type=click.Path(path_type=Path), required=True)
@click.option("--threads", type=int, default=4, show_default=True)
@click.option("--topk", "widths", type=int, multiple=True, default=(1, 4, 8, 32, 128), show_default=True)
def main(
    behavior_model: Path, current_model: Path, eval_root: str, output: Path, threads: int, widths: tuple[int, ...]
) -> None:
    tokenizer = AutoTokenizer.from_pretrained(behavior_model)
    contexts = _contexts(eval_root, tokenizer)
    behavior = _distributions(behavior_model, contexts, threads=threads)
    current = _distributions(current_model, contexts, threads=threads)
    rows = []
    for context, q_vectors, p_vectors in zip(contexts, behavior, current, strict=True):
        for (position, _), q, p in zip(POSITIONS, q_vectors, p_vectors, strict=True):
            if not torch.isclose(q.sum(), q.new_tensor(1), atol=1e-8) or not torch.isclose(
                p.sum(), p.new_tensor(1), atol=1e-8
            ):
                raise ValueError("a model probability vector is not normalized")
            seed = int(hashlib.sha256(f"{context.prompt_sha16}:{position}".encode()).hexdigest()[:8], 16)
            scenarios = (
                ("real_base_to_step8", p),
                ("calibrated_0.016", _calibrated_policy(q, 0.016, seed)),
                ("calibrated_0.05", _calibrated_policy(q, 0.05, seed)),
            )
            for scenario, current_policy in scenarios:
                for width in widths:
                    for cap in (1.001, 1.05, 2.0):
                        rows.append(
                            {
                                "suite": context.suite,
                                "prompt_sha16": context.prompt_sha16,
                                "position": position,
                                "scenario": scenario,
                                "topk": width,
                                "tis_cap": cap,
                                **approximation_metrics(current_policy, q, width, cap),
                            }
                        )
            if approximation_metrics(p, q, p.numel(), 1.05)["error_l1"] != 0:
                raise AssertionError("full-vocabulary approximation must be exact")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    click.echo(f"Wrote {len(rows)} full-vocabulary comparisons to {output}")


if __name__ == "__main__":
    main()
