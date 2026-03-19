# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .config import ModelConfig
from .model import BlockDiffusionDenoiser


@dataclass(frozen=True)
class NoiseSchedule:
    name: str = "loglinear"
    eps: float = 1e-3
    exp: float = 2.0

    def move_chance(self, t: torch.Tensor) -> torch.Tensor:
        if self.name == "loglinear":
            return t.clamp(min=self.eps, max=1.0)
        if self.name == "cosine":
            return (1.0 - torch.cos(t * torch.pi / 2.0)).clamp(min=self.eps, max=1.0)
        if self.name == "exp":
            return t.pow(self.exp).clamp(min=self.eps, max=1.0)
        raise ValueError(f"unsupported noise schedule {self.name!r}")


@dataclass
class CorruptedBatch:
    context_tokens: torch.Tensor
    target_tokens: torch.Tensor
    noisy_tokens: torch.Tensor
    masked_positions: torch.Tensor
    timesteps: torch.Tensor


class BlockDiffusionObjective:
    def __init__(self, config: ModelConfig, schedule: NoiseSchedule | None = None):
        self.config = config
        self.schedule = schedule or NoiseSchedule()

    def _sample_target_block(
        self,
        tokens: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _, seq_len = tokens.shape
        if seq_len % self.config.block_size != 0:
            raise ValueError(
                f"training tokens length {seq_len} must be divisible by block_size {self.config.block_size}"
            )
        num_blocks = seq_len // self.config.block_size
        if num_blocks < 1:
            raise ValueError("need at least one block per training sample")
        block_index = torch.randint(0, num_blocks, (1,), generator=generator, device=tokens.device).item()
        start = block_index * self.config.block_size
        end = start + self.config.block_size
        context = tokens[:, :start]
        target = tokens[:, start:end]
        return context, target

    def corrupt(
        self,
        target_tokens: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> CorruptedBatch:
        batch = target_tokens.shape[0]
        timesteps = torch.randint(
            0,
            self.config.diffusion_steps,
            (batch,),
            generator=generator,
            device=target_tokens.device,
        )
        continuous_t = (timesteps.float() + 1.0) / float(self.config.diffusion_steps)
        move_chance = self.schedule.move_chance(continuous_t).unsqueeze(-1)
        masked = torch.rand(target_tokens.shape, generator=generator, device=target_tokens.device) < move_chance
        if not masked.any(dim=-1).all():
            rescue = torch.randint(
                0,
                target_tokens.shape[1],
                (batch,),
                generator=generator,
                device=target_tokens.device,
            )
            masked[torch.arange(batch, device=target_tokens.device), rescue] = True
        noisy = torch.where(masked, torch.full_like(target_tokens, self.config.mask_token_id), target_tokens)
        return CorruptedBatch(
            context_tokens=target_tokens.new_empty(batch, 0),
            target_tokens=target_tokens,
            noisy_tokens=noisy,
            masked_positions=masked,
            timesteps=timesteps,
        )

    def prepare_batch(
        self,
        tokens: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> CorruptedBatch:
        context, target = self._sample_target_block(tokens, generator=generator)
        corrupted = self.corrupt(target, generator=generator)
        corrupted.context_tokens = context
        return corrupted

    def loss(
        self,
        model: BlockDiffusionDenoiser,
        tokens: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> dict[str, torch.Tensor]:
        batch = self.prepare_batch(tokens, generator=generator)
        logits = model(batch.context_tokens, batch.noisy_tokens, batch.timesteps)
        flat_loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            batch.target_tokens.reshape(-1),
            reduction="none",
        ).view_as(batch.target_tokens)
        masked_loss = flat_loss[batch.masked_positions]
        loss = masked_loss.mean()
        with torch.no_grad():
            token_accuracy = (logits.argmax(dim=-1) == batch.target_tokens)[batch.masked_positions].float().mean()
        return {
            "loss": loss,
            "masked_accuracy": token_accuracy,
            "num_masked": batch.masked_positions.sum().float(),
        }

    @torch.no_grad()
    def sample_block(
        self,
        model: BlockDiffusionDenoiser,
        *,
        context_tokens: torch.Tensor,
        temperature: float = 1.0,
        top_k: int | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        device = context_tokens.device
        batch = context_tokens.shape[0]
        current = torch.full(
            (batch, self.config.block_size),
            self.config.mask_token_id,
            dtype=torch.long,
            device=device,
        )
        for step in reversed(range(self.config.diffusion_steps)):
            t = torch.full((batch,), step, dtype=torch.long, device=device)
            logits = model(context_tokens, current, t) / max(temperature, 1e-5)
            if top_k is not None:
                top_values, _ = torch.topk(logits, k=min(top_k, logits.shape[-1]), dim=-1)
                cutoff = top_values[..., -1:].clone()
                logits = logits.masked_fill(logits < cutoff, float("-inf"))
            probs = logits.softmax(dim=-1)
            sampled = torch.multinomial(probs.view(-1, probs.shape[-1]), num_samples=1, generator=generator)
            sampled = sampled.view(batch, self.config.block_size)
            if step == 0:
                current = sampled
                continue
            next_t = torch.full((batch,), step - 1, dtype=torch.float32, device=device)
            remask_prob = self.schedule.move_chance((next_t + 1.0) / float(self.config.diffusion_steps)).unsqueeze(-1)
            remask = torch.rand(sampled.shape, generator=generator, device=device) < remask_prob
            current = torch.where(remask, torch.full_like(sampled, self.config.mask_token_id), sampled)
        return current

    @torch.no_grad()
    def sample_sequence(
        self,
        model: BlockDiffusionDenoiser,
        *,
        num_blocks: int,
        prompt_tokens: torch.Tensor | None = None,
        temperature: float = 1.0,
        top_k: int | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        if prompt_tokens is None:
            prompt_tokens = torch.empty(1, 0, dtype=torch.long, device=next(model.parameters()).device)
        context = prompt_tokens
        for _ in range(num_blocks):
            cropped = context[:, -self.config.max_context_tokens :]
            block = self.sample_block(
                model,
                context_tokens=cropped,
                temperature=temperature,
                top_k=top_k,
                generator=generator,
            )
            context = torch.cat([context, block], dim=1)
        return context
