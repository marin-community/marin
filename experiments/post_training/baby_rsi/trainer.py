# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The trainer box: on-policy GRPO with a LoRA adapter, as a stream from tasks to the graded rollouts it trains on.

Each step takes ``tasks_per_step`` tasks from the stream, runs ``samples_per_task`` attempts of each with the served
policy, yields the groups, and makes one optimizer step. The advantage of an attempt is its reward less the mean
reward of its group; an attempt that an infrastructure failure decided has no reward and no gradient. The loss is a
clipped importance-weighted policy gradient against the sampling log probabilities that vLLM returned, averaged
over all response tokens of the step. After the step, the new adapter loads into the server and serves the next
attempts, including those of other boxes, for example the solver rollouts of task synthesis.
"""

import asyncio
import json
import logging
import math
from collections.abc import AsyncIterable, AsyncIterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel, get_peft_model, set_peft_model_state_dict
from safetensors.torch import load_file
from taskcompendium.models import TaskSpec
from torch.utils.checkpoint import checkpoint
from transformers import AutoModelForCausalLM

from experiments.post_training.baby_rsi.policy import VllmServer
from experiments.post_training.baby_rsi.resume import OPTIMIZER_FILE, adapter_name
from experiments.post_training.baby_rsi.rollouts import PolicyRunner, TaskRollouts, reward
from experiments.post_training.baby_rsi.streams import batched

logger = logging.getLogger(__name__)

LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
LOGIT_CHUNK = 2048
"""Response positions whose logits one chunk computes, so that a long step does not hold all vocabulary logits."""


@dataclass(frozen=True)
class TrainerConfig:
    model: str
    lora_rank: int
    learning_rate: float
    tasks_per_step: int
    samples_per_task: int
    max_steps: int
    micro_batch_tokens: int
    """Padded tokens of one forward pass."""
    adapter_dir: str
    device: str
    clip_low: float = 0.2
    clip_high: float = 0.28
    max_grad_norm: float = 1.0
    seed: int = 0


@dataclass(frozen=True)
class Sample:
    """One attempt as a training sequence: its prompt and response tokens, and what each response token adds."""

    token_ids: tuple[int, ...]
    response_start: int
    loss_mask: tuple[int, ...]
    """One value per response token: 1 for a sampled token, 0 for an observation, such as a tool result."""
    behavior_logprobs: tuple[float, ...]
    advantage: float


def training_samples(groups: Sequence[TaskRollouts]) -> list[Sample]:
    """The samples of the attempts with a reward in groups whose rewards differ; equal rewards teach nothing."""
    samples = []
    for group in groups:
        scored = [(graded, value) for graded in group.rollouts if (value := reward(graded)) is not None]
        if len({value for _, value in scored}) < 2:
            continue
        mean = sum(value for _, value in scored) / len(scored)
        for graded, value in scored:
            conversations = graded.rollout.conversations
            if not conversations or conversations[-1].logprobs is None:
                continue
            conversation = conversations[-1]
            samples.append(
                Sample(
                    token_ids=(*conversation.prompt_token_ids, *conversation.response_token_ids),
                    response_start=len(conversation.prompt_token_ids),
                    loss_mask=conversation.loss_mask,
                    behavior_logprobs=conversation.logprobs,
                    advantage=value - mean,
                )
            )
    return samples


class LoraLearner:
    """The policy's trainable LoRA weights on one GPU, and their optimizer."""

    def __init__(self, config: TrainerConfig):
        torch.manual_seed(config.seed)
        self.config = config
        self.device = torch.device(config.device)
        model = AutoModelForCausalLM.from_pretrained(config.model, dtype=torch.bfloat16, attn_implementation="sdpa")
        model.config.use_cache = False
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        lora = LoraConfig(
            r=config.lora_rank,
            lora_alpha=2 * config.lora_rank,
            target_modules=list(LORA_TARGETS),
            lora_dropout=0.0,
            task_type="CAUSAL_LM",
        )
        self.model: PeftModel = get_peft_model(model, lora).to(self.device)
        self.model.enable_input_require_grads()
        self.parameters = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        self.optimizer = torch.optim.AdamW(self.parameters, lr=config.learning_rate, betas=(0.9, 0.99), weight_decay=0.0)

    def update(self, samples: Sequence[Sample]) -> dict[str, float]:
        """One optimizer step on ``samples``, with the loss averaged over all their sampled tokens."""
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        total = sum(sum(sample.loss_mask) for sample in samples)
        sums = {"loss": 0.0, "clipped": 0.0, "ratio": 0.0, "mismatch": 0.0}
        for micro in _micro_batches(samples, self.config.micro_batch_tokens):
            for name, value in self._backward(micro, total).items():
                sums[name] += value
        norm = torch.nn.utils.clip_grad_norm_(self.parameters, self.config.max_grad_norm)
        self.optimizer.step()
        metrics = {name: value / total for name, value in sums.items() if name != "loss"}
        return {"loss": sums["loss"], **metrics, "grad_norm": float(norm), "tokens": float(total)}

    def _backward(self, samples: Sequence[Sample], total: int) -> dict[str, float]:
        length = max(len(sample.token_ids) for sample in samples)
        input_ids = torch.zeros((len(samples), length), dtype=torch.long)
        attention = torch.zeros((len(samples), length), dtype=torch.long)
        rows, positions, targets, behavior, advantages = [], [], [], [], []
        for row, sample in enumerate(samples):
            input_ids[row, : len(sample.token_ids)] = torch.tensor(sample.token_ids)
            attention[row, : len(sample.token_ids)] = 1
            for offset, mask in enumerate(sample.loss_mask):
                if mask:
                    position = sample.response_start + offset
                    rows.append(row)
                    positions.append(position - 1)  # The hidden state before a token predicts it.
                    targets.append(sample.token_ids[position])
                    behavior.append(sample.behavior_logprobs[offset])
                    advantages.append(sample.advantage)
        base = self.model.get_base_model()
        hidden = base.model(input_ids=input_ids.to(self.device), attention_mask=attention.to(self.device))[0]
        selected = hidden[torch.tensor(rows, device=self.device), torch.tensor(positions, device=self.device)]
        target = torch.tensor(targets, device=self.device)
        old = torch.tensor(behavior, device=self.device, dtype=torch.float32)
        advantage = torch.tensor(advantages, device=self.device, dtype=torch.float32)
        losses, stats = [], []
        for start in range(0, len(targets), LOGIT_CHUNK):
            end = start + LOGIT_CHUNK
            # The chunk's logits recompute in the backward pass, so that the step holds one chunk of them at a time.
            chunk = checkpoint(
                self._chunk_loss,
                selected[start:end],
                target[start:end],
                old[start:end],
                advantage[start:end],
                use_reentrant=False,
            )
            assert chunk is not None
            loss, chunk_stats = chunk
            losses.append(loss)
            stats.append(chunk_stats)
        loss = torch.stack(losses).sum() / total
        loss.backward()
        clipped, ratio, mismatch = torch.stack(stats).sum(0).tolist()
        return {"loss": loss.detach().item(), "clipped": clipped, "ratio": ratio, "mismatch": mismatch}

    def _chunk_loss(
        self, hidden: torch.Tensor, target: torch.Tensor, old: torch.Tensor, advantage: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The negative clipped surrogate summed over the chunk, and its clipped count, ratio sum and mismatch sum."""
        logits = self.model.get_base_model().lm_head(hidden).float()
        logprob = logits.log_softmax(-1).gather(-1, target[:, None])[:, 0]
        ratio = torch.exp(logprob - old)
        bounded = ratio.clamp(1 - self.config.clip_low, 1 + self.config.clip_high)
        surrogate = torch.minimum(ratio * advantage, bounded * advantage)
        with torch.no_grad():
            stats = torch.stack([(bounded != ratio).sum().float(), ratio.sum(), (old - logprob).abs().sum()])
        return -surrogate.sum(), stats

    def save(self, path: Path) -> Path:
        """The adapter for vLLM, and the optimizer state last, so that a directory with it is complete."""
        self.model.save_pretrained(str(path))
        torch.save(self.optimizer.state_dict(), path / OPTIMIZER_FILE)
        return path

    def load(self, path: Path) -> None:
        """The adapter weights and the optimizer state of a saved step."""
        set_peft_model_state_dict(self.model, load_file(str(path / "adapter_model.safetensors")))
        self.optimizer.load_state_dict(torch.load(path / OPTIMIZER_FILE, map_location=self.device))


def _micro_batches(samples: Sequence[Sample], budget: int) -> list[list[Sample]]:
    """Groups of samples, sorted by length, whose padded size stays within ``budget`` tokens (or one sample)."""
    batches: list[list[Sample]] = []
    batch: list[Sample] = []
    for sample in sorted(samples, key=lambda item: len(item.token_ids)):
        if batch and len(sample.token_ids) * (len(batch) + 1) > budget:
            batches.append(batch)
            batch = []
        batch.append(sample)
    if batch:
        batches.append(batch)
    return batches


async def train(
    runner: PolicyRunner,
    server: VllmServer,
    learner: LoraLearner,
    tasks: AsyncIterable[TaskSpec],
    *,
    start_step: int = 0,
) -> AsyncIterator[TaskRollouts]:
    """Train the served policy on ``tasks`` and yield each group of graded attempts, before its update.

    A resumed run starts after ``start_step``, from the adapter that it saved in ``adapter_dir``. The stream ends after
    ``max_steps`` steps, or earlier when ``tasks`` ends.
    """
    config = learner.config
    step = start_step
    if step:
        path = Path(config.adapter_dir) / adapter_name(step)
        learner.load(path)
        await server.load_adapter(adapter_name(step), str(path))
        runner.policy.model = adapter_name(step)
    if step >= config.max_steps:
        return
    async for batch in batched(tasks, config.tasks_per_step):
        groups = await runner.run_all(batch, config.samples_per_task)
        for group in groups:
            yield group
        samples = training_samples(groups)
        metrics = await asyncio.to_thread(learner.update, samples) if samples else {}
        step += 1
        name = adapter_name(step)
        path = await asyncio.to_thread(learner.save, Path(config.adapter_dir) / name)
        await server.load_adapter(name, str(path))
        runner.policy.model = name
        rewards = [value for group in groups for value in group.rewards]
        logger.info(
            "train %s",
            json.dumps(
                {
                    "step": step,
                    "mean_reward": sum(rewards) / len(rewards) if rewards else math.nan,
                    "samples": len(samples),
                    "groups_with_signal": sum(len(set(group.rewards)) > 1 for group in groups),
                    **metrics,
                }
            ),
        )
        if step >= config.max_steps:
            return
