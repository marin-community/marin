"""Loop-boundary preferences and exact final-token preference loss.

The loss follows Liquid4All/antidoom reference_ftpo_trainer.py at
bd6a126476e18554b0cacaea3fd9f258fdde1f97 (Apache-2.0). It uses the native
suffix detector and per-batch frequency balancing; token IDs never round-trip
through decoded text.
"""

from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from verifyit.tool_call_markup import THINK_CLOSE, TOOL_OPEN, ToolCallStatus, tool_call_regions

from skyrl_train.config.ftpo import FTPOBoundary, FTPOConfig
from skyrl_train.trajectory_runners.trajectory_reward_shaping import LoopCreditConfig, tail_loop

ROLE_HEADER_START = "<|start_header_id|>"
ASSISTANT_HEADER = "<|start_header_id|>assistant<|end_header_id|>"


@dataclass(frozen=True)
class VerifiedFTPOBoundary:
    response_position: int
    chosen_token_ids: tuple[int, ...]


@dataclass(frozen=True)
class FTPOTargets:
    """Existing candidate IDs plus FTPO selection and frozen boundary logits."""

    candidate_ids: torch.Tensor
    chosen_mask: torch.Tensor
    reference_logits: torch.Tensor
    repair_mask: torch.Tensor | None = None
    weights: torch.Tensor | None = None

    def to(self, device: torch.device) -> "FTPOTargets":
        return FTPOTargets(
            self.candidate_ids.to(device),
            self.chosen_mask.to(device),
            self.reference_logits.to(device),
            None if self.repair_mask is None else self.repair_mask.to(device),
            None if self.weights is None else self.weights.to(device),
        )

    def pin_memory(self) -> "FTPOTargets":
        return FTPOTargets(
            self.candidate_ids.pin_memory(),
            self.chosen_mask.pin_memory(),
            self.reference_logits.pin_memory(),
            None if self.repair_mask is None else self.repair_mask.pin_memory(),
            None if self.weights is None else self.weights.pin_memory(),
        )


@dataclass(frozen=True)
class FTPOCounts:
    examples: float
    targets: float
    other: float
    repair_tokens: float = 0.0


@dataclass(frozen=True)
class FTPOInputs:
    logits: torch.Tensor
    targets: FTPOTargets
    rejected_ids: torch.Tensor
    counts: FTPOCounts


def boundary_values(values: torch.Tensor, chosen_mask: torch.Tensor) -> torch.Tensor:
    """Select the single FTPO position in each response; inactive rows select zero."""
    active = chosen_mask.any(-1)
    if torch.any(active.sum(-1) > 1):
        raise ValueError("FTPO supports at most one boundary per response")
    positions = active.long().argmax(-1)
    return values[torch.arange(values.shape[0], device=values.device), positions]


def compact_boundary_logits(
    logits: torch.Tensor,
    attention_mask: torch.Tensor,
    chosen_mask: torch.Tensor,
) -> torch.Tensor:
    """Gather prediction logits after Megatron removes masked sequence positions."""
    response_length = chosen_mask.shape[1]
    # Count real tokens through the predecessor of each response token. This
    # handles unequal prompts and response padding without scattering [B,S,V].
    compact_positions = attention_mask.long().cumsum(-1)[:, -response_length - 1 : -1] - 1
    positions = boundary_values(compact_positions, chosen_mask)
    positions = torch.where(chosen_mask.any((-1, -2)), positions, 0)
    return logits[torch.arange(logits.shape[0], device=logits.device), positions]


def _assistant_initial_thinking(prefix: list[int], decode: Callable[[list[int]], str]) -> bool:
    for index in range(len(prefix) - 1, -1, -1):
        if decode([prefix[index]]) != ROLE_HEADER_START:
            continue
        text = decode(prefix[index:])
        if not text.startswith(ASSISTANT_HEADER):
            return True
        return text[len(ASSISTANT_HEADER) :].strip() not in ("", THINK_CLOSE)
    return True


def repeated_tool_opener_boundary(
    prompt: Sequence[int], response: Sequence[int], mask: Sequence[int], decode: Callable[[list[int]], str]
) -> int | None:
    """Find a repeated opener, allowing whitespace, in a known assistant span.

    A bare assistant header establishes an outside-thinking start. Otherwise,
    require explicit end-thinking. Observations and quoted markers are never
    selected. A malformed earlier call ends parsing for that span.
    """
    start = 0
    while start < len(response):
        if not mask[start]:
            start += 1
            continue
        end = start + 1
        while end < len(response) and mask[end]:
            end += 1
        prefill: list[int] = []
        thinking = _assistant_initial_thinking(list(prompt) + list(response[:start]), decode)
        # Replay prompts can end at the first opener. Retain its assistant
        # context for parsing, but never select a token from that frozen prefix.
        if start == 0 and prompt and decode([prompt[-1]]) == TOOL_OPEN:
            for index in range(len(prompt) - 1, -1, -1):
                if decode([prompt[index]]) != ROLE_HEADER_START:
                    continue
                if decode(list(prompt[index:])).startswith(ASSISTANT_HEADER):
                    prefill = list(prompt[index:])
                    thinking = _assistant_initial_thinking(list(prompt[:-1]), decode)
                break
        span = prefill + list(response[start:end])
        text = decode(span)
        for region in tool_call_regions(text, initial_thinking=thinking):
            if region.status != ToolCallStatus.REPEATED_OPENER:
                continue
            open_positions = [index for index, token in enumerate(span) if decode([token]) == TOOL_OPEN]
            if len(open_positions) != text.count(TOOL_OPEN):
                raise ValueError("FTPO tool markers must align with single special tokens")
            opener_index = text[: region.start].count(TOOL_OPEN)
            position = open_positions[opener_index]
            if decode(span[: position + 1]) != text[: region.payload_start]:
                raise ValueError("FTPO boundary changed during token decoding")
            repeated = open_positions[opener_index + 1]
            if decode(span[: repeated + 1]) != text[: region.end]:
                raise ValueError("FTPO repeated opener changed during token decoding")
            if repeated >= len(prefill):
                return start + repeated - len(prefill)
        start = end
    return None


def select_ftpo_candidates(
    response_ids: Sequence[Sequence[int]],
    loss_masks: Sequence[Sequence[int]],
    candidate_ids: torch.Tensor,
    candidate_logprobs: torch.Tensor,
    *,
    prompt_ids: Sequence[Sequence[int]],
    decode: Callable[[list[int]], str],
    loop: LoopCreditConfig,
    config: FTPOConfig,
    seed: int,
    eligible: Sequence[bool],
    verified_boundaries: Sequence[VerifiedFTPOBoundary | None] | None = None,
    boundary_modes: Sequence[FTPOBoundary] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select detected or verified boundary alternatives and balance the batch.

    Rejected frequency controls example weights. Chosen frequency caps prune
    alternatives using a seeded permutation. No cross-batch state is needed.
    """
    chosen = torch.zeros_like(candidate_ids, dtype=torch.bool)
    weights = torch.zeros(candidate_ids.shape[:2], dtype=torch.float32)
    surfaces: dict[int, str] = {}
    rejected_by_row = {}
    modes = [config.boundary] * len(response_ids) if boundary_modes is None else boundary_modes
    if len(modes) != len(response_ids):
        raise ValueError("FTPO requires one boundary mode per response")
    if any(mode.uses_verified_boundaries for mode in modes):
        if verified_boundaries is None or len(verified_boundaries) != len(response_ids):
            raise ValueError("Verified closure FTPO requires one verified boundary entry per response")
    for row, (prompt, response, mask, allowed, mode) in enumerate(
        zip(prompt_ids, response_ids, loss_masks, eligible, modes, strict=True)
    ):
        if not allowed:
            continue
        verified = None
        if mode.uses_verified_boundaries:
            assert verified_boundaries is not None
            verified = verified_boundaries[row]
            position = None if verified is None else verified.response_position
        elif mode == FTPOBoundary.REPEATED_TOOL_OPENER:
            position = repeated_tool_opener_boundary(prompt, response, mask, decode)
        else:
            hit = tail_loop(response, mask, loop)
            position = None if hit is None else hit.repeat_start
        if position is None:
            continue
        if not 0 <= position < len(response) or not mask[position]:
            raise ValueError("FTPO boundary must identify an unmasked response token")
        rejected = response[position]
        ids = candidate_ids[row, position].tolist()
        scores = candidate_logprobs[row, position]
        if any(token < 0 for token in ids) or len(set(ids)) != len(ids) or not torch.isfinite(scores).all():
            raise ValueError("FTPO boundary candidates require unique token IDs and finite log probabilities")
        rejected_surface = decode([rejected]).lower()
        retained = []
        for column, token in enumerate(ids):
            if verified is not None:
                if token != rejected and token in verified.chosen_token_ids:
                    retained.append(column)
                continue
            if token not in surfaces:
                surfaces[token] = decode([token])
            surface = surfaces[token]
            if mode == FTPOBoundary.REPEATED_TOOL_OPENER and surface.strip() != "{":
                continue
            if token == rejected or surface.lower() == rejected_surface:
                continue
            if len(surface.strip()) < config.min_decoded_chars or (
                config.require_alnum and not any(c.isalnum() for c in surface)
            ):
                continue
            retained.append(column)
        if not retained:
            continue
        best = scores[retained].max()
        retained = [column for column in retained if (scores[column] - best).exp() >= config.min_p]
        # Keep the most likely surviving alternatives, independent of transport order.
        retained.sort(key=lambda column: (-float(scores[column]), ids[column]))
        retained = retained[: config.max_chosen_tokens]
        chosen[row, position, retained] = True
        rejected_by_row[row] = rejected

    occurrences: dict[int, list[tuple[int, int, int]]] = {}
    for row, position, column in chosen.nonzero().tolist():
        occurrences.setdefault(int(candidate_ids[row, position, column]), []).append((row, position, column))
    generator = torch.Generator().manual_seed(seed)
    if occurrences and config.chosen_balance_strength:
        median = float(torch.tensor([len(items) for items in occurrences.values()], dtype=torch.float32).quantile(0.5))
        for items in occurrences.values():
            keep = max(1, round(len(items) * min(1.0, (median / len(items)) ** config.chosen_balance_strength)))
            for index in torch.randperm(len(items), generator=generator)[keep:].tolist():
                chosen[items[index]] = False
    rejected_counts = Counter(token for row, token in rejected_by_row.items() if chosen[row].any())
    if rejected_counts:
        median = float(torch.tensor(list(rejected_counts.values()), dtype=torch.float32).quantile(0.5))
        for row, token in rejected_by_row.items():
            active = chosen[row].any(-1)
            if active.any():
                weights[row, active] = min(1.0, (median / rejected_counts[token]) ** config.rejected_balance_strength)
    return chosen, weights


@torch.no_grad()
def ftpo_counts(
    targets: Sequence[FTPOTargets],
    weights: Sequence[torch.Tensor],
    all_reduce_sum: Callable[[torch.Tensor], torch.Tensor],
) -> FTPOCounts:
    """Count weighted examples and vocabulary entries across the optimizer step."""
    counts = torch.zeros(4, dtype=torch.float64, device=weights[0].device)
    for target, weight in zip(targets, weights, strict=True):
        if target.weights is not None:
            weight = target.weights
        active_weights = weight * target.chosen_mask.any(-1)
        target_count = target.chosen_mask.sum(-1) + 1  # chosen IDs and the rejected ID
        counts[0] += active_weights.sum()
        counts[1] += (active_weights * target_count).sum()
        counts[2] += (active_weights * (target.reference_logits.shape[-1] - target_count)).sum()
        if target.repair_mask is not None:
            if target.repair_mask.shape != weight.shape or torch.any(target.repair_mask & target.chosen_mask.any(-1)):
                raise ValueError("Repair masks must align with responses and exclude FTPO boundary tokens")
            counts[3] += (weight * target.repair_mask).sum()
    return FTPOCounts(*all_reduce_sum(counts).tolist())


def ftpo_loss(inputs: FTPOInputs, config: FTPOConfig) -> tuple[torch.Tensor, dict[str, float]]:
    """Return response-shaped losses for the existing global token-mean reducer.

    MSE denominators count vocabulary entries over the entire optimizer step,
    matching the reference even with different chosen counts or microbatching.
    """
    targets = inputs.targets
    active = targets.chosen_mask.any(-1)
    active_rows = active.any(-1)
    chosen = boundary_values(targets.chosen_mask, targets.chosen_mask)
    ids = boundary_values(targets.candidate_ids, targets.chosen_mask).masked_fill(~chosen, 0)
    logits = torch.where(active_rows[:, None], inputs.logits.float(), 0)
    reference = torch.where(active_rows[:, None], targets.reference_logits.float(), 0).detach()
    if reference.shape != logits.shape:
        raise ValueError("FTPO policy and reference vocabulary shapes must match")
    rejected = inputs.rejected_ids
    rejected_logits = logits.gather(-1, rejected[:, None])
    delta = logits.gather(-1, ids) - rejected_logits
    gap = config.margin - delta
    preferences = (F.softplus(gap) * (gap / config.margin).clamp(0, 1) * chosen).sum(-1) / chosen.sum(-1).clamp(min=1)
    target_mask = torch.zeros_like(logits, dtype=torch.bool)
    rows, columns = chosen.nonzero(as_tuple=True)
    target_mask[rows, ids[rows, columns]] = True
    target_mask.scatter_(1, rejected[:, None], True)
    difference = logits - reference
    other = difference.square().masked_fill(target_mask, 0).sum(-1)
    target = (difference.abs() - config.tau_mse_target).clamp(min=0).square().masked_fill(~target_mask, 0).sum(-1)
    counts = inputs.counts
    losses = preferences + counts.examples * (
        config.lambda_mse * other / (counts.other or 1.0) + config.lambda_mse_target * target / (counts.targets or 1.0)
    )
    values = torch.where(active, losses[:, None], 0)
    wins = ((delta > 0) & chosen).sum(-1) / chosen.sum(-1).clamp(min=1)
    # The worker aggregates these counts before computing the chosen-win rate.
    metrics = {
        "ftpo/active_examples": float(active_rows.sum()),
        "ftpo/chosen_win_sum": float(wins[active_rows].detach().sum()),
    }
    return values, metrics
