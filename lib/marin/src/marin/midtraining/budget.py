# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""CPT token-budget policy.

Three kinds — fraction of pretrain, fixed tokens, fixed steps — cover
every historical CPT sweep (K=0.20, 10 B / 20 B, smoke). Per-base
variable budgets are intentionally not modeled; the cell author can
construct one ``BudgetPolicy`` per base in their launcher loop.

Cooldown does not use this module — it keeps the original pretrain
``num_train_steps`` so the WSD schedule remains consistent.
"""

from dataclasses import dataclass
from enum import StrEnum


class BudgetKind(StrEnum):
    PRETRAIN_FRACTION = "pretrain_fraction"
    FIXED_TOKENS = "fixed_tokens"
    FIXED_STEPS = "fixed_steps"


@dataclass(frozen=True)
class BudgetPolicy:
    """Declarative CPT budget. Construct via the classmethod helpers."""

    kind: BudgetKind
    label: str | None = None
    fraction: float | None = None
    tokens: int | None = None
    steps: int | None = None

    def __post_init__(self) -> None:
        kind_to_field = {
            BudgetKind.PRETRAIN_FRACTION: ("fraction", self.fraction is not None),
            BudgetKind.FIXED_TOKENS: ("tokens", self.tokens is not None),
            BudgetKind.FIXED_STEPS: ("steps", self.steps is not None),
        }
        expected_field, expected_set = kind_to_field[self.kind]
        if not expected_set:
            raise ValueError(f"BudgetPolicy(kind={self.kind!r}) requires {expected_field}=... to be set")
        for kind, (field_name, is_set) in kind_to_field.items():
            if kind != self.kind and is_set:
                raise ValueError(
                    f"BudgetPolicy(kind={self.kind!r}) must not also set {field_name}; "
                    f"that field belongs to kind {kind!r}"
                )
        if self.kind == BudgetKind.PRETRAIN_FRACTION and not (0 < (self.fraction or 0) <= 1):
            raise ValueError(f"PRETRAIN_FRACTION requires fraction in (0, 1], got {self.fraction!r}")
        if self.kind == BudgetKind.FIXED_TOKENS and (self.tokens or 0) <= 0:
            raise ValueError(f"FIXED_TOKENS requires positive tokens, got {self.tokens!r}")
        if self.kind == BudgetKind.FIXED_STEPS and (self.steps or 0) <= 0:
            raise ValueError(f"FIXED_STEPS requires positive steps, got {self.steps!r}")

    @classmethod
    def pretrain_fraction(cls, fraction: float, *, label: str | None = None) -> "BudgetPolicy":
        return cls(kind=BudgetKind.PRETRAIN_FRACTION, fraction=fraction, label=label)

    @classmethod
    def fixed_tokens(cls, tokens: int, *, label: str | None = None) -> "BudgetPolicy":
        return cls(kind=BudgetKind.FIXED_TOKENS, tokens=tokens, label=label)

    @classmethod
    def fixed_steps(cls, steps: int, *, label: str | None = None) -> "BudgetPolicy":
        return cls(kind=BudgetKind.FIXED_STEPS, steps=steps, label=label)


@dataclass(frozen=True)
class ResolvedBudget:
    """Concrete numbers used for one CPT cell."""

    policy: BudgetPolicy
    base_flops_key: str
    batch_size: int
    seq_len: int
    requested_tokens: int
    actual_tokens: int
    num_train_steps: int
    pretrain_tokens: int
    pretrain_fraction_actual: float
    label: str

    def __post_init__(self) -> None:
        if self.num_train_steps <= 0:
            raise ValueError(f"num_train_steps must be positive, got {self.num_train_steps!r}")
        if self.requested_tokens <= 0:
            raise ValueError(f"requested_tokens must be positive, got {self.requested_tokens!r}")
        if self.actual_tokens <= 0:
            raise ValueError(f"actual_tokens must be positive, got {self.actual_tokens!r}")
        if self.batch_size <= 0 or self.seq_len <= 0:
            raise ValueError(f"batch_size and seq_len must be positive, got {self.batch_size}, {self.seq_len}")


def resolve_cpt_budget(
    policy: BudgetPolicy,
    *,
    base_flops_key: str,
    base_pretrain_tokens: int,
    batch_size: int,
    seq_len: int,
) -> ResolvedBudget:
    """Resolve a :class:`BudgetPolicy` against one base into a concrete budget."""
    if base_pretrain_tokens <= 0:
        raise ValueError(f"base_pretrain_tokens must be positive, got {base_pretrain_tokens!r}")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size!r}")
    if seq_len <= 0:
        raise ValueError(f"seq_len must be positive, got {seq_len!r}")

    tokens_per_step = batch_size * seq_len
    if policy.kind == BudgetKind.PRETRAIN_FRACTION:
        assert policy.fraction is not None
        requested_tokens = round(base_pretrain_tokens * policy.fraction)
    elif policy.kind == BudgetKind.FIXED_TOKENS:
        assert policy.tokens is not None
        requested_tokens = policy.tokens
    elif policy.kind == BudgetKind.FIXED_STEPS:
        assert policy.steps is not None
        requested_tokens = policy.steps * tokens_per_step
    else:
        raise ValueError(f"Unhandled BudgetKind {policy.kind!r}")

    num_train_steps = max(1, round(requested_tokens / tokens_per_step))
    actual_tokens = num_train_steps * tokens_per_step
    pretrain_fraction_actual = actual_tokens / base_pretrain_tokens
    label = policy.label or default_budget_label(actual_tokens)
    return ResolvedBudget(
        policy=policy,
        base_flops_key=base_flops_key,
        batch_size=batch_size,
        seq_len=seq_len,
        requested_tokens=requested_tokens,
        actual_tokens=actual_tokens,
        num_train_steps=num_train_steps,
        pretrain_tokens=base_pretrain_tokens,
        pretrain_fraction_actual=pretrain_fraction_actual,
        label=label,
    )


def default_budget_label(tokens: int) -> str:
    """Compact human-readable label, e.g. ``9p25b`` for 9.25 B tokens."""
    if tokens % 1_000_000_000 == 0:
        return f"{tokens // 1_000_000_000}b"
    return f"{tokens / 1_000_000_000:.2f}b".replace(".", "p")
