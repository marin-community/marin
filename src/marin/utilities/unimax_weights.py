from __future__ import annotations


def unimax_weights(
    corpus_tokens: dict[str, float],
    budget: float,
    max_epochs: float = 1.0,
) -> tuple[dict[str, float], dict[str, float]]:
    """Compute UniMax sampling weights and token allocations.

    Args:
        corpus_tokens: Mapping from subset name to its token count.
        budget: Total token budget available for training.
        max_epochs: Epoch cap ``E``. If the total budget exceeds
            ``E * sum(corpus_tokens)``, a :class:`ValueError` is raised.

    Returns:
        A tuple ``(weights, alloc)`` where ``weights`` is a mapping from subset
        name to its sampling probability and ``alloc`` is the total number of
        tokens allocated to the subset.
    """

    if budget < 0:
        raise ValueError("budget must be non-negative")

    if max_epochs <= 0:
        raise ValueError("max_epochs must be positive")

    if any(v < 0 for v in corpus_tokens.values()):
        raise ValueError("corpus sizes must be non-negative")

    # Handle empty corpora early
    total_tokens = float(sum(corpus_tokens.values()))
    if total_tokens == 0:
        if budget > 0:
            raise ValueError("budget must be zero when corpus is empty")
        return {k: 0.0 for k in corpus_tokens}, {k: 0.0 for k in corpus_tokens}

    # If the requested budget is larger than the allowed max_epochs, fail fast.
    if max_epochs < budget / total_tokens:
        raise ValueError(
            "Impossible constraints: budget requires more epochs than max_epochs allows"
        )
    effective_epochs = max_epochs
    caps = {k: effective_epochs * v for k, v in corpus_tokens.items()}

    remaining_budget = budget
    remaining_caps = {k: c for k, c in caps.items()}
    alloc = {k: 0.0 for k in caps}

    while remaining_caps and remaining_budget > 0:
        share = remaining_budget / len(remaining_caps)
        # Saturate any subset whose cap is below the equal share
        saturated = [k for k, cap in remaining_caps.items() if cap <= share]
        if saturated:
            for k in saturated:
                cap = remaining_caps.pop(k)
                alloc[k] = cap
                remaining_budget -= cap
        else:
            for k in list(remaining_caps.keys()):
                alloc[k] = share
            remaining_budget = 0

    # Normalise allocations to probabilities
    weights = {k: alloc[k] / budget if budget > 0 else 0.0 for k in alloc}

    return weights, alloc
