"""Check the new panel and deterministic 12-item diagnostic allocation."""

from typing import Any

import pytest

from experiments.post_training.capability_driven_envs.biotasks_library.run_openrouter_first_wave import (
    load_panel as first_panel,
)
from experiments.post_training.capability_driven_envs.biotasks_library.run_openrouter_v3_additional20 import (
    load_panel,
    select_pilot,
)


def test_panel_is_disjoint_and_has_pinned_routes() -> None:
    models, _ = load_panel()
    first, _ = first_panel()
    assert len(models) == 20
    assert not {m.model for m in models} & {m.model for m in first}
    assert all(m.provider_tag and m.provider_name and m.context_window >= 65536 for m in models)


def test_pilot_spans_prior_difficulties_without_repeated_item() -> None:
    items = {f"item-{i:03d}" for i in range(230)}
    rows: list[dict[str, Any]] = [
        {"task_id": item, "model": f"model-{model}", "score": int(model < int(item[-3:]) % 13)}
        for item in sorted(items)
        for model in range(12)
    ]
    first = select_pilot(rows, items)
    assert first == select_pilot(list(reversed(rows)), items)
    assert len(first) == len(set(first)) == 12
    assert len({sum(r["score"] for r in rows if r["task_id"] == item) for item in first}) >= 3
    with pytest.raises(ValueError, match="Invalid first-wave"):
        select_pilot(rows[1:], items)
