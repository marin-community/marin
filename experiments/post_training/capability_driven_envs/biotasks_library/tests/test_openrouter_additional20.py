"""Check the new panel and deterministic 12-item diagnostic allocation."""

import json
from pathlib import Path
from typing import Any

import pytest

from experiments.post_training.capability_driven_envs.biotasks_library.finalize_openrouter_v3_effective20 import (
    SELECTION_ID,
    selected_models,
)
from experiments.post_training.capability_driven_envs.biotasks_library.run_openrouter_first_wave import (
    load_panel as first_panel,
)
from experiments.post_training.capability_driven_envs.biotasks_library.run_openrouter_v3_additional20 import (
    load_panel,
    select_pilot,
)
from experiments.post_training.capability_driven_envs.biotasks_library.run_openrouter_v3_topup16 import (
    load_panel as load_topup_panel,
)


def test_panel_is_disjoint_and_has_pinned_routes() -> None:
    models, _ = load_panel()
    first, _ = first_panel()
    assert len(models) == 20
    assert not {m.model for m in models} & {m.model for m in first}
    assert all(m.provider_tag and m.provider_name and m.context_window >= 65536 for m in models)
    topup, _ = load_topup_panel()
    assert len(topup) == 16
    assert not {m.model for m in topup} & {m.model for m in [*models, *first]}


def test_effective_selection_requires_seven_plus_thirteen_distinct_models(tmp_path: Path) -> None:
    initial, initial_sha = load_panel()
    topup, topup_sha = load_topup_panel()
    selection = {
        "selection_id": SELECTION_ID,
        "initial_panel_sha256": initial_sha,
        "topup_panel_sha256": topup_sha,
        "initial_models": [m.model for m in initial[:7]],
        "topup_models": [m.model for m in topup[:13]],
    }
    path = tmp_path / "selection.json"
    path.write_text(json.dumps(selection))
    models, _ = selected_models(path)
    assert len({model.model for model, _, _, _ in models}) == 20
    selection["topup_models"][0] = selection["topup_models"][1]
    path.write_text(json.dumps(selection))
    with pytest.raises(ValueError, match="Effective20 selection"):
        selected_models(path)


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
