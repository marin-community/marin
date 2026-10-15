# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path


def _load_rank_module():
    path = Path(__file__).with_name("rank.py")
    spec = importlib.util.spec_from_file_location("sotopia_length_control_rank", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_length_control_removes_synthetic_verbosity_bonus() -> None:
    rank = _load_rank_module()
    episodes = []
    for task_index in range(40):
        task = f"task-{task_index}"
        task_effect = (task_index % 5 - 2) * 0.08
        short_length = 80 + 3 * task_index
        long_length = 280 + 11 * task_index
        median = (short_length + long_length) / 2
        scale = 180
        for model, length in (("short", short_length), ("long", long_length)):
            length_bonus = 0.9 * math.tanh((length - median) / scale)
            score = 1 / (1 + math.exp(-(0.7 + task_effect + length_bonus)))
            episodes.append(rank.EpisodeScore(model, task, score, length))

    scores, _, _ = rank.length_controlled_scores(episodes, regularization=(100.0,))
    by_model = {row.model: row for row in scores}

    assert by_model["long"].raw_overall > by_model["short"].raw_overall + 0.15
    assert math.isclose(
        by_model["long"].length_controlled_overall,
        by_model["short"].length_controlled_overall,
        abs_tol=0.02,
    )
