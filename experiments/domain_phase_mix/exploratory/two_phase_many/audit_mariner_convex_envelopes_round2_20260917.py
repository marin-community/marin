# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "scipy", "joblib"]
# ///

"""Replay saved convexification heads and selected CV scores from direct designs."""

import dataclasses
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.domain_phase_mix.exploratory.two_phase_many import (
    screen_mariner_convex_envelopes_round2_20260917 as screen,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    single_phase_observatory_models_20260902 as base,
)


def cv_score(design: base.Design, response: np.ndarray, ridge: float, head: base.HeadSpec, inner) -> float:
    errors = []
    for train, validation in inner:
        fitted = base.fit_head(
            base.Design(design.values[train], design.ridge, design.names), response[train], ridge, head
        )
        prediction = base.predict_head(fitted, design.values[validation], head)
        errors.append((prediction - response[validation]) ** 2)
    return float(np.sqrt(np.mean(np.concatenate(errors))))


def main() -> None:
    screen.register_candidates()
    manifest = json.loads((screen.OUTPUT / "screen_manifest.json").read_text())
    for filename, checksum in manifest["sources"].items():
        assert screen.previous.digest(Path(filename)) == checksum
    tasks = []
    for tier, models in (("screen", ("ce2_majorant",)), ("certify", ("ce2_envelope",))):
        plan = dataclasses.replace(
            screen.bench.tier_plan(tier, panels=screen.bench.THIRTY_NINE_BUCKET_PANELS), curves=()
        )
        tasks.extend(screen.bench.plan_tasks(plan, models))
    tasks.sort(key=lambda task: (task.model_id, task.panel, task.target, task.fold, task.component_index))
    score_units = set()
    maximum_prediction_error = 0.0
    maximum_score_error = 0.0
    vertex_clip_count = 0
    for task in tasks:
        panel = screen.bench.load_panel(task.panel)
        payload = screen.bench.load_shard(screen.bench.shard_path(screen.OUTPUT, task))
        assert payload is not None and str(payload["status"]) == "ok"
        entry = screen.registry.ENTRY_BY_ID[task.model_id]
        accepted = screen.bench.task_protocol_hashes(task, entry, manifest["split_hash"], panel, screen.OUTPUT)
        assert str(payload["protocol_hash"].item()) in accepted
        features = dataclasses.replace(panel.features, component=task.component)
        model = entry.build(features)
        shape = json.loads(str(payload["shape_json"]))
        ridge = float(payload["ridge"])
        diagnostic = json.loads(str(payload["diagnostics_json"]))
        row, column = np.unravel_index(np.argmin(payload["cv_table"]), payload["cv_table"].shape)
        assert dict(model.base.candidate_shapes(features)[row]) == shape
        assert model.base.ridge_grid[column] == ridge
        table_score_error = abs(float(payload["cv_table"][row, column]) - diagnostic["selection_inner_cv_rmse"])
        assert table_score_error < 1e-12
        design = screen.candidates.design(
            features,
            shape,
            task.model_id,
            tuple(map(float, panel.features.inventory)),
            screen.candidates.MODULE_SOURCE_HASH,
        )
        head = model._spec(shape, base.LinkKind(diagnostic["link"]), float(diagnostic["kappa"]))
        logits = design.values[payload["test"]] @ payload["head_coefficients"] + float(payload["head_intercept"])
        reconstructed = base.link_inverse(logits, float(payload["head_floor"]), head, float(payload["head_cap"]))
        prediction_error = float(np.max(np.abs(reconstructed - payload["prediction"])))
        maximum_prediction_error = max(maximum_prediction_error, prediction_error)
        assert prediction_error < 1e-10
        vertex_clip_count += bool(payload["upper_clip_active_at_vertices"])
        score_unit = (task.model_id, task.panel, task.target, task.fold)
        if score_unit not in score_units:
            split = next(item for item in screen.bench.panel_splits(panel, 1) if item.fold == task.fold)
            response = panel.group(task.target).outcomes[:, task.component_index]
            initial_score = cv_score(design, response, ridge, model.base.head_for(shape), split.inner)
            final_score = cv_score(design, response, ridge, head, split.inner)
            score_error = max(
                abs(initial_score - diagnostic["selection_inner_cv_rmse"]),
                abs(final_score - diagnostic["inner_cv_rmse"]),
            )
            maximum_score_error = max(maximum_score_error, score_error)
            assert score_error < 1e-10
            score_units.add(score_unit)
    result = {
        "fit_records": len(tasks),
        "all_protocol_identities_valid": True,
        "all_saved_shape_ridge_pairs_match_cv_argmin": True,
        "reconstructed_heads": len(tasks),
        "maximum_prediction_error": maximum_prediction_error,
        "independent_selected_cv_replays": len(score_units),
        "maximum_cv_score_error": maximum_score_error,
        "upper_logit_clip_active_at_vertices": vertex_clip_count,
        "audit_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (screen.OUTPUT / "independent_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
