# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Fit SOTOPIA length correction from a versioned comparison cohort."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from typing import cast

import numpy as np
from finestore.reader import ReadView
from rigging.filesystem.storage_path import StoragePath, prefix_join
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

from harbor.models.trial.result import TrialResult
from harbor_config.errors import ErrorCategory, error_category
from harbor_config.models.job.config import SotopiaCohort

SUMMARY = re.compile(r"/([^/]+)/attempts/(\d+)/agent/sotopia-summary\.json$")
DIMENSIONS = (
    "sotopia_believability",
    "sotopia_relationship",
    "sotopia_knowledge",
    "sotopia_secret",
    "sotopia_social_rules",
    "sotopia_financial_and_material_benefits",
    "sotopia_goal",
)
REGULARIZATION = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0)


@dataclass(frozen=True)
class SotopiaCell:
    model: str
    results_path: str


@dataclass(frozen=True)
class EpisodeScore:
    model: str
    task_id: str
    normalized_overall: float
    response_characters: int | None


@dataclass(frozen=True)
class ModelScore:
    model: str
    length_controlled_overall: float
    raw_overall: float
    scored_episodes: int
    judged_episodes: int
    median_response_characters: float


@dataclass(frozen=True)
class AdjustedEpisodeScore:
    model: str
    task_id: str
    raw_overall: float
    response_characters: int | None
    length_controlled_overall: float


@dataclass(frozen=True)
class LengthControlFit:
    scores: list[ModelScore]
    episodes: list[AdjustedEpisodeScore]
    selected_regularization: float
    cross_validation_losses: dict[float, float]


def _scored_failure(grading: dict | None) -> bool:
    if not grading or not grading.get("detail"):
        return False
    detail = json.loads(grading["detail"])
    return detail.get("scored") is True and grading.get("score") == 0


def _normalized_overall(metrics: dict[str, float]) -> float | None:
    if not all(name in metrics for name in DIMENSIONS):
        return None
    return sum(float(metrics[name]) for name in DIMENSIONS) / len(DIMENSIONS)


def _evaluated_response_characters(summary: dict) -> int:
    names: list[str] = []
    for event in summary["transcript"][0]:
        if event["sender"] == "Environment" and event["receiver"] not in names:
            names.append(event["receiver"])
    if len(names) != 2:
        raise ValueError("SOTOPIA transcript does not identify two participants")
    evaluated_name = names[int(summary["evaluated_agent_index"])]
    return sum(
        len(str(event["message"]))
        for turn in summary["transcript"]
        for event in turn
        if event["sender"] == evaluated_name
    )


def _summary_overall(summary: dict) -> float:
    ranges = {
        "believability": (0.0, 10.0),
        "relationship": (-5.0, 5.0),
        "knowledge": (0.0, 10.0),
        "secret": (-10.0, 0.0),
        "social_rules": (-10.0, 0.0),
        "financial_and_material_benefits": (-5.0, 5.0),
        "goal": (0.0, 10.0),
    }
    raw = summary["scores"][int(summary["evaluated_agent_index"])]
    return sum(
        (float(raw[name]) - minimum) / (maximum - minimum)
        for name, (minimum, maximum) in ranges.items()
    ) / len(ranges)


def read_episodes(cell: SotopiaCell) -> list[EpisodeScore]:
    """Load scoreable episodes and transcript lengths from a sealed archive."""
    rows = ReadView(cell.results_path).scan(
        "samples", columns=["doc_id", "trial_id", "metrics", "grading"]
    )
    if rows is None:
        raise ValueError(f"no normalized SOTOPIA samples for {cell.model}")
    sample_rows = rows.to_pylist(maps_as_pydicts="strict")
    pattern = StoragePath(prefix_join(cell.results_path, "harbor_jobs/**/sotopia-summary.json"))
    summaries: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for path in pattern.glob():
        if match := SUMMARY.search(str(path)):
            summaries[match.group(1)].append((int(match.group(2)), str(path)))

    summary_paths = {
        row["trial_id"]: max(summaries[row["trial_id"]])[1]
        for row in sample_rows
        if summaries[row["trial_id"]]
    }
    with ThreadPoolExecutor() as executor:
        payloads = dict(zip(
            summary_paths.values(),
            executor.map(StoragePath.read_bytes, (StoragePath(path) for path in summary_paths.values())),
            strict=True,
        ))

    episodes = []
    for row in sample_rows:
        metrics = dict(row.get("metrics") or {})
        overall = _normalized_overall(metrics)
        if overall is None:
            if _scored_failure(row.get("grading")):
                episodes.append(EpisodeScore(cell.model, row["doc_id"], 0.0, None))
            continue
        summary_path = summary_paths.get(row["trial_id"])
        if summary_path is None:
            raise ValueError(
                f"missing SOTOPIA summary for {cell.model} / {row['trial_id']}"
            )
        summary = json.loads(payloads[summary_path])
        if not math.isclose(_summary_overall(summary), overall, abs_tol=1e-9):
            raise ValueError(
                f"summary score differs from FineStore for {cell.model} / {row['trial_id']}"
            )
        episodes.append(
            EpisodeScore(
                cell.model,
                row["doc_id"],
                overall,
                _evaluated_response_characters(summary),
            )
        )
    if len(episodes) < 90:
        raise ValueError(f"fewer than 90 scoreable SOTOPIA episodes for {cell.model}")
    return episodes


def _design_matrix(
    episodes: list[EpisodeScore],
    model_index: dict[str, int],
    task_index: dict[str, int],
    task_medians: dict[str, float],
    length_scale: float,
    *,
    length_controlled: bool,
) -> sparse.csr_matrix:
    rows = np.arange(len(episodes))
    models = np.asarray([model_index[row.model] for row in episodes])
    tasks = np.asarray([task_index[row.task_id] for row in episodes])
    length_delta = np.asarray(
        [float(row.response_characters) - task_medians[row.task_id] for row in episodes]
    )
    length_feature = np.tanh(length_delta / length_scale)
    if length_controlled:
        length_feature.fill(0)
    model_effects = sparse.csr_matrix(
        (np.ones(len(rows)), (rows, models)), shape=(len(rows), len(model_index))
    )
    task_effects = sparse.csr_matrix(
        (np.ones(len(rows)), (rows, tasks)), shape=(len(rows), len(task_index))
    )
    length_effect = sparse.csr_matrix(length_feature[:, np.newaxis])
    return cast(
        sparse.csr_matrix,
        sparse.hstack([model_effects, task_effects, length_effect], format="csr"),
    )


def _fractional_log_loss(expected: np.ndarray, predicted: np.ndarray) -> float:
    predicted = np.clip(predicted, 1e-12, 1 - 1e-12)
    return float(
        np.mean(
            -(expected * np.log(predicted) + (1 - expected) * np.log(1 - predicted))
        )
    )


def length_controlled_scores(
    episodes: list[EpisodeScore],
    regularization: tuple[float, ...] = REGULARIZATION,
) -> LengthControlFit:
    """Fit the cohort GLM and predict every judged episode at zero length delta."""
    judged = [row for row in episodes if row.response_characters is not None]
    models = sorted({row.model for row in episodes})
    tasks = sorted({row.task_id for row in judged})
    model_index = {model: index for index, model in enumerate(models)}
    task_index = {task: index for index, task in enumerate(tasks)}
    lengths: dict[str, list[int]] = defaultdict(list)
    for row in judged:
        lengths[row.task_id].append(int(row.response_characters))
    task_medians = {task: float(np.median(values)) for task, values in lengths.items()}
    deltas = np.asarray(
        [float(row.response_characters) - task_medians[row.task_id] for row in judged]
    )
    length_scale = float(np.std(deltas, ddof=1))
    if not math.isfinite(length_scale) or length_scale == 0:
        raise ValueError("SOTOPIA cohort has no usable response-length variation")

    matrix = _design_matrix(
        judged,
        model_index,
        task_index,
        task_medians,
        length_scale,
        length_controlled=False,
    )
    outcomes = np.asarray([row.normalized_overall for row in judged])
    duplicated_matrix = sparse.vstack([matrix, matrix], format="csr")
    labels = np.concatenate(
        [np.ones(len(outcomes), dtype=int), np.zeros(len(outcomes), dtype=int)]
    )
    weights = np.concatenate([outcomes, 1 - outcomes])
    task_groups = np.asarray([task_index[row.task_id] for row in judged])
    folds = min(5, len(tasks))
    if folds < 2:
        raise ValueError("SOTOPIA length control requires at least two matched tasks")

    losses = {}
    for strength in regularization:
        fold_losses = []
        for train, test in GroupKFold(n_splits=folds).split(
            matrix, outcomes, groups=task_groups
        ):
            duplicated_train = np.concatenate([train, train + len(outcomes)])
            model = LogisticRegression(
                C=strength,
                solver="liblinear",
                fit_intercept=False,
                random_state=0,
                max_iter=3000,
            )
            model.fit(
                duplicated_matrix[duplicated_train],
                labels[duplicated_train],
                sample_weight=weights[duplicated_train],
            )
            fold_losses.append(
                _fractional_log_loss(
                    outcomes[test], model.predict_proba(matrix[test])[:, 1]
                )
            )
        losses[strength] = float(np.mean(fold_losses))
    selected = min(losses, key=lambda strength: losses[strength])
    model = LogisticRegression(
        C=selected,
        solver="liblinear",
        fit_intercept=False,
        random_state=0,
        max_iter=5000,
    )
    model.fit(duplicated_matrix, labels, sample_weight=weights)
    controlled_matrix = _design_matrix(
        judged,
        model_index,
        task_index,
        task_medians,
        length_scale,
        length_controlled=True,
    )
    controlled = model.predict_proba(controlled_matrix)[:, 1]

    predictions: dict[str, list[float]] = defaultdict(list)
    adjusted_episodes = []
    for row, prediction in zip(judged, controlled, strict=True):
        predictions[row.model].append(float(prediction))
        adjusted_episodes.append(
            AdjustedEpisodeScore(
                model=row.model,
                task_id=row.task_id,
                raw_overall=row.normalized_overall,
                response_characters=row.response_characters,
                length_controlled_overall=float(prediction),
            )
        )
    adjusted_episodes.extend(
        AdjustedEpisodeScore(
            model=row.model,
            task_id=row.task_id,
            raw_overall=row.normalized_overall,
            response_characters=None,
            length_controlled_overall=0.0,
        )
        for row in episodes
        if row.response_characters is None
    )
    results = []
    for model_name in models:
        model_rows = [row for row in episodes if row.model == model_name]
        model_judged = [
            row for row in model_rows if row.response_characters is not None
        ]
        failures = len(model_rows) - len(model_judged)
        controlled_values = predictions[model_name] + [0.0] * failures
        results.append(
            ModelScore(
                model=model_name,
                length_controlled_overall=float(np.mean(controlled_values)),
                raw_overall=float(
                    np.mean([row.normalized_overall for row in model_rows])
                ),
                scored_episodes=len(model_rows),
                judged_episodes=len(model_judged),
                median_response_characters=(
                    float(np.median([row.response_characters for row in model_judged]))
                    if model_judged
                    else 0.0
                ),
            )
        )
    results.sort(key=lambda row: (-row.length_controlled_overall, row.model))
    adjusted_episodes.sort(key=lambda row: (row.model, row.task_id))
    return LengthControlFit(results, adjusted_episodes, selected, losses)


@dataclass(frozen=True)
class SotopiaCorrection:
    cohort: dict
    baseline_sources: list[SotopiaCell]
    cohort_sha256: str
    source_episodes: list[EpisodeScore]
    source_episodes_sha256: str
    scores: list[ModelScore]
    adjusted_episodes: list[AdjustedEpisodeScore]
    selected_regularization: float
    cross_validation_losses: dict[float, float]
    canonical_metrics: dict[str, float]


def correct_sotopia(
    trials: list[TrialResult], *, model: str, cohort: SotopiaCohort
) -> SotopiaCorrection:
    """Return corrected scores and the full inputs needed to reproduce the fit."""
    cohort_document = cohort.model_dump()
    sources = [SotopiaCell(row.model, row.results_path) for row in cohort.models if row.model != model]
    episodes = [episode for source in sources for episode in read_episodes(source)]
    candidate = []
    for trial in trials:
        category = (
            error_category(trial.exception_info.exception_type)
            if trial.exception_info
            else None
        )
        if category is not None and category not in (
            ErrorCategory.AGENT,
            ErrorCategory.PASSTHROUGH,
        ):
            continue
        rewards = trial.verifier_result.rewards if trial.verifier_result else None
        if rewards is None:
            if category is ErrorCategory.AGENT:
                candidate.append(EpisodeScore(model, trial.task_name, 0.0, None))
            continue
        overall = _normalized_overall(rewards)
        if overall is None:
            if category is ErrorCategory.AGENT:
                candidate.append(EpisodeScore(model, trial.task_name, 0.0, None))
                continue
            raise ValueError(f"SOTOPIA trial {trial.trial_name} lacks dimension scores")
        directory = trial.resolve_trial_path()
        paths = list(directory.glob("attempts/*/agent/sotopia-summary.json"))
        path = (
            max(paths, key=lambda item: int(item.parent.parent.name))
            if paths
            else directory / "agent/sotopia-summary.json"
        )
        summary = json.loads(path.read_text())
        if not math.isclose(_summary_overall(summary), overall, abs_tol=1e-9):
            raise ValueError(
                f"SOTOPIA summary disagrees with rewards for {trial.trial_name}"
            )
        candidate.append(
            EpisodeScore(
                model, trial.task_name, overall, _evaluated_response_characters(summary)
            )
        )
    if not candidate:
        raise ValueError("SOTOPIA has no scored candidate episodes")
    episodes.extend(candidate)
    fit = length_controlled_scores(episodes)
    selected = next(score for score in fit.scores if score.model == model)
    return SotopiaCorrection(
        cohort=cohort_document,
        baseline_sources=sources,
        cohort_sha256=hashlib.sha256(
            json.dumps(cohort_document, sort_keys=True).encode()
        ).hexdigest(),
        source_episodes=episodes,
        source_episodes_sha256=hashlib.sha256(
            json.dumps([asdict(row) for row in episodes], sort_keys=True).encode()
        ).hexdigest(),
        scores=fit.scores,
        adjusted_episodes=fit.episodes,
        selected_regularization=fit.selected_regularization,
        cross_validation_losses=fit.cross_validation_losses,
        canonical_metrics={
            "length_controlled_overall": selected.length_controlled_overall
        },
    )
