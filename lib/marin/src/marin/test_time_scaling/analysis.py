# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections import defaultdict

from marin.test_time_scaling.budgets import budget_totals_from_candidates
from marin.test_time_scaling.config import SelectorName, TestTimeScalingConfig
from marin.test_time_scaling.manifests import PromptManifest
from marin.test_time_scaling.results import CandidateRecord, RunSummary, SelectionRecord, SelectorSummary
from marin.test_time_scaling.selectors import select_candidate


def group_candidates_by_prompt(candidates: list[CandidateRecord]) -> dict[str, list[CandidateRecord]]:
    """Group generated candidates by prompt id."""

    grouped_candidates: dict[str, list[CandidateRecord]] = defaultdict(list)
    for candidate in candidates:
        grouped_candidates[candidate.prompt_id].append(candidate)
    for prompt_id in grouped_candidates:
        grouped_candidates[prompt_id].sort(key=lambda candidate: candidate.sample_index)
    return dict(grouped_candidates)


def replay_selectors(
    candidates: list[CandidateRecord],
    selectors: tuple[SelectorName, ...],
) -> list[SelectionRecord]:
    """Replay selectors against a saved candidate pool."""

    selections: list[SelectionRecord] = []
    for prompt_id, prompt_candidates in group_candidates_by_prompt(candidates).items():
        oracle_values = [candidate.is_correct for candidate in prompt_candidates if candidate.is_correct is not None]
        oracle_correct = any(oracle_values) if oracle_values else None
        for selector_name in selectors:
            chosen_candidate = select_candidate(prompt_candidates, selector_name)
            correctness = chosen_candidate.is_correct
            oracle_gap = None
            if oracle_correct is not None and correctness is not None:
                oracle_gap = oracle_correct and not correctness
            selections.append(
                SelectionRecord(
                    prompt_id=prompt_id,
                    selector_name=selector_name,
                    chosen_candidate_id=chosen_candidate.candidate_id,
                    chosen_sample_index=chosen_candidate.sample_index,
                    final_selected_answer=chosen_candidate.extracted_answer or chosen_candidate.raw_text,
                    correctness=correctness,
                    oracle_correct=oracle_correct,
                    oracle_gap=oracle_gap,
                )
            )
    return selections


def build_run_summary(
    manifest: PromptManifest,
    run_config: TestTimeScalingConfig,
    candidates: list[CandidateRecord],
    selections: list[SelectionRecord],
) -> RunSummary:
    """Build the top-level run summary from saved candidates and selections."""

    budget_totals = budget_totals_from_candidates(candidates)
    parse_values = [candidate.parse_valid for candidate in candidates if candidate.parse_valid is not None]
    parse_valid_rate = sum(parse_values) / len(parse_values) if parse_values else None

    duplicate_count = 0
    for prompt_candidates in group_candidates_by_prompt(candidates).values():
        unique_texts = {candidate.raw_text.strip() for candidate in prompt_candidates}
        duplicate_count += max(0, len(prompt_candidates) - len(unique_texts))
    duplicate_rate = duplicate_count / len(candidates) if candidates else None

    oracle_by_prompt: dict[str, bool] = {}
    for selection in selections:
        if selection.oracle_correct is None or selection.prompt_id in oracle_by_prompt:
            continue
        oracle_by_prompt[selection.prompt_id] = selection.oracle_correct
    oracle_values = list(oracle_by_prompt.values())
    oracle_accuracy = sum(oracle_values) / len(oracle_values) if oracle_values else None

    selector_summaries: list[SelectorSummary] = []
    for selector_name in run_config.selectors:
        selector_rows = [selection for selection in selections if selection.selector_name == selector_name]
        scored_values = [selection.correctness for selection in selector_rows if selection.correctness is not None]
        gap_values = [selection.oracle_gap for selection in selector_rows if selection.oracle_gap is not None]
        accuracy = sum(scored_values) / len(scored_values) if scored_values else None
        oracle_gap_rate = sum(gap_values) / len(gap_values) if gap_values else None
        selector_summaries.append(
            SelectorSummary(
                selector_name=selector_name,
                num_prompts=len(selector_rows),
                num_scored_prompts=len(scored_values),
                accuracy=accuracy,
                oracle_gap_rate=oracle_gap_rate,
            )
        )

    return RunSummary(
        manifest_id=manifest.manifest_id,
        task_name=manifest.task_name,
        num_prompts=len(manifest.records),
        total_candidates=len(candidates),
        oracle_accuracy=oracle_accuracy,
        parse_valid_rate=parse_valid_rate,
        duplicate_rate=duplicate_rate,
        total_prompt_tokens=budget_totals.total_prompt_tokens,
        total_completion_tokens=budget_totals.total_completion_tokens,
        total_request_latency_seconds=budget_totals.total_request_latency_seconds,
        selector_summaries=tuple(selector_summaries),
        metadata={
            "generation": {
                "num_candidates": run_config.generation.num_candidates,
                "temperature": run_config.generation.temperature,
                "top_p": run_config.generation.top_p,
                "max_gen_toks": run_config.generation.max_gen_toks,
                "seed": run_config.generation.seed,
            }
        },
    )
