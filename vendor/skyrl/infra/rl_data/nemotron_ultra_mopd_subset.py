"""Bounded, routed samples of the Nemotron Ultra MOPD blend for Snowball MOPD runs.

Both sampling modes read the same seeded 1 MiB byte ranges of the blend instead of downloading
it. ``--rows-per-agent`` fills an equal quota per generator. ``--rows`` keeps routed rows in read
order, so the sample follows the blend's own generator mix; ``--swe-rows`` caps its Harbor SWE
slice. Math rows that NVIDIA ships as Hugging Face placeholders are restored from their source
datasets. SWE rows are bound to TaskTrove proxy tasks, and SWE states TaskTrove lacks are skipped.

Every generator is routed, so rows whose verifier needs a judge only train under
``environment.task_sessions.nemotron_ultra.grading: skip``.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import random
from collections import Counter
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import datasets
from skyrl_gym.envs.data_contracts import get_data_contract

from infra.rl_data.nemotron_ultra_sample import (
    HUGGING_FACE_RESOLVE_URL,
    RANGE_BYTES,
    RANGES_PER_BATCH,
    range_rows,
)
from infra.rl_data.nemotron_ultra_swe import (
    SWEProxyKey,
    bind_tasktrove_swe_proxies,
    load_tasktrove_swe_proxy_index,
    prepare_swe_task_artifact,
)
from infra.rl_data.sources import (
    NEMOTRON_ULTRA_REVISION,
    NEMOTRON_ULTRA_RL_DATASET,
    NEMOTRON_ULTRA_SWE_AGENT,
    TASKTROVE_PROXY_PATH_KEY,
    PreparedRow,
    load_nemotron_ultra_placeholder_sources,
    nemotron_ultra_mopd_source,
    restore_nemotron_ultra_placeholder,
)

MOPD_FILENAME = "mopd.jsonl"
MOPD_SIZE = 5_496_714_762
MAX_RANGES = 128
ROUTE_COLUMN = "teacher_route"
TRAIN_FILENAME = "train.parquet"
SWE_TASKS_DIRNAME = "swe-tasks"
MANIFEST_FILENAME = "manifest.json"
_PLACEHOLDER_KEY = "_hf_question_placeholder"

# Hardcoded generator -> Snowball expert routes. NVIDIA's own assignment is not public.
TEACHER_ROUTES: dict[str, str] = {
    "reasoning_gym_simple_agent": "math",
    "nvarc_inductive_simple_agent": "math",
    "nvarc_transductive_simple_agent": "math",
    "math_with_judge_simple_agent": "math",
    "ns_tools_simple_agent": "math",
    "math_formal_lean_refinement_agent": "math",
    "code_gen_simple_agent": "swe",
    "single_step_tool_use_with_argument_comparison_agent": "swe",
    "toolcall_schema_single_step_tool_use_with_argument_comparison_agent": "swe",
    NEMOTRON_ULTRA_SWE_AGENT: "swe",
    "instruction_following_simple_agent": "terminal",
    "structured_outputs_simple_agent": "terminal",
    "structured_outputs_v3_simple_agent": "terminal",
    "mcqa_simple_agent": "terminal",
    "calendar_simple_agent": "terminal",
    "citation_format_simple_agent": "terminal",
    "freeform_formatting_simple_agent": "terminal",
    "rdkit_chemistry_agent": "terminal",
    "abstention_simple_agent": "terminal",
    "multichallenge_simple_agent": "terminal",
    "jailbreak_engagement_with_disclaimer": "terminal",
    "jailbreak_hard_refusal_no_redirection": "terminal",
    "jailbreak_hard_refusal_with_helplines": "terminal",
    "jailbreak_refusal_with_explanation": "terminal",
    "genrm_simple_agent": "terminal",
    "genrm_simple_agent_reasoning_off": "terminal",
    "indirect_prompt_injection_simple_agent": "terminal",
}


def routed_agent(row: Mapping[str, Any], *, max_request_characters: int) -> str | None:
    """Return the routed generator of a raw blend row, or None when the row is excluded."""
    agent_ref = row.get("agent_ref")
    agent = agent_ref.get("name") if isinstance(agent_ref, Mapping) else None
    if agent not in TEACHER_ROUTES:
        return None
    request = row.get("responses_create_params")
    if not isinstance(request, Mapping) or len(json.dumps(request, ensure_ascii=False)) > max_request_characters:
        return None
    return agent


def _blend_rows(seed: int, revision: str, max_ranges: int) -> Iterator[dict[str, Any]]:
    """Yield blend rows from seeded 1 MiB ranges in a fixed order, one batch of reads at a time."""
    rng = random.Random(f"{seed}:mopd:{revision}")
    offsets = [rng.randrange(0, MOPD_SIZE - RANGE_BYTES) for _ in range(max_ranges)]
    url = HUGGING_FACE_RESOLVE_URL.format(revision=revision, filename=MOPD_FILENAME)
    for batch_start in range(0, len(offsets), RANGES_PER_BATCH):
        batch = offsets[batch_start : batch_start + RANGES_PER_BATCH]
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(batch)) as executor:
            for rows in executor.map(lambda offset: range_rows(url, offset), batch):
                yield from rows


def _routed_rows(
    *,
    seed: int,
    revision: str,
    max_ranges: int,
    swe_proxies: Mapping[SWEProxyKey, Mapping[str, Any]],
    max_request_characters: int,
) -> Iterator[tuple[str, dict[str, Any]]]:
    """Yield ``(generator, row)`` for routed rows, with SWE rows bound to TaskTrove proxies."""
    routed = (
        row
        for row in _blend_rows(seed, revision, max_ranges)
        if routed_agent(row, max_request_characters=max_request_characters) is not None
    )
    for row in bind_tasktrove_swe_proxies(routed, swe_proxies):
        yield row["agent_ref"]["name"], dict(row)


def sample_per_agent_rows(
    *,
    seed: int,
    rows_per_agent: int,
    swe_proxies: Mapping[SWEProxyKey, Mapping[str, Any]],
    max_request_characters: int,
    revision: str = NEMOTRON_ULTRA_REVISION,
    max_ranges: int = MAX_RANGES,
) -> list[dict[str, Any]]:
    """Collect up to ``rows_per_agent`` raw rows per routed generator with bounded transfer."""
    if rows_per_agent <= 0 or max_ranges <= 0:
        raise ValueError("rows_per_agent and max_ranges must be positive")
    candidates: dict[str, list[dict[str, Any]]] = {agent: [] for agent in TEACHER_ROUTES}
    routed = _routed_rows(
        seed=seed,
        revision=revision,
        max_ranges=max_ranges,
        swe_proxies=swe_proxies,
        max_request_characters=max_request_characters,
    )
    for agent, row in routed:
        if len(candidates[agent]) < rows_per_agent:
            candidates[agent].append(row)
        if all(len(agent_rows) >= rows_per_agent for agent_rows in candidates.values()):
            break
    return [row for agent in sorted(candidates) for row in candidates[agent]]


def sample_proportional_rows(
    *,
    seed: int,
    rows: int,
    swe_rows: int,
    swe_proxies: Mapping[SWEProxyKey, Mapping[str, Any]],
    max_request_characters: int,
    revision: str = NEMOTRON_ULTRA_REVISION,
    max_ranges: int = MAX_RANGES,
) -> list[dict[str, Any]]:
    """Keep routed rows in read order up to ``rows``, with at most ``swe_rows`` Harbor SWE rows."""
    if rows <= 0 or not 0 <= swe_rows <= rows or max_ranges <= 0:
        raise ValueError("rows and max_ranges must be positive and swe_rows must lie in [0, rows]")
    sample: list[dict[str, Any]] = []
    swe_count = 0
    routed = _routed_rows(
        seed=seed,
        revision=revision,
        max_ranges=max_ranges,
        swe_proxies=swe_proxies,
        max_request_characters=max_request_characters,
    )
    for agent, row in routed:
        if agent == NEMOTRON_ULTRA_SWE_AGENT:
            if swe_count == swe_rows:
                continue
            swe_count += 1
        sample.append(row)
        if len(sample) == rows:
            break
    return sample


def prepare_subset_rows(rows: list[dict[str, Any]]) -> list[PreparedRow]:
    """Turn raw rows into SkyRL rows carrying the hardcoded ``teacher_route`` column."""
    source = nemotron_ultra_mopd_source()
    contract = get_data_contract(source.env_id)
    prepared: list[PreparedRow] = []
    for row in rows:
        item = source.prepare_row(row, len(prepared), contract)
        item[ROUTE_COLUMN] = TEACHER_ROUTES[row["agent_ref"]["name"]]
        prepared.append(item)
    return prepared


def swe_proxy_paths(rows: list[dict[str, Any]]) -> set[str]:
    return {
        row["metadata"][TASKTROVE_PROXY_PATH_KEY] for row in rows if row["agent_ref"]["name"] == NEMOTRON_ULTRA_SWE_AGENT
    }


def subset_manifest(
    prepared: list[PreparedRow], *, seed: int, revision: str, sampling: Mapping[str, Any]
) -> dict[str, Any]:
    """Describe a sample; every route must have at least one row."""
    missing_routes = sorted(set(TEACHER_ROUTES.values()) - {row[ROUTE_COLUMN] for row in prepared})
    if missing_routes:
        raise RuntimeError(f"The sample has no rows for routes {missing_routes}")
    agents = Counter(row["extra_info"]["nemotron_ultra"]["agent"] for row in prepared)
    return {
        "dataset": NEMOTRON_ULTRA_RL_DATASET,
        "revision": revision,
        "blend": "mopd",
        "seed": seed,
        "sampling": dict(sampling),
        "rows": len(prepared),
        "rows_per_agent": dict(sorted(agents.items())),
        "rows_per_route": dict(sorted(Counter(row[ROUTE_COLUMN] for row in prepared).items())),
    }


def write_mopd_subset(
    output_dir: Path, prepared: list[PreparedRow], proxy_paths: set[str], manifest: dict[str, Any]
) -> dict[str, Any]:
    """Write ``train.parquet``, the SWE task artifact when SWE rows exist, and the manifest."""
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite MOPD subset: {output_dir}")
    output_dir.mkdir(parents=True)
    datasets.Dataset.from_list(prepared).to_parquet(str(output_dir / TRAIN_FILENAME))
    if proxy_paths:
        manifest["swe_tasks"] = prepare_swe_task_artifact(
            output_dir / SWE_TASKS_DIRNAME,
            desired_paths=proxy_paths,
            blend_files=(MOPD_FILENAME,),
            blend_revision=manifest["revision"],
        )
    (output_dir / MANIFEST_FILENAME).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory for the routed subset.")
    parser.add_argument("--seed", type=int, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--rows-per-agent", type=int, help="Equal quota per routed generator.")
    mode.add_argument("--rows", type=int, help="Total rows in blend proportion.")
    parser.add_argument("--swe-rows", type=int, default=0, help="Cap on Harbor SWE rows in --rows mode.")
    parser.add_argument(
        "--max-request-characters",
        type=int,
        required=True,
        help="Drop rows whose serialized request is longer; size it to the run's prompt budget.",
    )
    parser.add_argument("--max-ranges", type=int, default=MAX_RANGES, help="Bounded 1 MiB range reads.")
    args = parser.parse_args()

    swe_proxies = load_tasktrove_swe_proxy_index()
    bounds = {"max_request_characters": args.max_request_characters, "max_ranges": args.max_ranges}
    if args.rows is not None:
        rows = sample_proportional_rows(
            seed=args.seed, rows=args.rows, swe_rows=args.swe_rows, swe_proxies=swe_proxies, **bounds
        )
        sampling = {"mode": "proportional", "rows": args.rows, "swe_rows": args.swe_rows, **bounds}
    else:
        rows = sample_per_agent_rows(seed=args.seed, rows_per_agent=args.rows_per_agent, swe_proxies=swe_proxies, **bounds)
        sampling = {"mode": "per_agent", "rows_per_agent": args.rows_per_agent, **bounds}
    if any(_PLACEHOLDER_KEY in row for row in rows):
        placeholder_sources = load_nemotron_ultra_placeholder_sources()
        rows = [dict(restore_nemotron_ultra_placeholder(row, placeholder_sources)) for row in rows]

    prepared = prepare_subset_rows(rows)
    manifest = subset_manifest(prepared, seed=args.seed, revision=NEMOTRON_ULTRA_REVISION, sampling=sampling)
    print(json.dumps(write_mopd_subset(args.output_dir, prepared, swe_proxy_paths(rows), manifest), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
