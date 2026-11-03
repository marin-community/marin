"""Bounded random sampling of every generator in the released Ultra RLVR blends."""

from __future__ import annotations

import concurrent.futures
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
from skyrl_gym.envs.data_contracts import get_data_contract

from infra.rl_data.nemotron_ultra_swe import (
    SWEProxyKey,
    bind_tasktrove_swe_proxies,
    load_tasktrove_swe_proxy_index,
)
from infra.rl_data.sources import (
    NEMOTRON_ULTRA_REVISION,
    NEMOTRON_ULTRA_RL_DATASET,
    NEMOTRON_ULTRA_RLVR1_AGENTS,
    NEMOTRON_ULTRA_RLVR2_AGENTS,
    NEMOTRON_ULTRA_SWE_AGENT,
    Source,
    nemotron_ultra_rlvr1_source,
    nemotron_ultra_rlvr2_source,
)

HUGGING_FACE_RESOLVE_URL = (
    f"https://huggingface.co/datasets/{NEMOTRON_ULTRA_RL_DATASET}/resolve/{{revision}}/{{filename}}"
)
RANGE_BYTES = 1_048_576
RANGES_PER_BATCH = 24
MAX_RANGES = 256
MAX_INPUT_CHARACTERS = 60_000


@dataclass(frozen=True)
class BlendSampleSpec:
    name: str
    filename: str
    size: int
    agents: frozenset[str]
    source: Source


BLEND_SAMPLE_SPECS = (
    BlendSampleSpec(
        "rlvr1",
        "rlvr1.jsonl",
        5_040_426_486,
        NEMOTRON_ULTRA_RLVR1_AGENTS,
        nemotron_ultra_rlvr1_source(),
    ),
    BlendSampleSpec(
        "rlvr2",
        "rlvr2.jsonl",
        5_030_917_015,
        NEMOTRON_ULTRA_RLVR2_AGENTS,
        nemotron_ultra_rlvr2_source(),
    ),
)


def range_rows(url: str, start: int) -> list[dict[str, Any]]:
    response = requests.get(
        url,
        headers={"Range": f"bytes={start}-{start + RANGE_BYTES - 1}"},
        timeout=60,
    )
    response.raise_for_status()
    if response.status_code != 206 or len(response.content) > RANGE_BYTES:
        raise RuntimeError(
            f"Server ignored bounded JSONL range request: status={response.status_code}, bytes={len(response.content)}"
        )

    rows: list[dict[str, Any]] = []
    for line in response.content.splitlines()[1:-1]:
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def sample_raw_generator_rows(
    spec: BlendSampleSpec,
    *,
    seed: int,
    revision: str = NEMOTRON_ULTRA_REVISION,
    max_ranges: int = MAX_RANGES,
    swe_proxies: dict[SWEProxyKey, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Select one random embedded row for each generator with bounded transfer."""
    if max_ranges <= 0:
        raise ValueError("max_ranges must be positive")
    rng = random.Random(f"{seed}:{spec.name}:{revision}")
    offsets = [rng.randrange(0, spec.size - RANGE_BYTES) for _ in range(max_ranges)]
    url = HUGGING_FACE_RESOLVE_URL.format(revision=revision, filename=spec.filename)
    candidates: dict[str, list[dict[str, Any]]] = {agent: [] for agent in spec.agents}
    if NEMOTRON_ULTRA_SWE_AGENT in spec.agents and swe_proxies is None:
        swe_proxies = load_tasktrove_swe_proxy_index()

    for batch_start in range(0, len(offsets), RANGES_PER_BATCH):
        batch = offsets[batch_start : batch_start + RANGES_PER_BATCH]
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(batch)) as executor:
            chunks = executor.map(lambda offset: range_rows(url, offset), batch)
            for rows in chunks:
                for row in rows:
                    agent_ref = row.get("agent_ref")
                    agent = agent_ref.get("name") if isinstance(agent_ref, dict) else None
                    if agent is not None and agent not in candidates:
                        raise ValueError(f"{spec.name} contains unsupported generator {agent!r}")
                    if agent not in candidates or row.get("_hf_question_placeholder") is not None:
                        continue
                    if agent == NEMOTRON_ULTRA_SWE_AGENT:
                        bound = list(bind_tasktrove_swe_proxies([row], swe_proxies or {}))
                        if not bound:
                            continue
                        row = dict(bound[0])
                    request = row.get("responses_create_params")
                    if (
                        not isinstance(request, dict)
                        or len(json.dumps(request, ensure_ascii=False)) > MAX_INPUT_CHARACTERS
                    ):
                        continue
                    candidates[agent].append(row)
        if all(candidates.values()):
            return [rng.choice(candidates[agent]) for agent in sorted(spec.agents)]

    missing = sorted(agent for agent, rows in candidates.items() if not rows)
    raise RuntimeError(
        f"Could not sample every {spec.name} generator after {max_ranges} MiB of bounded reads; missing={missing}"
    )


def write_generator_sample(
    output_path: Path,
    *,
    seed: int,
    revision: str = NEMOTRON_ULTRA_REVISION,
) -> dict[str, Any]:
    """Write a heterogeneous parquet containing one live row per blend generator."""
    import datasets

    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite generator sample: {output_path}")
    prepared_rows = []
    sampled_uuids: dict[str, dict[str, str]] = {}
    swe_proxies = load_tasktrove_swe_proxy_index()
    for spec in BLEND_SAMPLE_SPECS:
        raw_rows = sample_raw_generator_rows(spec, seed=seed, revision=revision, swe_proxies=swe_proxies)
        contract = get_data_contract(spec.source.env_id)
        sampled_uuids[spec.name] = {}
        for index, row in enumerate(raw_rows):
            prepared = spec.source.prepare_row(row, index, contract)
            ultra = prepared["extra_info"]["nemotron_ultra"]
            sampled_uuids[spec.name][ultra["agent"]] = ultra["uuid"]
            prepared_rows.append(prepared)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    datasets.Dataset.from_list(prepared_rows).to_parquet(str(output_path))
    return {
        "dataset": NEMOTRON_ULTRA_RL_DATASET,
        "revision": revision,
        "seed": seed,
        "rows": len(prepared_rows),
        "samples": sampled_uuids,
    }
