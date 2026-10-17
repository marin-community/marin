# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Datasets for the mixed-domain RLVR pivot arms (#10214): RLVR1 alone, with selected pivots, and with random turns.

Each arm's rows carry a top-level ``teacher_route`` column naming their pool: ``rlvr``, ``swe`` or
``terminal``. MarinSkyRL's domain-weighted sampling (``data.sampling.kind=domain-weighted``) fills
every ``train_batch_size`` window from the pools in the proportions ``data.sampling.domain_weights``
gives, so one dataset serves every pivot share and :func:`domain_weights` sets the share at launch.
RLVR rows come first and the sampler shuffles the ``rlvr`` pool before the others, so arms that
list ``rlvr`` first and share a sampling seed draw RLVR prompts in the same order; arm A uses
domain-weighted sampling too, with ``rlvr`` alone.

The arm datasets pin their own version. Upstream pass-rate and selection artifacts were built at
version ``2026.10.09``; build at that version so they are reused: ``uv run python -m experiments.post_training.pivotrl.mixed_rlvr --version 2026.10.09``.
"""

import json
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import click
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from fray.types import ResourceConfig
from marin.execution.artifact import Artifact
from marin.execution.lazy import OUT, ArtifactStep, apply
from marin.execution.remote import remote
from marin.experiment.cli import build_options
from marin.experiment.data import hf_download
from marin.experiment.namespacing import user_owned_name
from marin.rl.nemotron_pivot import VALIDATION_FILENAME
from marin.rl.pass_rates import PASS_RATES_FILENAME
from rigging.filesystem.storage_path import StoragePath
from zephyr.writers import write_parquet_file

from experiments.post_training.pivotrl.pipeline import candidate_step, pivot_steps
from experiments.post_training.pivotrl.runs import GRUG_K16_LAMBDA_065_RUNS, SWE_CANDIDATES
from experiments.post_training.pivotrl.selection import TRAIN_FILENAME, count_columns

ROUTE_COLUMN = "teacher_route"
"""The column MarinSkyRL's domain-weighted sampling reads each row's pool from."""


class Route(StrEnum):
    """A pool of rows; ``data.sampling.domain_weights`` is keyed by these names."""

    RLVR = "rlvr"
    SWE = "swe"
    TERMINAL = "terminal"


MACHINE_AGENTS = (
    "code_gen_simple_agent",
    "math_formal_lean_refinement_agent",
    "ns_tools_simple_agent",
    "nvarc_inductive_simple_agent",
)
"""RLVR1 agents whose sessions execute code in a Docker task machine, which Iris task pods cannot start."""

RETIRED_ROUTE = "skyrl_gym"
TASK_SESSION_ROUTE = "task_session"


RLVR1 = hf_download(
    "raw/open-athena-nemotron-rl-ultra-rlvr1-cleaned-v3-2026-09-30",
    hf_id="open-athena/Nemotron-RL-Ultra-RLVR1-Cleaned-v3-2026-09-30",
    revision="600262dd37891b86075a80517e61dbfb105d7cbe",
    version="2026.10.16",
    urls_glob=("data/train.parquet", "data/validation.parquet"),
)


def _pool_rows(pool: dict[str, Any]) -> pa.Table:
    with (StoragePath(pool["path"]) / pool["filename"]).open("rb") as source:
        table = pq.read_table(source)
    if pool["filename"] == PASS_RATES_FILENAME:
        # Random turns are the candidates the frozen policy was scored on, without their counts.
        table = table.filter(pc.field("exclusion").is_null()).drop_columns(count_columns(table))
    if "exclude_agents" in pool:
        agents = pc.struct_field(pc.struct_field(table.column("extra_info"), "nemotron_ultra"), "agent")
        table = table.filter(pc.invert(pc.is_in(agents, pa.array(pool["exclude_agents"]))))
    if "max_output_tokens" in pool:
        table = _with_output_cap(table, pool["max_output_tokens"])
    return table


def _with_output_cap(rows: pa.Table, max_output_tokens: int) -> pa.Table:
    """Cap each row's reply through its request, which MarinSkyRL's sessions pass to the model per row."""
    extra = rows.column("extra_info").combine_chunks()
    ultra = pc.struct_field(extra, "nemotron_ultra")
    requests = pc.struct_field(ultra, "request_json")
    capped = [
        json.dumps({**json.loads(request), "max_output_tokens": max_output_tokens}) for request in requests.to_pylist()
    ]
    ultra = _replace_field(ultra, "request_json", pa.array(capped, requests.type))
    return rows.set_column(
        rows.column_names.index("extra_info"), "extra_info", _replace_field(extra, "nemotron_ultra", ultra)
    )


def _with_task_session_route(rows: pa.Table) -> pa.Table:
    """Name MarinSkyRL's task-session route on rows written before it replaced ``skyrl_gym``.

    Both RLVR1 and #9933's candidates predate MarinSkyRL #879, which grades every Nemotron Ultra
    row in a task session and rejects any other route.
    """
    extra = rows.column("extra_info").combine_chunks()
    ultra = pc.struct_field(extra, "nemotron_ultra")
    route = pc.struct_field(ultra, "route")
    unknown = set(pc.unique(route).to_pylist()) - {RETIRED_ROUTE, TASK_SESSION_ROUTE}
    if unknown:
        raise ValueError(f"unexpected Nemotron Ultra routes: {sorted(unknown)}")
    route = pc.if_else(pc.equal(route, RETIRED_ROUTE), pa.scalar(TASK_SESSION_ROUTE, route.type), route)
    ultra = _replace_field(ultra, "route", route)
    return rows.set_column(
        rows.column_names.index("extra_info"), "extra_info", _replace_field(extra, "nemotron_ultra", ultra)
    )


def _replace_field(struct: pa.StructArray, name: str, values: pa.Array) -> pa.StructArray:
    names = [field.name for field in struct.type]
    children = [values if child == name else pc.struct_field(struct, child) for child in names]
    return pa.StructArray.from_arrays(children, names=names, mask=struct.is_null())


def _has_empty_message(prompt: list[dict[str, Any]]) -> bool:
    """Whether a text message is blank; MarinSkyRL's task conversations reject them."""
    return any(
        not (message["content"] or "").strip()
        for message in prompt
        if message["role"] != "tool" and not (message["role"] == "assistant" and message.get("tool_calls"))
    )


def _without_empty_messages(rows: pa.Table) -> pa.Table:
    keep = [not _has_empty_message(prompt) for prompt in rows.column("prompt").to_pylist()]
    return rows.filter(pa.array(keep))


def _routed(pools: dict[str, dict[str, Any]]) -> tuple[pa.Table, dict[str, dict[str, int]]]:
    tables = {route: _pool_rows(pool) for route, pool in pools.items()}
    counts = {route: {"rows": table.num_rows} for route, table in tables.items()}
    tables = {route: _without_empty_messages(table) for route, table in tables.items()}
    for route, table in tables.items():
        counts[route]["dropped_empty_message"] = counts[route]["rows"] - table.num_rows
        counts[route]["rows"] = table.num_rows
    rows = _with_task_session_route(pa.concat_tables(tables.values(), promote_options="permissive"))
    routes = pa.array([route for route, table in tables.items() for _ in range(table.num_rows)], pa.string())
    return rows.append_column(ROUTE_COLUMN, routes), counts


def mix_routes(*, train: dict[str, dict[str, Any]], validation: dict[str, dict[str, Any]], output_path: str) -> None:
    """Write ``train.parquet`` and ``validation.parquet``, each its pools' rows tagged by route, plus counts.

    Pools are plain dicts so the executor resolves the artifact steps inside them.

    Args:
        train: Pools by route, each ``{"path": artifact, "filename": file}``, in row order. A
            ``pass_rates.parquet`` pool keeps the rows its policy was scored on, a pool with
            ``exclude_agents`` drops those Nemotron Ultra agents' rows, and a pool with
            ``max_output_tokens`` caps its rows' replies at that many tokens.
        validation: Validation pools in the same form.
        output_path: Where the two files and ``manifest.json`` go.
    """
    output = StoragePath(output_path)
    manifest = {}
    for split, pools, filename in (("train", train, TRAIN_FILENAME), ("validation", validation, VALIDATION_FILENAME)):
        rows, counts = _routed(pools)
        write_parquet_file(rows.to_batches(), str(output / filename), schema=rows.schema)
        manifest[split] = counts
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def domain_weights(pivot_share: float, *, batch_size: int = 512) -> dict[Route, int]:
    """Prompts per route in every batch: ``pivot_share`` of it split evenly between SWE and Terminal.

    The two benchmarks weigh equally in #10214's decision rules, so the split ignores pool sizes.
    """
    if not 0 <= pivot_share < 1:
        raise ValueError("pivot_share must be in [0, 1)")
    pivots = round(pivot_share * batch_size)
    weights = {Route.RLVR: batch_size - pivots, Route.SWE: (pivots + 1) // 2, Route.TERMINAL: pivots // 2}
    return {route: count for route, count in weights.items() if count}


def _capped(pool: dict[str, Any], max_output_tokens: int | None) -> dict[str, Any]:
    return pool if max_output_tokens is None else {**pool, "max_output_tokens": max_output_tokens}


def arm_datasets() -> dict[str, ArtifactStep[Artifact]]:
    """One dataset per kind of arm: ``rlvr1``, ``selected`` pivots and ``random`` candidate turns."""
    runs = {Route.SWE if run.candidates == SWE_CANDIDATES else Route.TERMINAL: run for run in GRUG_K16_LAMBDA_065_RUNS}
    steps = {route: pivot_steps(run) for route, run in runs.items()}
    rlvr = {Route.RLVR: {"path": RLVR1, "filename": "data/train.parquet", "exclude_agents": MACHINE_AGENTS}}
    validation = {
        Route.RLVR: {"path": RLVR1, "filename": "data/validation.parquet", "exclude_agents": MACHINE_AGENTS},
        **{
            route: {"path": candidate_step(run.candidates), "filename": VALIDATION_FILENAME}
            for route, run in runs.items()
        },
    }
    # Pivot replies get the cap their pass rates were sampled at.
    caps = {route: run.sampling.max_tokens for route, run in runs.items()}
    validation = {route: _capped(pool, caps.get(route)) for route, pool in validation.items()}
    pivots = {
        "selected": {
            route: _capped({"path": step["pivots"], "filename": TRAIN_FILENAME}, caps[route])
            for route, step in steps.items()
        },
        "random": {
            route: _capped({"path": step["pass-rates"], "filename": PASS_RATES_FILENAME}, caps[route])
            for route, step in steps.items()
        },
    }
    return {
        name: apply(
            user_owned_name(f"pivotrl/mixed-rlvr/{name}"),
            remote(mix_routes, resources=ResourceConfig.with_cpu(cpu=4, ram="64g")),
            version="2026.10.16.3",
            train={**rlvr, **pools},
            validation=validation,
            output_path=OUT,
        )
        for name, pools in {"rlvr1": {}, "selected": pivots["selected"], "random": pivots["random"]}.items()
    }


@dataclass(frozen=True)
class Arm:
    dataset: str
    """Key in :func:`arm_datasets`."""
    pivot_share: float


ARMS = {
    "A": Arm("rlvr1", 0.0),
    "B": Arm("selected", 0.2),
    "C": Arm("random", 0.2),
    "B-10": Arm("selected", 0.1),
    "B-30": Arm("selected", 0.3),
}
"""B-10 and B-30 are the optional arms that move the pivot share."""


@click.command(help=__doc__)
@build_options
def main() -> dict[str, ArtifactStep[Artifact]]:
    for name, arm in ARMS.items():
        weights = {str(route): count for route, count in domain_weights(arm.pivot_share).items()}
        click.echo(f"{name}: {arm.dataset}, domain_weights={weights}")
    return arm_datasets()


if __name__ == "__main__":
    main()
