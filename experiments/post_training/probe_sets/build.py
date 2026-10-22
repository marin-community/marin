# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build the probe-set prompts, the P_agent task list, and the reserved-benchmark file.

    uv run python -m experiments.post_training.probe_sets.build reserved --out DIR
    uv run python -m experiments.post_training.probe_sets.build sets --out DIR --seed N \\
        --stage0 FILE... --x-train DIR/x_train.jsonl --r-train DIR/r_train.jsonl \\
        --x-probe DIR/x_probe.jsonl --r-probe DIR/r_probe.jsonl

``reserved`` runs first: it writes every AIME 2024, AIME 2025, HMMT February 2025 and
OlympiadBench problem for the R task's exclusion. ``sets`` writes one JSONL file per
set, ``p_agent_harbor.yaml``, and ``manifest.json`` with sources, seeds, counts and
hashes. Reference continuations are recorded per architecture afterwards.
"""

from __future__ import annotations

import json
from pathlib import Path

import click
import yaml

from experiments.post_training.probe_sets.builder import (
    HARBOR_SWE_BENCH_DATASET,
    HARBOR_SWE_BENCH_REF,
    P_AGENT_CONTEXT_TOKENS,
    P_AGENT_GENERATION_TOKENS,
    PROMPT_SOURCES,
    T_TOKENS_PER_DOMAIN,
    ScreeningInputs,
    build_p_chat,
    build_p_think,
    build_t,
    build_task_probe,
    lines_sha256,
    p_agent_harbor_config,
    p_agent_instance_ids,
    require_pairwise_disjoint,
    source_entry,
    write_rows,
)
from experiments.post_training.probe_sets.sources import (
    competition_math,
    constraint_prompts,
    multi_turn_chats,
    pile_documents,
    swe_bench_instance_ids,
    task_probe_rows,
)
from experiments.post_training.task_data.expert_traces import family_tokenizer, token_counter

# T's token budget is measured with Snowball's tokenizer; the text is shared across architectures.
T_BUDGET_FAMILY = "grug"

_paths = click.Path(exists=True, dir_okay=False, path_type=Path)


@click.group()
def cli() -> None:
    pass


@cli.command()
@click.option("--out", "output_dir", required=True, type=click.Path(file_okay=False, path_type=Path))
def reserved(output_dir: Path) -> None:
    """Write every problem of the benchmarks P_think draws from, for the R task's exclusion."""
    output_dir.mkdir(parents=True, exist_ok=True)
    records = [
        {"problem": row.user_text(), "source": row.source} for rows in competition_math().values() for row in rows
    ]
    write_rows(output_dir / "reserved_benchmarks.jsonl", records)


@cli.command()
@click.option("--out", "output_dir", required=True, type=click.Path(file_okay=False, path_type=Path))
@click.option("--seed", required=True, type=int)
@click.option("--stage0", "stage0_files", required=True, multiple=True, type=_paths)
@click.option("--x-train", required=True, type=_paths)
@click.option("--r-train", required=True, type=_paths)
@click.option("--x-probe", required=True, type=_paths)
@click.option("--r-probe", required=True, type=_paths)
def sets(
    output_dir: Path,
    seed: int,
    stage0_files: tuple[Path, ...],
    x_train: Path,
    r_train: Path,
    x_probe: Path,
    r_probe: Path,
) -> None:
    """Select, screen and write the six probe sets with their manifest."""
    output_dir.mkdir(parents=True, exist_ok=True)
    inputs = ScreeningInputs(stage0_files=stage0_files, task_train_files=(x_train, r_train), x_train_file=x_train)
    p_chat, chat_counts = build_p_chat(constraint_prompts(), multi_turn_chats(), inputs, seed=seed)
    p_think, think_counts = build_p_think(competition_math(), inputs, seed=seed)
    p_x, x_counts = build_task_probe(task_probe_rows(x_probe, "task_X"), inputs, name="P_X", seed=seed, x_task_rows=True)
    p_r, r_counts = build_task_probe(
        task_probe_rows(r_probe, "task_R"), inputs, name="P_R", seed=seed, x_task_rows=False
    )
    prompt_sets = {"P_chat": p_chat, "P_think": p_think, "P_X": p_x, "P_R": p_r}
    require_pairwise_disjoint(prompt_sets)

    manifest: dict = {"seed": seed, "sets": {}}
    for name, rows in prompt_sets.items():
        digest = write_rows(output_dir / f"{name}.jsonl", [row.record() for row in rows])
        manifest["sets"][name] = {"rows": len(rows), "sha256": digest}
    manifest["sets"]["P_chat"] |= {"counts": chat_counts, "sources": [source_entry(s) for s in PROMPT_SOURCES["P_chat"]]}
    manifest["sets"]["P_think"] |= {
        "counts": think_counts,
        "sources": [source_entry(s) for s in PROMPT_SOURCES["P_think"]],
    }
    manifest["sets"]["P_X"] |= {"counts": x_counts, "source_file": str(x_probe)}
    manifest["sets"]["P_R"] |= {"counts": r_counts, "source_file": str(r_probe)}

    instance_ids = p_agent_instance_ids(swe_bench_instance_ids(), seed=seed)
    harbor = p_agent_harbor_config(instance_ids)
    (output_dir / "p_agent_harbor.yaml").write_text(yaml.safe_dump(harbor, sort_keys=False))
    manifest["sets"]["P_agent"] = {
        "rows": len(instance_ids),
        "sha256": lines_sha256({"instance_id": instance_id} for instance_id in instance_ids),
        "instance_ids": instance_ids,
        "harbor_dataset": HARBOR_SWE_BENCH_DATASET,
        "harbor_ref": HARBOR_SWE_BENCH_REF,
        "context_tokens": P_AGENT_CONTEXT_TOKENS,
        "generation_tokens": P_AGENT_GENERATION_TOKENS,
        "sources": [source_entry(s) for s in PROMPT_SOURCES["P_agent"]],
    }

    t_rows, t_tokens = build_t(pile_documents(), token_counter(family_tokenizer(T_BUDGET_FAMILY)))
    manifest["sets"]["T"] = {
        "rows": len(t_rows),
        "sha256": write_rows(output_dir / "T.jsonl", t_rows),
        "tokens_per_domain": t_tokens,
        "budget_per_domain": T_TOKENS_PER_DOMAIN,
        "budget_tokenizer_family": T_BUDGET_FAMILY,
        "sources": [source_entry(s) for s in PROMPT_SOURCES["T"]],
    }
    manifest["screening"] = {
        "stage0_files": [str(path) for path in stage0_files],
        "task_train_files": [str(x_train), str(r_train)],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    cli()
