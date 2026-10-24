# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build the probe-set prompts, the P_agent task list, and the reserved-benchmark file.

    uv run python -m experiments.post_training.probe_sets.build reserved --out DIR
    uv run python -m experiments.post_training.probe_sets.build unscreened --out DIR --seed N
    uv run python -m experiments.post_training.probe_sets.build screened --out DIR --seed N \\
        --stage0 FILE... --x-train DIR/x_train.jsonl --r-train DIR/r_train.jsonl \\
        --x-probe DIR/x_probe.jsonl --r-probe DIR/r_probe.jsonl

``reserved`` runs first: it writes every AIME 2024, AIME 2025, HMMT February 2025 and
OlympiadBench problem for the R task's exclusion. ``unscreened`` writes the sets no
overlap screen applies to: the P_agent task list with ``p_agent_harbor.yaml``, and T.
``screened`` writes P_chat, P_think, P_X and P_R, which are screened against Stage 0
and the task training data, so it waits for both. Each writes one JSONL file per set
and merges its sets into ``manifest.json`` (sources, seeds, counts and hashes); both
must use the same seed. Reference continuations are recorded per architecture
afterwards.
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
    STAGE0_CONTAMINATION_SHARE,
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
    screening_hits,
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


def _merge_manifest(output_dir: Path, seed: int, sets: dict, extra: dict) -> None:
    """Merge ``sets`` and ``extra`` into ``output_dir/manifest.json``; raise on a seed mismatch."""
    path = output_dir / "manifest.json"
    manifest: dict = json.loads(path.read_text()) if path.exists() else {"seed": seed, "sets": {}}
    if manifest["seed"] != seed:
        raise ValueError(f"{path} was built with seed {manifest['seed']}, not {seed}")
    manifest["sets"] |= sets
    manifest |= extra
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


@cli.command()
@click.option("--out", "output_dir", required=True, type=click.Path(file_okay=False, path_type=Path))
@click.option("--seed", required=True, type=int)
@click.option("--stage0", "stage0_files", required=True, multiple=True, type=_paths)
@click.option("--x-train", required=True, type=_paths)
@click.option("--r-train", required=True, type=_paths)
@click.option("--x-probe", required=True, type=_paths)
@click.option("--r-probe", required=True, type=_paths)
def screened(
    output_dir: Path,
    seed: int,
    stage0_files: tuple[Path, ...],
    x_train: Path,
    r_train: Path,
    x_probe: Path,
    r_probe: Path,
) -> None:
    """Select, screen and write P_chat, P_think, P_X and P_R."""
    output_dir.mkdir(parents=True, exist_ok=True)
    inputs = ScreeningInputs(stage0_files=stage0_files, task_train_files=(x_train, r_train), x_train_file=x_train)
    constraints, chats, benchmarks = constraint_prompts(), multi_turn_chats(), competition_math()
    x_rows, r_rows = task_probe_rows(x_probe, "task_X"), task_probe_rows(r_probe, "task_R")
    candidates = [*constraints, *chats, *(row for rows in benchmarks.values() for row in rows), *x_rows, *r_rows]
    hits = screening_hits(candidates, inputs)
    p_chat, chat_counts = build_p_chat(constraints, chats, hits, seed=seed)
    p_think, think_counts = build_p_think(benchmarks, hits, seed=seed)
    p_x, x_counts = build_task_probe(x_rows, hits, name="P_X", seed=seed, x_task_rows=True)
    p_r, r_counts = build_task_probe(r_rows, hits, name="P_R", seed=seed, x_task_rows=False)
    prompt_sets = {"P_chat": p_chat, "P_think": p_think, "P_X": p_x, "P_R": p_r}
    require_pairwise_disjoint(prompt_sets)

    sets: dict = {}
    for name, rows in prompt_sets.items():
        digest = write_rows(output_dir / f"{name}.jsonl", [row.record() for row in rows])
        sets[name] = {"rows": len(rows), "sha256": digest}
    sets["P_chat"] |= {"counts": chat_counts, "sources": [source_entry(s) for s in PROMPT_SOURCES["P_chat"]]}
    sets["P_think"] |= {"counts": think_counts, "sources": [source_entry(s) for s in PROMPT_SOURCES["P_think"]]}
    sets["P_X"] |= {"counts": x_counts, "source_file": str(x_probe)}
    sets["P_R"] |= {"counts": r_counts, "source_file": str(r_probe)}
    sources = {row.probe_id: row.source for row in candidates}
    screening = {
        "stage0_files": [str(path) for path in stage0_files],
        "task_train_files": [str(x_train), str(r_train)],
        "stage0_contamination_share": STAGE0_CONTAMINATION_SHARE,
        # Every candidate sharing any 13-gram with Stage 0; contaminated ones were dropped.
        "stage0_matches": [
            {
                "probe_id": probe_id,
                "source": sources[probe_id],
                "matched_share": round(share, 4),
                "contaminated": share >= STAGE0_CONTAMINATION_SHARE,
            }
            for probe_id, share in sorted(hits.stage0_shares.items())
        ],
    }
    _merge_manifest(output_dir, seed, sets, {"screening": screening})


@cli.command()
@click.option("--out", "output_dir", required=True, type=click.Path(file_okay=False, path_type=Path))
@click.option("--seed", required=True, type=int)
def unscreened(output_dir: Path, seed: int) -> None:
    """Write the P_agent task list with its Harbor config, and T."""
    output_dir.mkdir(parents=True, exist_ok=True)
    instance_ids = p_agent_instance_ids(swe_bench_instance_ids(), seed=seed)
    harbor = p_agent_harbor_config(instance_ids)
    (output_dir / "p_agent_harbor.yaml").write_text(yaml.safe_dump(harbor, sort_keys=False))
    sets: dict = {
        "P_agent": {
            "rows": len(instance_ids),
            "sha256": lines_sha256({"instance_id": instance_id} for instance_id in instance_ids),
            "instance_ids": instance_ids,
            "harbor_dataset": HARBOR_SWE_BENCH_DATASET,
            "harbor_ref": HARBOR_SWE_BENCH_REF,
            "context_tokens": P_AGENT_CONTEXT_TOKENS,
            "generation_tokens": P_AGENT_GENERATION_TOKENS,
            "sources": [source_entry(s) for s in PROMPT_SOURCES["P_agent"]],
        }
    }
    t_rows, t_tokens = build_t(pile_documents(), token_counter(family_tokenizer(T_BUDGET_FAMILY)))
    sets["T"] = {
        "rows": len(t_rows),
        "sha256": write_rows(output_dir / "T.jsonl", t_rows),
        "tokens_per_domain": t_tokens,
        "budget_per_domain": T_TOKENS_PER_DOMAIN,
        "budget_tokenizer_family": T_BUDGET_FAMILY,
        "sources": [source_entry(s) for s in PROMPT_SOURCES["T"]],
    }
    _merge_manifest(output_dir, seed, sets, {})


if __name__ == "__main__":
    cli()
