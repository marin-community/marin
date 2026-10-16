# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build the dataset release: `data/<config>/<table>.csv|parquet`, `MANIFEST.json` and `figures_tables.csv`.

`uv run python -m experiments.domain_phase_mix.release.export_dataset [--out release_build] [--private-dir DIR]`

`build_tables` runs every exporter once. `write` replaces the build's data folder, writes each table (Parquet for the
`fits` tables, the StarCoder dense grid and the trajectories; CSV with pandas' shortest round-trip float repr for the
rest, so every value is bit-identical to the exported one), copies figures_tables.csv and writes the public manifest:
per table its config, path, format, rows, columns, bytes, SHA-256, where its observation ids (W&B run ids) come from
(`keying`), notes and repository-relative sources, and the cross-table consistency problems. SOURCES.json (every
table's absolute source paths) goes to the private directory, which must lie outside the build. The command exits
non-zero, after writing the manifest, when two tables disagree about a run.
"""

import argparse
import hashlib
import json
import re
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from experiments.domain_phase_mix.release import checks, context, naming, paths
from experiments.domain_phase_mix.release.configs import (
    domain_sweeps,
    fits,
    heldout,
    ladder,
    partition_tasks,
    starcoder,
    swarm_llama,
    swarm_qwen3,
    validation,
)

FIGURES_TABLES = paths.REPO / "experiments/domain_phase_mix/release/figures_tables.csv"
MANIFEST = "MANIFEST.json"
SOURCES = "SOURCES.json"  # private
# Configs whose tables share one folder: every `starcoder_*`, `fits_*` and `domain_sweeps_*` table.
SHARED_FOLDERS = ("starcoder", "fits", "domain_sweeps")
PARQUET_TABLES = ("starcoder_dense_grid", "starcoder_trajectories")  # and every `fits_*` table
# The exporter behind each config, whose `required_sources()` are the config's sources; `run_sets` has none.
EXPORTER = {
    "partition": partition_tasks,
    "tasks": partition_tasks,
    "swarm_qwen3_360m_1p6b": swarm_qwen3,
    "swarm_llama_160m_1p2b": swarm_llama,
    "swarm_llama_200m_6b": swarm_llama,
    "heldout": heldout,
    "validation_3e18": validation,
    "ladder": ladder,
    "ladder_accuracy": ladder,
    "ladder_compute_equivalence": ladder,
    "ladder_unimax_cap_sweep": ladder,
    "starcoder": starcoder,
    "domain_sweeps": domain_sweeps,
    "fits": fits,
}
# The public manifest shows `*` for a folder named by a 64-hex content hash.
HASH_FOLDER = re.compile(r"(?<=/)[0-9a-f]{64}(?=/)")
MIXSEL_PREFIX = "mixture-selection/"

NO_IDS = "The table has no observation ids."
# Where each table's observation ids, W&B run ids, come from; one training run has one id wherever it appears.
KEYING = {
    "partition": NO_IDS,
    "tasks": NO_IDS,
    "swarm_qwen3_360m_1p6b": (
        "Each run's W&B run id: its training run name with its six-hex-digit hash. The 42 phase-constant fit rows (3 "
        "baselines, 39 deletions) take the id of the two-phase swarm's run that trained the mixture in both phases."
    ),
    "swarm_llama_160m_1p2b": "Each run's W&B run id: its training run name with its six-hex-digit hash.",
    "swarm_llama_200m_6b": "Each run's W&B run id: its training run name with its six-hex-digit hash.",
    "heldout": (
        "Each run's W&B run id: the training run name with its six-hex-digit hash that the registry or a validation "
        "launch records, or for the 134 Llama 200M runs, which the registry records without one, the id read from W&B. "
        "W&B shortened 20 of those ids by dropping leading characters of the run name."
    ),
    "validation_3e18": (
        "Each run's W&B run id: its training run name with its six-hex-digit hash, from its evaluation-metrics path or "
        "its launch's record of run names."
    ),
    "ladder": (
        "Each run's W&B run id: its training run name with its six-hex-digit hash; the 3e18 copies keep the ids of "
        "their validation_3e18 and swarm_qwen3_360m_1p6b rows."
    ),
    "ladder_accuracy": "The ladder's ids of the four scored 1e21 runs.",
    "ladder_compute_equivalence": NO_IDS,
    "ladder_unimax_cap_sweep": NO_IDS,
    "starcoder_observations": (
        "Each atlas run's W&B run id: its training run name, after the prefix c2v4c_ for the 78 coupled "
        "learning-rate-onset runs, whose launch put it before every W&B run id."
    ),
    "starcoder_curve_membership": "The starcoder_observations ids of each curve's runs.",
    "starcoder_curves": NO_IDS,
    "starcoder_fit_parameters": NO_IDS,
    "starcoder_dense_grid": NO_IDS,
    "starcoder_diagonal": "Each run's W&B run id, its training run name.",
    "starcoder_curricula": NO_IDS,
    "starcoder_curricula_matched_nd": NO_IDS,
    "starcoder_curricula_matched_nd_confirmation": "The W&B run ids of each pair's two runs.",
    "starcoder_transition": (
        "Each run's W&B run id: its training run name, or for a gradient-conflict run gcfv9_ and its trajectory id."
    ),
    "starcoder_ramp_designs": "The W&B run ids of each ramp run, its abrupt source run and its tied comparator.",
    "starcoder_trajectories": (
        "Each run's W&B run id: its training run name, or for a gradient-conflict run gcfv9_ and its trajectory id."
    ),
    "starcoder_repeat_noise": "Each run's W&B run id: its training run name with its six-hex-digit hash.",
    "domain_sweeps": "Each run's W&B run id: its training run name, to which these launches append no hash.",
    "domain_sweeps_olmix_fits": NO_IDS,
    "fits_task_parameters": NO_IDS,
    "fits_swarm_predictions": "The ids of the fit rows of swarm_qwen3_360m_1p6b.",
    "fits_heldout_predictions": "The heldout ids of every held-out run at each predicted registry coordinate.",
    "fits_reliability": NO_IDS,
    "fits_learning_curves": NO_IDS,
    "fits_proposal_predictions": "The validation_3e18 ids of every run of each predicted proposal's mixture.",
    "fits_olmix_task_laws": NO_IDS,
    "fits_surrogate_benchmark": NO_IDS,
    "run_sets": "The ids of the run tables.",
}
NOTES = {
    "swarm_qwen3_360m_1p6b": [
        "The ten proportional repeats (design_kind repeat, fit_inclusion false) pair their OlmoBaseEval Easy and "
        "Uncheatable components by run letter: one source indexes the runs by letter, the other by run name, and the "
        "noise panel's pooled Uncheatable BPB of each letter confirms the pairing.",
        "The ten proportional repeats are also rows of the ladder's 3e18 rung.",
    ],
    "swarm_llama_160m_1p2b": [
        "The 39 single-bucket deletions (design_kind deletion) are fit rows here and frozen held-out runs in heldout, "
        "under the same observation ids; pool the two tables by observation_id.",
        "No launch manifest of the deletions is among the sources, so their seeds are null.",
    ],
    "swarm_llama_200m_6b": [
        "The proportional fit row (run_label baseline_proportional) keeps its own run's values. The parity panel "
        "behind the swarm's fits, and the July packet copied from it, replaced them by the mean of 11 proportional "
        "runs: the ten repeats here and the two-phase swarm's proportional run, which this release omits.",
        "The stratified baseline is a two-phase swarm run whose launch left no manifest among the sources, so its "
        "seeds are null.",
    ],
    "heldout": [
        "The 39 Llama 160M single-bucket deletions are also fit rows of swarm_llama_160m_1p2b, and 117 Qwen3 runs "
        "(the 116 expanded-cohort validation runs and the UniMax-8 seed-0 run) are also rows of validation_3e18; each "
        "such run has one observation id and equal BPB, weights, pool fractions and seeds in both tables (labels may "
        "differ).",
        "Ten frozen archive runs, in four exposure groups, are deterministic re-trainings: their aggregates are "
        "bit-identical within each group.",
        "628 of the 929 runs (455 Qwen3, 117 Llama 200M and 56 Llama 160M runs) have all 58 per-task values.",
        "One archived Llama 160M run, which the registry scores by its pooled Uncheatable aggregate only, takes its "
        "seven Uncheatable components and its data seed from its W&B run summary and config; it logged no OlmoBaseEval "
        "Easy metrics, so those values are null.",
        "The 134 Llama 200M runs take their W&B run ids and seeds from W&B, read on 2026-10-15, as the registry records "
        "neither.",
        "proposal_objective uses the public objective names: none for the UniMax-8 baseline (epoch cap 8, as in "
        "validation_3e18), uncheatable_and_obe_easy for designs built for both objectives, and other where the "
        "registry's run names name no objective; the Llama panels leave it null.",
    ],
    "validation_3e18": [
        "UniMax-8 trainer seeds 1 and 2 carry recorded aggregates only; their per-task values are not among the "
        "sources.",
        "117 runs are also rows of heldout, and the ladder's 3e18 rung copies 15: trainer seeds 0 to 2 of MARINER's "
        "and of the matched Olmix proposals for each objective, and the UniMax-8 repeats.",
    ],
    "ladder": [
        "The 3e18 rung copies 15 validation_3e18 rows and the ten proportional repeats of swarm_qwen3_360m_1p6b "
        "under their observation ids, and adds the June proportional run, which no other table holds; the scaling "
        "figure averages these runs at 3e18.",
        "UniMax-8 trainer seeds 1 and 2 at 3e18 carry recorded aggregates only. The proportional and UniMax-8 runs "
        "at 2e19 and 3e20 have Uncheatable components only: their Uncheatable aggregate is recomputed and their "
        "OlmoBaseEval Easy mean is the recorded one (components_available false).",
    ],
    "ladder_unimax_cap_sweep": [
        "The UniMax cap sweep's runs at caps 1, 4 and 12, at the 3e18 rung's model size and token count; its cap-8 "
        "and proportional runs are ladder rows. The runs take their seven Uncheatable components from their W&B run "
        "summaries; the data seed in each run's W&B config equals the one its launch recorded.",
    ],
    "domain_sweeps": [
        "These models use the TinyLlama tokenizer named in `tokenizer`, not the tokenizer of the tasks table, which "
        "describes the paper's swarms.",
    ],
    "fits_task_parameters": [
        "The qwen3_360m_1p6b rows are the paper's reference fit; the llama_160m_1p2b and llama_200m_6b rows refit "
        "MARINER with mixture-selection on those swarms' fit rows, with anchors from each swarm's ten proportional "
        "repeats and the reference fit's inner-fold rule.",
    ],
    "fits_swarm_predictions": [
        "In-sample fitted values of the reference fit, which trained on these runs. True out-of-fold predictions "
        "are not exported; fits_learning_curves and fits_reliability hold the out-of-fold metrics.",
    ],
    "fits_learning_curves": ["The out-of-fold predictions behind these metrics are not exported."],
}
RELEASE_NOTES = [
    "observation_id is a training run's W&B run id in the public project marin-community/marin, whose logs are at "
    "wandb.ai/marin-community/marin/runs/<observation_id>. One training run has one id in every table that holds it, "
    "and run_sets lists its memberships, so pool tables by observation_id.",
    "mixture_id hashes a row's weights rounded to 1e-9; exposure_id hashes its weights, pool fractions and the "
    "partition revision.",
    "bpb__uncheatable (the seven Uncheatable corpora weighted by the tasks table's aggregate_weight, the paper's fixed "
    "byte shares) and bpb__obe_easy (the mean of the 51 OlmoBaseEval Easy tasks) are computed from a run's components "
    "whenever it has all of that objective's components, and otherwise are the aggregate its source recorded under "
    "the same weights, or null where the source recorded none under those weights.",
    "Source paths are relative to the marin repository's root, those under mixture-selection/ to the "
    "mixture-selection repository's; `*` stands for a hash that names a folder.",
    "figures_tables.csv lists the paper's 26 figures and 20 tables with their builders (paths in the paper folder, or "
    "in the repository under experiments/ and .agents/) and the configs they read: deferred marks two-phase data "
    "this release omits, and constant marks configuration constants typed into the paper.",
]


def config_of(table: str) -> str:
    for config in SHARED_FOLDERS:
        if table == config or table.startswith(f"{config}_"):
            return config
    return table


def table_format(table: str) -> str:
    return "parquet" if table.startswith("fits_") or table in PARQUET_TABLES else "csv"


def build_tables(ctx: context.ExportContext) -> dict[str, pd.DataFrame]:
    """Every release table; each exporter runs once, and `fits` reuses the held-out and validation tables."""
    tables: dict[str, pd.DataFrame] = {}
    for exporter in (partition_tasks, swarm_qwen3, swarm_llama, heldout, validation):
        _add(tables, exporter.export(ctx))
    ladder_tables = ladder.export(ctx)
    ladder_tables["ladder"] = ladder.with_3e18_rows(
        ladder_tables["ladder"], tables["validation_3e18"], tables["swarm_qwen3_360m_1p6b"]
    )
    _add(tables, ladder_tables)
    _add(tables, starcoder.export())
    _add(tables, domain_sweeps.export())
    _add(tables, fits.export(ctx, heldout_table=tables["heldout"], validation_table=tables["validation_3e18"]))
    _add(tables, {"run_sets": checks.run_sets(tables)})
    return tables


def _add(tables: dict[str, pd.DataFrame], exported: dict[str, pd.DataFrame]) -> None:
    clash = sorted(tables.keys() & exported.keys())
    if clash:
        raise ValueError(f"two exporters write {clash}")
    tables.update(exported)


def public_source(path: Path) -> str:
    """A source's path relative to its repository, with hash-named folders replaced by `*`."""
    if path.is_relative_to(paths.REPO):
        relative = path.relative_to(paths.REPO).as_posix()
    elif path.is_relative_to(paths.MIXSEL):
        relative = MIXSEL_PREFIX + path.relative_to(paths.MIXSEL).as_posix()
    else:
        raise ValueError(f"a source outside the marin and mixture-selection repositories: {path}")
    return HASH_FOLDER.sub("*", relative)


def _sources(table: str) -> list[Path]:
    exporter = EXPORTER.get(config_of(table))
    return sorted(exporter.required_sources()) if exporter else []


def _notes(table: str, df: pd.DataFrame) -> list[str]:
    notes = list(NOTES.get(table, []))
    if "components_available" in df:
        notes.append(_aggregate_counts(df))
    return notes


def _aggregate_counts(df: pd.DataFrame) -> str:
    """How many rows of a run table compute each aggregate from their components, take it from their source's record,
    or leave it null."""
    counts = []
    for objective, weights in naming.task_weights().items():
        computed = df[[f"bpb__{t}" for t in weights]].notna().all(axis=1)
        present = df[f"bpb__{objective}"].notna()
        counts.append(
            f"bpb__{objective} is computed from components in {computed.sum()} rows, recorded in "
            f"{(present & ~computed).sum()} and null in {(~present).sum()}"
        )
    return f"Aggregates: {'; '.join(counts)}."


def _write_table(df: pd.DataFrame, path: Path, form: str) -> None:
    if form == "parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)  # pandas' default float repr round-trips every value exactly


def _commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=paths.REPO, check=True
    ).stdout.strip()


def _require_private_outside(out: Path, private_dir: Path) -> None:
    """SOURCES.json, which holds absolute local paths, must never land in the build, which is uploaded."""
    if private_dir.resolve().is_relative_to(out.resolve()):
        raise ValueError(f"the private directory {private_dir} is inside the build {out}")


def write(tables: dict[str, pd.DataFrame], out: Path, private_dir: Path) -> dict:
    """Write the build under `out` and SOURCES.json under `private_dir`; return the manifest.

    The data folder is replaced only when a previous build's MANIFEST.json sits beside it, so the export never
    deletes a folder it did not write.
    """
    _require_private_outside(out, private_dir)
    undocumented = sorted(set(tables) - set(KEYING))
    if undocumented:
        raise ValueError(f"tables without a keying sentence: {undocumented}")
    data = out / "data"
    if data.exists():
        if not (out / MANIFEST).exists():
            raise ValueError(f"{data} has no {MANIFEST} beside it, so the export did not write it; not replacing it")
        shutil.rmtree(data)
    out.mkdir(parents=True, exist_ok=True)
    commit, built_at = _commit(), datetime.now(UTC).isoformat(timespec="seconds")
    manifest = {
        "commit": commit,
        "built_at": built_at,
        "notes": RELEASE_NOTES,
        "context_sources": [public_source(p) for p in context.required_sources()],
        "tables": [],
    }
    private = {
        "commit": commit,
        "built_at": built_at,
        "context": [str(p) for p in context.required_sources()],
        "tables": {},
    }
    for name, df in tables.items():
        config, form = config_of(name), table_format(name)
        path = data / config / f"{name}.{form}"
        path.parent.mkdir(parents=True, exist_ok=True)
        _write_table(df, path, form)
        sources = _sources(name)
        manifest["tables"].append(
            {
                "config": config,
                "table": name,
                "path": path.relative_to(out).as_posix(),
                "format": form,
                "rows": len(df),
                "columns": [str(c) for c in df.columns],
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "keying": KEYING[name],
                "notes": _notes(name, df),
                "sources": sorted({public_source(p) for p in sources}),
            }
        )
        private["tables"][name] = [str(p) for p in sources]
    shutil.copyfile(FIGURES_TABLES, out / FIGURES_TABLES.name)
    manifest["consistency_problems"] = checks.consistency(tables)
    (out / MANIFEST).write_text(json.dumps(manifest, indent=1) + "\n")
    private_dir.mkdir(parents=True, exist_ok=True)
    (private_dir / SOURCES).write_text(json.dumps(private, indent=1) + "\n")
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=paths.BUILD)
    parser.add_argument("--private-dir", type=Path, default=paths.PRIVATE)
    args = parser.parse_args(argv)
    _require_private_outside(args.out, args.private_dir)  # before the export runs
    manifest = write(build_tables(context.build()), args.out, args.private_dir)
    problems = manifest["consistency_problems"]
    total = sum(t["bytes"] for t in manifest["tables"])
    print(f"wrote {len(manifest['tables'])} tables ({total / 1e6:.1f} MB) to {args.out}; {len(problems)} problems")
    if problems:
        raise SystemExit("\n".join(problems[:50]))


if __name__ == "__main__":
    main()
