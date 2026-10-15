# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build the dataset release: `data/<config>/<table>.csv|parquet`, `MANIFEST.json` and `figures_tables.csv`.

`uv run python -m experiments.domain_phase_mix.release.export_dataset [--out release_build] [--private-dir DIR]`

`build_tables` runs every exporter once. `write` replaces the build's data folder, writes each table (Parquet for the
`fits` tables, the StarCoder dense grid and the trajectories; CSV with pandas' shortest round-trip float repr for the
rest, so every value is bit-identical to the exported one), copies figures_tables.csv and writes the public manifest:
per table its config, path, format, rows, columns, bytes, SHA-256, the string its observation ids hash (`keying`),
notes and repository-relative sources, and the cross-table consistency problems. The salt, the run-name map and
SOURCES.json (every table's absolute source paths) go to the private directory, which must lie outside the build. The
command exits non-zero, after writing the manifest, when two tables disagree about a run.
"""

import argparse
import hashlib
import json
import re
import shutil
import subprocess
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from experiments.domain_phase_mix.release import checks, context, paths
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
# The public manifest shows `*` for the run name that begins a per-run file name (it ends in a six-hex-digit hash) and
# for a folder named by a 64-hex content hash.
RUN_NAME_IN_FILE = re.compile(r"(?<=/)[^/]+-[0-9a-f]{6}(?=_[^/]+$)")
HASH_FOLDER = re.compile(r"(?<=/)[0-9a-f]{64}(?=/)")
MIXSEL_PREFIX = "mixture-selection/"

NO_IDS = "The table has no observation ids."
# The string each table's observation ids hash; one training run has one id wherever it appears.
KEYING = {
    "partition": NO_IDS,
    "tasks": NO_IDS,
    "swarm_qwen3_360m_1p6b": (
        "The training run name: without its six-hex-digit hash for the 280 fit rows, as the mixture-selection bundle "
        "names them, and with it for the ten proportional repeats, as W&B names them."
    ),
    "swarm_llama_160m_1p2b": "The training run name with its six-hex-digit hash, which W&B records as the run id.",
    "swarm_llama_200m_6b": "The training run name with its six-hex-digit hash, which W&B records as the run id.",
    "heldout": (
        "The training run name with its six-hex-digit hash wherever the registry or a validation launch records it, "
        "and the registry's row id for the 134 Llama 200M runs, which record no run name."
    ),
    "validation_3e18": (
        "The training run name with its six-hex-digit hash, from each run's evaluation-metrics path or its launch's "
        "record of run names."
    ),
    "ladder": (
        "The training run name with its six-hex-digit hash; the 3e18 copies keep the ids of their validation_3e18 "
        "and swarm_qwen3_360m_1p6b rows."
    ),
    "ladder_accuracy": "The ladder's ids of the four scored 1e21 runs.",
    "ladder_compute_equivalence": NO_IDS,
    "ladder_unimax_cap_sweep": NO_IDS,
    "starcoder_observations": "The training run name of each atlas run.",
    "starcoder_curve_membership": "The training run name of each atlas run.",
    "starcoder_curves": NO_IDS,
    "starcoder_fit_parameters": NO_IDS,
    "starcoder_dense_grid": NO_IDS,
    "starcoder_diagonal": "The training run name.",
    "starcoder_curricula": NO_IDS,
    "starcoder_curricula_matched_nd": NO_IDS,
    "starcoder_curricula_matched_nd_confirmation": "The training run names of each pair's two runs (their W&B ids).",
    "starcoder_transition": (
        "The training run name; for the gradient-conflict runs, the trajectory id without its W&B prefix."
    ),
    "starcoder_ramp_designs": "The training run names of each ramp run, its abrupt source run and its tied comparator.",
    "starcoder_trajectories": (
        "The training run name; for the gradient-conflict runs, the trajectory id without its W&B prefix."
    ),
    "starcoder_repeat_noise": "The training run name with its six-hex-digit hash.",
    "domain_sweeps": "The training run name, to which these launches append no hash.",
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
        "The source bundle records components only: bpb__uncheatable_source applies the fixed byte shares to each "
        "run's seven Uncheatable components, and the table has no OlmoBaseEval Easy source aggregate.",
        "The ten proportional repeats are also rows of the ladder's 3e18 rung.",
    ],
    "swarm_llama_160m_1p2b": [
        "The 39 single-bucket deletions (design_kind deletion) are fit rows here and frozen held-out runs in heldout, "
        "under the same observation ids; pool the two tables by observation_id.",
        "No launch manifest of the deletions is among the sources, so their seeds are null.",
        "bpb__uncheatable_source uses the legacy Uncheatable weights (aggregate_version legacy) and differs from "
        "bpb__uncheatable_fixed by less than 1e-4 BPB.",
    ],
    "swarm_llama_200m_6b": [
        "The proportional fit row (run_label baseline_proportional) keeps its own run's values. The parity panel "
        "behind the swarm's fits, and the July packet copied from it, replaced them by the mean of 11 proportional "
        "runs: the ten repeats here and the two-phase swarm's proportional run, which this release omits.",
        "The stratified baseline is a two-phase swarm run whose launch left no manifest among the sources, so its "
        "seeds are null.",
        "bpb__uncheatable_source uses the legacy Uncheatable weights (aggregate_version legacy) and differs from "
        "bpb__uncheatable_fixed by less than 1e-4 BPB.",
    ],
    "heldout": [
        "The 39 Llama 160M single-bucket deletions are also fit rows of swarm_llama_160m_1p2b, and 117 Qwen3 runs "
        "(the 116 expanded-cohort validation runs and the UniMax-8 seed-0 run) are also rows of validation_3e18; each "
        "such run has one observation id and equal BPB, weights and pool fractions in both tables (seeds and labels "
        "may differ).",
        "Ten frozen archive runs, in four exposure groups, are deterministic re-trainings: their aggregates are "
        "bit-identical within each group.",
        "628 of the 929 runs (455 Qwen3, 117 Llama 200M and 56 Llama 160M runs) have all 58 per-task values.",
        "aggregate_version is fixed_20260927 for the Qwen3 runs and legacy for the Llama runs, except for the 14 "
        "path-midpoint and RegMix-reference runs (pooled_schema2), whose recorded Uncheatable aggregate is the "
        "BPB-schema-2 pooled value, 0.008 to 0.011 BPB above bpb__uncheatable_fixed; compare runs on "
        "bpb__uncheatable_fixed.",
        "proposal_objective uses the public objective names: none for the UniMax-8 baseline (epoch cap 8, as in "
        "validation_3e18), uncheatable_and_obe_easy for designs built for both objectives, and other where the "
        "registry's run names name no objective; the Llama panels leave it null.",
    ],
    "validation_3e18": [
        "UniMax-8 trainer seeds 1 and 2 carry source aggregates only; their per-task values are not among the sources.",
        "The 14 path_midpoints and regmix_reference runs record the BPB-schema-2 pooled Uncheatable aggregate "
        "(aggregate_version pooled_schema2), 0.008 to 0.011 BPB above bpb__uncheatable_fixed.",
        "117 runs are also rows of heldout, and the ladder's 3e18 rung copies 15: trainer seeds 0 to 2 of MARINER's "
        "and of the matched Olmix proposals for each objective, and the UniMax-8 repeats.",
    ],
    "ladder": [
        "The 3e18 rung copies 15 validation_3e18 rows and the ten proportional repeats of swarm_qwen3_360m_1p6b "
        "under their observation ids, and adds the June proportional run, which no other table holds; the scaling "
        "figure averages these runs at 3e18.",
        "UniMax-8 trainer seeds 1 and 2 at 3e18 carry aggregates only, and the proportional and UniMax-8 runs at "
        "2e19 and 3e20 carry aggregates and Uncheatable components only (components_available false).",
        "The swarm repeats' bpb__obe_easy_source is the mean of their 51 components, as every source computes it.",
    ],
    "ladder_unimax_cap_sweep": [
        "The UniMax cap sweep's runs at caps 1, 4 and 12, at the 3e18 rung's model size and token count; its cap-8 "
        "and proportional runs are ladder rows. Their Uncheatable values exist only in the runs' evaluation logs, so "
        "the table holds the 51 OlmoBaseEval Easy components and their mean.",
    ],
    "domain_sweeps": [
        "These models use the TinyLlama tokenizer named in `tokenizer`, not the tokenizer of the tasks table, which "
        "describes the paper's swarms.",
    ],
    "fits_swarm_predictions": [
        "In-sample fitted values of the reference fit, which trained on these runs. True out-of-fold predictions "
        "are not exported; fits_learning_curves and fits_reliability hold the out-of-fold metrics.",
    ],
    "fits_learning_curves": ["The out-of-fold predictions behind these metrics are not exported."],
}
RELEASE_NOTES = [
    "observation_id is a salted hash of the string each table's keying names; one training run has one id in every "
    "table that holds it, and run_sets lists its memberships, so pool tables by observation_id.",
    "mixture_id hashes a row's weights rounded to 1e-9; exposure_id hashes its weights, pool fractions and the "
    "partition revision.",
    "aggregate_version names the weighting behind bpb__uncheatable_source: fixed_20260927 (the paper's fixed byte "
    "shares, which bpb__uncheatable_fixed applies to every run with components), legacy (the archived swarms' "
    "weights) or pooled_schema2 (BPB schema 2's pooled bytes).",
    "Source paths are relative to the marin repository's root, those under mixture-selection/ to the "
    "mixture-selection repository's; `*` stands for a run name that begins a file name or a hash that names a folder.",
    "figures_tables.csv lists the paper's 26 figures and 20 tables with their builders (paths in the paper folder, or "
    "in the repository under experiments/ and .agents/) and the configs they read: deferred marks two-phase data "
    "this release omits, constant configuration constants typed into the paper, and untraced numbers whose inputs "
    "the release does not hold.",
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
    _add(tables, starcoder.export(ctx))
    _add(tables, domain_sweeps.export(ctx))
    _add(tables, fits.export(ctx, heldout_table=tables["heldout"], validation_table=tables["validation_3e18"]))
    _add(tables, {"run_sets": checks.run_sets(tables)})
    return tables


def _add(tables: dict[str, pd.DataFrame], exported: dict[str, pd.DataFrame]) -> None:
    clash = sorted(tables.keys() & exported.keys())
    if clash:
        raise ValueError(f"two exporters write {clash}")
    tables.update(exported)


def public_source(path: Path) -> str:
    """A source's path relative to its repository, with run names and hash-named folders replaced by `*`."""
    if path.is_relative_to(paths.REPO):
        relative = path.relative_to(paths.REPO).as_posix()
    elif path.is_relative_to(paths.MIXSEL):
        relative = MIXSEL_PREFIX + path.relative_to(paths.MIXSEL).as_posix()
    else:
        raise ValueError(f"a source outside the marin and mixture-selection repositories: {path}")
    return HASH_FOLDER.sub("*", RUN_NAME_IN_FILE.sub("*", relative))


def _sources(table: str) -> list[Path]:
    exporter = EXPORTER.get(config_of(table))
    return sorted(exporter.required_sources()) if exporter else []


def _notes(table: str, df: pd.DataFrame) -> list[str]:
    notes = list(NOTES.get(table, []))
    if "aggregate_version" in df:
        counts = Counter(df.aggregate_version)
        notes.append("aggregate_version: " + ", ".join(f"{v} ({n} rows)" for v, n in sorted(counts.items())))
    return notes


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
    """The salt, the run-name map and SOURCES.json must never land in the build, which is uploaded."""
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
    _require_private_outside(args.out, args.private_dir)  # before the context creates a salt there
    ctx = context.build(private_dir=args.private_dir)
    manifest = write(build_tables(ctx), args.out, args.private_dir)
    ctx.registry.save()
    problems = manifest["consistency_problems"]
    total = sum(t["bytes"] for t in manifest["tables"])
    print(f"wrote {len(manifest['tables'])} tables ({total / 1e6:.1f} MB) to {args.out}; {len(problems)} problems")
    if problems:
        raise SystemExit("\n".join(problems[:50]))


if __name__ == "__main__":
    main()
