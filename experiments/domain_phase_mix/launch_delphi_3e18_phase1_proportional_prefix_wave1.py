# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Continue the exact Delphi proportional prefix over the frozen Wave-1 panel."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import cast

import fsspec
from fray.cluster import ResourceConfig
from marin.execution.context import executor_context
from marin.execution.executor import ExecutorMainConfig, executor_main, get_git_commit
from marin.execution.remote import remote
from marin.execution.types import ExecutorStep, this_output_path, versioned
from marin.processing.tokenize import step_to_lm_mixture_component
from rigging.filesystem import marin_prefix_for_region

from experiments.domain_phase_mix import launch_delphi_3e18_phase0_prefix_candidates as prefixes
from experiments.domain_phase_mix import launch_delphi_3e18_phase0_prefix_replay as replay
from experiments.domain_phase_mix import launch_delphi_3e18_phase1_harsh_cap_branches as runtime
from experiments.domain_phase_mix import launch_delphi_augmented_swarm_3e18 as base
from experiments.llama import llama3_tokenizer

logger = logging.getLogger(__name__)

EXPERIMENT_NAME = "pinlin_calvin_xu/data_mixture/delphi_3e18_phase1_proportional_prefix_wave1_v6e8_20260825"
REFERENCE_OUTPUTS = Path(__file__).resolve().parent / "exploratory" / "two_phase_many" / "reference_outputs"
DEFAULT_CANDIDATE_WEIGHTS = REFERENCE_OUTPUTS / "delphi_phase0_prefix_candidates_20260824" / "candidate_weights.csv"
DEFAULT_SELECTED_PREFIXES = REFERENCE_OUTPUTS / "delphi_phase0_proportional_prefix_20260825" / "selected_prefixes.json"
DEFAULT_DESIGN_DIR = REFERENCE_OUTPUTS / "delphi_phase1_proportional_prefix_wave1_20260825"
DEFAULT_CONTINUATION_SUMMARY = DEFAULT_DESIGN_DIR / "continuation_summary.csv"
DEFAULT_CONTINUATION_WEIGHTS = DEFAULT_DESIGN_DIR / "continuation_weights.csv"
DEFAULT_DESIGN_MANIFEST = DEFAULT_DESIGN_DIR / "manifest.json"
LOCAL_DRY_RUN_DIR = DEFAULT_DESIGN_DIR / "launch_dry_run"
TARGET_PREFIX = "proportional_control"
PREFIX_HARDWARE = runtime.TpuHardware(tpu_type="v5p-8", region="us-east5", zone="us-east5-a")
CONTINUATION_HARDWARE = runtime.TpuHardware(tpu_type="v6e-8", region="us-east5", zone="us-east5-b")
EXPECTED_PREFIX_SEEDS = (0, 1)
BRANCH_RUN_ID_BASE = 976_000
DEFAULT_MAX_CONCURRENT = 102


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_uri_bytes(uri: str) -> bytes:
    fs, path = fsspec.core.url_to_fs(uri)
    with fs.open(path, "rb") as handle:
        return handle.read()


def source_prefix_specs(
    candidate_weights_path: Path,
    expected_candidate_weights_sha256: str,
    analysis_output_path: str,
) -> dict[tuple[str, int], base.DelphiSwarmRunSpec]:
    specs, _ = prefixes.candidate_specs(
        candidate_weights_path=candidate_weights_path,
        expected_sha256=expected_candidate_weights_sha256,
        analysis_output_path=analysis_output_path,
        tpu_region=PREFIX_HARDWARE.region,
        tpu_zone=PREFIX_HARDWARE.zone,
    )
    selected = {}
    for spec in specs:
        if spec.run_name != f"prefix_{TARGET_PREFIX}_seed{spec.trainer_seed}":
            continue
        if spec.trainer_seed not in EXPECTED_PREFIX_SEEDS:
            continue
        selected[(TARGET_PREFIX, spec.trainer_seed)] = spec
    expected = {(TARGET_PREFIX, seed) for seed in EXPECTED_PREFIX_SEEDS}
    if set(selected) != expected:
        raise ValueError(f"Exact proportional prefix specs are incomplete: {sorted(selected)}")
    return selected


def load_and_validate_prefixes(
    selected_prefixes_path: str,
    expected_selected_prefixes_sha256: str,
    candidate_weights_sha256: str,
    prefix_replay_code_commit: str,
    specs: dict[tuple[str, int], base.DelphiSwarmRunSpec],
) -> list[runtime.PrefixCheckpoint]:
    payload_bytes = read_uri_bytes(selected_prefixes_path)
    actual_sha256 = hashlib.sha256(payload_bytes).hexdigest()
    if actual_sha256 != expected_selected_prefixes_sha256:
        raise ValueError(f"Selected-prefix manifest changed: {actual_sha256} != {expected_selected_prefixes_sha256}")
    payload = json.loads(payload_bytes)
    if payload.get("candidate_weights_sha256") != candidate_weights_sha256:
        raise ValueError("Selected-prefix manifest references different candidate weights")
    if payload.get("prefix_replay_code_commit") != prefix_replay_code_commit:
        raise ValueError("Selected-prefix manifest references different prefix code")
    aliases = payload.get("selected_aliases")
    if not isinstance(aliases, list) or [row.get("canonical_candidate_id") for row in aliases] != [TARGET_PREFIX]:
        raise ValueError("Selected-prefix manifest must name only the proportional prefix")
    source_hardware = payload.get("source_prefix_hardware")
    if source_hardware != {
        "evidence": "successful executor records pinned in each prefix row",
        "tensor_parallel_size": 1,
        "tpu_region": PREFIX_HARDWARE.region,
        "tpu_type": PREFIX_HARDWARE.tpu_type,
        "tpu_zone": PREFIX_HARDWARE.zone,
    }:
        raise ValueError("Selected-prefix hardware evidence changed")
    prefix_payloads = payload.get("prefixes", [])
    rows = [
        runtime.PrefixCheckpoint(
            candidate_id=str(row["candidate_id"]),
            repeat_seed=int(row["repeat_seed"]),
            checkpoint_uri=str(row["checkpoint_uri"]),
            provenance_sha256=str(row["provenance_sha256"]),
        )
        for row in prefix_payloads
    ]
    if {(row.candidate_id, row.repeat_seed) for row in rows} != set(specs):
        raise ValueError("Selected-prefix checkpoints do not match the frozen proportional seeds")
    payload_by_identity = {(str(row["candidate_id"]), int(row["repeat_seed"])): row for row in prefix_payloads}
    for row in rows:
        spec = specs[(row.candidate_id, row.repeat_seed)]
        source_row = payload_by_identity[(row.candidate_id, row.repeat_seed)]
        executor_info_bytes = read_uri_bytes(str(source_row["executor_info_uri"]))
        if hashlib.sha256(executor_info_bytes).hexdigest() != source_row["executor_info_sha256"]:
            raise ValueError(f"Prefix executor record changed for seed {row.repeat_seed}")
        executor_info = json.loads(executor_info_bytes)
        executor_spec = executor_info["config"]["prefix_config"]["run_spec"]
        if {
            "run_name": executor_spec["run_name"],
            "data_seed": executor_spec["data_seed"],
            "trainer_seed": executor_spec["trainer_seed"],
            "tpu_type": executor_spec["tpu_type"],
            "tpu_region": executor_spec["tpu_region"],
            "tpu_zone": executor_spec["tpu_zone"],
            "tensor_parallel_size": executor_spec["tensor_parallel_size"],
        } != {
            "run_name": spec.run_name,
            "data_seed": spec.data_seed,
            "trainer_seed": spec.trainer_seed,
            "tpu_type": PREFIX_HARDWARE.tpu_type,
            "tpu_region": PREFIX_HARDWARE.region,
            "tpu_zone": PREFIX_HARDWARE.zone,
            "tensor_parallel_size": spec.tensor_parallel_size,
        }:
            raise ValueError(f"Prefix executor record does not match seed {row.repeat_seed}")
        if not row.checkpoint_uri.startswith("gs://marin-us-east5/"):
            raise ValueError(f"Prefix checkpoint is not east5-local: {row.checkpoint_uri}")
        if f"/{prefixes.EXPERIMENT_NAME}/" not in row.checkpoint_uri:
            raise ValueError(f"Prefix checkpoint is outside the frozen candidate experiment: {row.checkpoint_uri}")
        if not row.checkpoint_uri.endswith(f"/checkpoints/step-{replay.EXPECTED_PREFIX_HF_STEP}"):
            raise ValueError(f"Prefix checkpoint is not the exact boundary state: {row.checkpoint_uri}")
        fs, checkpoint_path = fsspec.core.url_to_fs(row.checkpoint_uri)
        metadata_path = os.path.join(checkpoint_path, "metadata.json")
        if not fs.exists(metadata_path):
            raise FileNotFoundError(f"Prefix checkpoint metadata is missing: {row.checkpoint_uri}")
        with fs.open(metadata_path) as handle:
            metadata = json.load(handle)
        if metadata.get("step") != replay.EXPECTED_PREFIX_HF_STEP or metadata.get("is_temporary") is not False:
            raise ValueError(f"Prefix checkpoint is not permanent: {metadata}")
        output_root = row.checkpoint_uri.rsplit("/checkpoints/", maxsplit=1)[0]
        provenance_bytes = read_uri_bytes(f"{output_root}/{prefixes.CANDIDATE_PROVENANCE_FILENAME}")
        if hashlib.sha256(provenance_bytes).hexdigest() != row.provenance_sha256:
            raise ValueError(f"Prefix provenance changed for seed {row.repeat_seed}")
        expected_provenance = {
            "candidate_id": TARGET_PREFIX,
            "candidate_weights_sha256": candidate_weights_sha256,
            "checkpoint_step": replay.EXPECTED_PREFIX_HF_STEP,
            "checkpoint_uri": row.checkpoint_uri,
            "data_seed": prefixes.DATA_SEED_BASE + row.repeat_seed,
            "experiment_name": prefixes.EXPERIMENT_NAME,
            "phase_weights_sha256": prefixes.phase_weights_sha256(spec.phase_weights),
            "replay_code_commit": prefix_replay_code_commit,
            "run_id": (
                prefixes.RUN_ID_BASE
                + prefixes.CANDIDATE_IDS.index(TARGET_PREFIX) * len(prefixes.REPEAT_SEEDS)
                + row.repeat_seed
            ),
            "run_name": f"prefix_{TARGET_PREFIX}_seed{row.repeat_seed}",
            "run_order": prefixes.CANDIDATE_IDS.index(TARGET_PREFIX) * len(prefixes.REPEAT_SEEDS) + row.repeat_seed,
            "trainer_seed": row.repeat_seed,
            "trainer_state_step": replay.EXPECTED_PREFIX_TRAIN_STEPS,
        }
        if json.loads(provenance_bytes) != expected_provenance:
            raise ValueError(f"Prefix provenance does not match the frozen seed-{row.repeat_seed} state")
    return sorted(rows, key=lambda row: row.repeat_seed)


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-weights", type=Path, default=DEFAULT_CANDIDATE_WEIGHTS)
    parser.add_argument("--expected-candidate-sha256", required=True)
    parser.add_argument("--continuation-summary", type=Path, default=DEFAULT_CONTINUATION_SUMMARY)
    parser.add_argument("--expected-continuation-summary-sha256", required=True)
    parser.add_argument("--continuation-weights", type=Path, default=DEFAULT_CONTINUATION_WEIGHTS)
    parser.add_argument("--expected-continuation-weights-sha256", required=True)
    parser.add_argument("--design-manifest", type=Path, default=DEFAULT_DESIGN_MANIFEST)
    parser.add_argument("--expected-design-manifest-sha256", required=True)
    parser.add_argument("--selected-prefixes", default=str(DEFAULT_SELECTED_PREFIXES))
    parser.add_argument("--expected-selected-prefixes-sha256", required=True)
    parser.add_argument("--prefix-replay-code-commit", required=True)
    parser.add_argument("--analysis-output-path", default=base.DEFAULT_ANALYSIS_OUTPUT_PATH)
    parser.add_argument("--branch-run-id-base", type=int, default=BRANCH_RUN_ID_BASE)
    parser.add_argument("--max-concurrent", type=int, default=DEFAULT_MAX_CONCURRENT)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--run-order", action="append", type=int, dest="run_orders")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dry-run-output-dir", type=Path)
    return parser.parse_known_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args, remaining = parse_args()
    sys.argv = [sys.argv[0], *remaining]
    if not 1 <= args.max_concurrent <= DEFAULT_MAX_CONCURRENT:
        raise ValueError(f"--max-concurrent must be in [1, {DEFAULT_MAX_CONCURRENT}]")
    expected_prefix = marin_prefix_for_region(CONTINUATION_HARDWARE.region)
    if os.environ.get("MARIN_PREFIX", expected_prefix) != expected_prefix:
        raise ValueError(f"MARIN_PREFIX must be {expected_prefix}")
    os.environ["MARIN_PREFIX"] = expected_prefix
    code_commit = replay.validate_replay_code_commit(args.code_commit, get_git_commit())
    specs = source_prefix_specs(args.candidate_weights, args.expected_candidate_sha256, args.analysis_output_path)
    checkpoints = load_and_validate_prefixes(
        args.selected_prefixes,
        args.expected_selected_prefixes_sha256,
        args.expected_candidate_sha256,
        args.prefix_replay_code_commit,
        specs,
    )
    design_rows = runtime.load_design(
        args.continuation_summary,
        args.expected_continuation_summary_sha256,
        args.continuation_weights,
        args.expected_continuation_weights_sha256,
        args.design_manifest,
        args.expected_design_manifest_sha256,
        (TARGET_PREFIX,),
    )
    all_rows = runtime.enrich_rows(design_rows, checkpoints, specs, args.branch_run_id_base)
    full_design_rows = len(all_rows)
    rows = all_rows
    if args.run_orders is not None:
        selected_orders = tuple(dict.fromkeys(args.run_orders))
        unknown = sorted(set(selected_orders) - {int(row["run_order"]) for row in rows})
        if unknown:
            raise ValueError(f"Unknown --run-order values: {unknown}")
        rows = [row for row in rows if int(row["run_order"]) in selected_orders]
    serializable_rows = [{**row, "prefix": asdict(cast(runtime.PrefixCheckpoint, row["prefix"]))} for row in all_rows]
    manifest_identity = hashlib.sha256(
        json.dumps(
            {
                "selected_prefixes_sha256": args.expected_selected_prefixes_sha256,
                "candidate_weights_sha256": args.expected_candidate_sha256,
                "continuation_summary_sha256": args.expected_continuation_summary_sha256,
                "continuation_weights_sha256": args.expected_continuation_weights_sha256,
                "design_manifest_sha256": args.expected_design_manifest_sha256,
                "prefix_replay_code_commit": args.prefix_replay_code_commit,
                "branch_code_commit": code_commit,
                "branch_run_id_base": args.branch_run_id_base,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    manifest_config = runtime.SaveManifestConfig(
        experiment_name=EXPERIMENT_NAME,
        output_path=str(args.dry_run_output_dir or LOCAL_DRY_RUN_DIR),
        selected_prefixes_json=json.dumps([asdict(row) for row in checkpoints], sort_keys=True),
        selected_prefixes_sha256=args.expected_selected_prefixes_sha256,
        candidate_weights_sha256=args.expected_candidate_sha256,
        candidate_aliases_sha256=args.expected_selected_prefixes_sha256,
        continuation_summary_sha256=args.expected_continuation_summary_sha256,
        continuation_weights_sha256=args.expected_continuation_weights_sha256,
        design_manifest_sha256=args.expected_design_manifest_sha256,
        prefix_replay_code_commit=args.prefix_replay_code_commit,
        code_commit=code_commit,
        branch_run_id_base=args.branch_run_id_base,
        full_design_rows=full_design_rows,
        branch_rows_json=json.dumps(serializable_rows, sort_keys=True),
        prefix_hardware=PREFIX_HARDWARE,
        manifest_identity=versioned(manifest_identity),
    )
    if args.dry_run:
        runtime.save_manifest(manifest_config)
        logger.info("Wrote the %d-row manifest and selected %d rows", full_design_rows, len(rows))
        return

    validation_steps = base._default_validation_sets(tokenizer=llama3_tokenizer)
    validation_configs = {
        name: step_to_lm_mixture_component(step, include_raw_paths=False) for name, step in validation_steps.items()
    }
    steps = []
    with executor_context():
        for row in rows:
            prefix = cast(runtime.PrefixCheckpoint, row["prefix"])
            source = specs[(prefix.candidate_id, prefix.repeat_seed)]
            run_spec = replace(
                source,
                run_order=int(row["run_order"]),
                run_id=int(row["run_id"]),
                run_name=str(row["run_name"]),
                source_run_name=str(row["run_name"]),
                source_experiment=EXPERIMENT_NAME,
                panel_source="sequential_phase1_proportional_prefix_wave1",
                data_seed=int(row["data_seed"]),
                trainer_seed=int(row["trainer_seed"]),
                tpu_type=CONTINUATION_HARDWARE.tpu_type,
                tpu_region=CONTINUATION_HARDWARE.region,
                tpu_zone=CONTINUATION_HARDWARE.zone,
                tensor_parallel_size=base._tensor_parallel_size(
                    source.model_hidden_dim,
                    CONTINUATION_HARDWARE.tpu_type,
                ),
                max_simulated_epoch=float(row["max_simulated_epoch"]),
                q95_simulated_epoch=float(row["q95_simulated_epoch"]),
                mean_phase_tv_to_proportional=float(row["mean_phase_tv_to_proportional"]),
                phase_weights=cast(dict[str, dict[str, float]], row["phase_weights"]),
            )
            resources = ResourceConfig.with_tpu(
                CONTINUATION_HARDWARE.tpu_type,
                regions=[CONTINUATION_HARDWARE.region],
                zone=CONTINUATION_HARDWARE.zone,
            )
            steps.append(
                ExecutorStep(
                    name=f"{EXPERIMENT_NAME}/{run_spec.run_name}",
                    fn=remote(
                        runtime.run_phase_1_branch,
                        resources=resources,
                        env_vars={base.HF_HUB_DISABLE_XET_ENV_VAR: "1"},
                    ),
                    resources=resources,
                    config=runtime.HarshBranchTrainingConfig(
                        experiment_name=EXPERIMENT_NAME,
                        analysis_output_path=args.analysis_output_path,
                        output_path=this_output_path(),
                        run_spec=run_spec,
                        validation_configs=validation_configs,
                        prefix_checkpoint=prefix,
                        prefix_replay_code_commit=args.prefix_replay_code_commit,
                        candidate_weights_sha256=args.expected_candidate_sha256,
                        candidate_aliases_sha256=args.expected_selected_prefixes_sha256,
                        continuation_weights_sha256=args.expected_continuation_weights_sha256,
                        design_manifest_sha256=args.expected_design_manifest_sha256,
                        continuation_id=str(row["continuation_id"]),
                        code_commit=code_commit,
                        prefix_hardware=PREFIX_HARDWARE,
                        panel_identity=manifest_config.manifest_identity,
                    ),
                )
            )
        steps.append(
            ExecutorStep(
                name=f"{EXPERIMENT_NAME}/manifest",
                fn=runtime.save_manifest,
                config=replace(manifest_config, output_path=this_output_path()),
            )
        )
    if os.getenv("CI") is not None:
        logger.info("Built %d branch steps; skipping launch in CI", len(steps))
        return
    executor_main(
        ExecutorMainConfig(max_concurrent=args.max_concurrent),
        steps=steps,
        description=("Delphi 3e18 proportional-prefix Wave 1: fixed-prefix phase-1 response coverage and controls"),
    )


if __name__ == "__main__":
    main()
