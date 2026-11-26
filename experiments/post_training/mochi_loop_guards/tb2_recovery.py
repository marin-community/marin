# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The registered TB2 infrastructure-recovery chain: ledger, selection, and stitch.

Promoted from the ad-hoc ``submit-tb2-repetition-ledger.py``,
``submit-tb2-recovery-selection.py`` and ``stitch-tb2-recovery-regional.py`` worker
scripts. The registered S0 rule is fixed: 267 TB2 trials, 33 infrastructure outcomes
across 18 tasks, three frozen repetitions per task, master seed 20261125. One replay
replaces only those frozen task/repetition identities; every original graded outcome
is preserved verbatim.

All three subcommands run in region under SkyRL's frozen CPU environment:

    … cpu-submit --skyrl --cpu 8 --memory 32GB --disk 64GB -- \\
      python -m experiments.post_training.mochi_loop_guards tb2-repetition-ledger …

The stitcher also accepts smaller explicit expectations so a dry fixture can exercise
the same code path.
"""

import argparse
import gzip
import hashlib
import io
import json
import zipfile
from collections import Counter, defaultdict
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import yaml
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath

MASTER_SEED = 20261125
REGISTERED_ORIGINAL_TRIALS = 267
REGISTERED_RECOVERY_TARGETS = 33
REGISTERED_RECOVERY_TASKS = 18
REPETITIONS_PER_TASK = 3
REGISTERED_SELECTION_VERSION = "2026.11.25.1"


def digest(value) -> str:
    """Hash one JSON-serializable value canonically."""
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def row_key(row: Mapping) -> tuple:
    """The scientific-content identity of one saved trial: prompt plus outcome fields."""
    return (
        row["uid"],
        digest(row["response_token_ids"]),
        digest(row.get("messages")),
        (row.get("verification") or {}).get("status"),
        row.get("exception_type"),
    )


def stitched_rows(
    originals: Mapping[tuple, tuple[dict, Mapping]],
    replays: Mapping[tuple, tuple[dict, str, str]],
    targets: set[tuple],
    *,
    sampling_seed,
) -> tuple[list[dict], list[dict]]:
    """Join ledger identities with replays, replacing only registered infrastructure outcomes.

    ``originals`` maps one frozen task/repetition identity to its saved row and ledger
    metadata; ``replays`` maps the same identities to their replay row, record id and
    archive. Every original graded outcome is preserved verbatim; every target identity
    must have exactly one replay. Returns the stitched rows and their replacement
    provenance records.
    """
    result = []
    replacements = []
    for identity, (original, metadata) in sorted(originals.items()):
        row = dict(original)
        provenance = {
            "task_id": identity[0],
            "repetition_id": identity[1],
            "sampling_seed": sampling_seed(*identity),
            "original_record_id": metadata["record_id"],
            "original_dump_ordinal": original["ordinal"],
        }
        if identity in targets:
            replay, record_id, archive = replays[identity]
            row = dict(replay)
            provenance.update(replaced_original_infrastructure=True, replay_record_id=record_id, replay_archive=archive)
            replacements.append(
                {
                    **provenance,
                    "replay_verification_status": (replay.get("verification") or {}).get("status"),
                }
            )
        row["recovery_provenance"] = provenance
        result.append(row)
    return result, replacements


def stitch(argv: list[str] | None = None) -> None:
    """Stitch one registered replay over the original TB2 run's ungraded identities."""
    # The vendored SkyRL scripts ship in SkyRL's frozen CPU environment (cpu-submit
    # --skyrl); they are not Marin workspace dependencies.
    from vendor.skyrl.scripts.loop_audit import is_infrastructure_error, read_rows
    from skyrl_train.mismatch_probe.protocol import request_seed

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-run-output", required=True)
    parser.add_argument("--recovery-run-output", required=True)
    parser.add_argument("--recovery-receipt", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--expected-originals", type=int, default=REGISTERED_ORIGINAL_TRIALS)
    parser.add_argument("--expected-targets", type=int, default=REGISTERED_RECOVERY_TARGETS)
    parser.add_argument("--expected-tasks", type=int, default=REGISTERED_RECOVERY_TASKS)
    args = parser.parse_args(argv)
    configure_coreweave_s3()
    receipt = json.loads(StoragePath(args.recovery_receipt).read_text())
    ledger = json.loads(StoragePath(receipt["original_ledger"]).read_text())
    original_rows = list(read_rows(ledger["source"]))
    original_groups = defaultdict(list)
    for row in original_rows:
        original_groups[row_key(row)].append(row)
    originals = {}
    for item in ledger["trials"]:
        group = original_groups[tuple(item["join_key"])]
        if len(group) != 1:
            raise ValueError("Original row/repetition attribution is ambiguous")
        identity = item["task_id"], item["repetition_id"]
        originals[identity] = (group[0], item)
    targets = {(item["task_id"], item["repetition_id"]) for item in receipt["target_trials"]}
    task_ids = set(receipt["selected_task_ids"])
    expected_replays = args.expected_tasks * REPETITIONS_PER_TASK
    if len(originals) != args.expected_originals or len(targets) != args.expected_targets or len(task_ids) != args.expected_tasks:
        raise ValueError("Recovery census differs from the registered rule")
    if {identity for identity, (row, _) in originals.items() if is_infrastructure_error(row)} != targets:
        raise ValueError("Recovery targets include a graded outcome or omit an original infrastructure outcome")

    config_path = "resolved-launch.yaml"
    original_config = yaml.safe_load((StoragePath(args.original_run_output) / config_path).read_text())["config"]
    recovery_root = StoragePath(args.recovery_run_output)
    recovery_config = yaml.safe_load((recovery_root / config_path).read_text())["config"]
    original_model = original_config["skyrl"]["trainer"]["policy"]["model"]
    recovery_model = recovery_config["skyrl"]["trainer"]["policy"]["model"]
    if original_model != recovery_model:
        raise ValueError("Recovery changed the initializer")
    paths = list((recovery_root / "exports/dumped_evals/*/eval_only/DCAgent2_terminal_bench_2.jsonl").glob())
    if len(paths) != 1:
        raise ValueError("Select exactly one completed recovery evaluation session")
    replay_rows = [row for row in read_rows(str(paths[0])) if row["task_id"] in task_ids]
    if len(replay_rows) != expected_replays:
        raise ValueError(f"Expected all {args.expected_tasks} recovery tasks at all three frozen repetitions")
    replay_groups = defaultdict(list)
    task_by_uid = {}
    for row in replay_rows:
        replay_groups[row_key(row)].append(row)
        task_by_uid[row["uid"]] = row["task_id"]
    trajectory_root = StoragePath(recovery_config["artifacts"]["attempts_root"]) / "trajectories"

    def extract(path):
        extracted = []
        with zipfile.ZipFile(io.BytesIO(path.read_bytes())) as archive:
            for name in archive.namelist():
                if not name.startswith("records/") or not name.endswith(".json.gz"):
                    continue
                record = json.loads(gzip.decompress(archive.read(name)))
                if record["trajectory"]["environment_extras"].get("data_source") != "DCAgent2/terminal_bench_2":
                    continue
                uid = record["trajectory"]["instance_id"].rsplit("-", 1)[-1]
                if uid not in task_by_uid:
                    continue
                key = (
                    uid,
                    digest(record["response"]["token_ids"]),
                    digest(record["response"].get("messages")),
                    (record.get("verification_result") or {}).get("status"),
                    record["disposition"].get("exception_type"),
                )
                extracted.append(
                    (
                        key,
                        record["trajectory"]["repetition_id"],
                        record["provenance"]["sampling"].get("seed"),
                        record["record_id"],
                        str(path),
                    )
                )
        return extracted

    archive_paths = sorted((trajectory_root / "schema_v7/archives/phase=eval/step=eval_only/*.zip").glob(), key=str)
    retained = defaultdict(list)
    with ThreadPoolExecutor(max_workers=8) as pool:
        for records in pool.map(extract, archive_paths):
            for key, repetition, seed, record_id, archive in records:
                retained[key].append((repetition, seed, record_id, archive))
    if {key: len(value) for key, value in retained.items()} != {key: len(value) for key, value in replay_groups.items()}:
        raise ValueError("Retained/replay scientific-content multisets differ")
    replays = {}
    for key, records in retained.items():
        if len(records) != 1:
            raise ValueError("Replay row/repetition attribution is ambiguous; retain richer identity before stitching")
        repetition, seed, record_id, archive = records[0]
        row = replay_groups[key][0]
        identity = row["task_id"], repetition
        if identity in replays or seed != request_seed(MASTER_SEED, *identity):
            raise ValueError("Repeated identity or changed frozen seed")
        replays[identity] = (row, record_id, archive)
    if set(replays) != {(task_id, repetition) for task_id in task_ids for repetition in range(REPETITIONS_PER_TASK)}:
        raise ValueError("Recovery task/repetition set is incomplete")

    result, replacements = stitched_rows(
        originals, replays, targets, sampling_seed=partial(request_seed, MASTER_SEED)
    )
    for replacement in replacements:
        replay = replays[(replacement["task_id"], replacement["repetition_id"])][0]
        replacement["replay_still_infrastructure"] = is_infrastructure_error(replay)
    destination = StoragePath(args.output)
    destination.mkdirs()
    (destination / "DCAgent2_terminal_bench_2.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in result)
    )
    report = {
        "analysis_source_sha256": args.source_sha256,
        "original_run_output": args.original_run_output,
        "recovery_run_output": args.recovery_run_output,
        "registered_recovery_receipt": args.recovery_receipt,
        "trials": len(result),
        "original_graded_preserved": len(result) - len(replacements),
        "original_infrastructure_replaced": len(replacements),
        "extra_replayed_graded_ignored": len(replays) - len(replacements),
        "remaining_infrastructure": sum(is_infrastructure_error(row) for row in result),
        "frozen_task_repetition_seed_set_sha256": digest(
            [
                (task, repetition, request_seed(MASTER_SEED, task, repetition))
                for task, repetition in sorted(originals)
            ]
        ),
        "replacements": replacements,
    }
    (destination / "stitch-receipt.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "replacements"}))


def repetition_ledger(argv: list[str] | None = None) -> None:
    """Bind every retained TB2 trial to its frozen task/repetition identity and seed."""
    # The vendored SkyRL scripts ship in SkyRL's frozen CPU environment (cpu-submit
    # --skyrl); they are not Marin workspace dependencies.
    from vendor.skyrl.scripts.loop_audit import is_infrastructure_error, read_rows
    from skyrl_train.mismatch_probe.protocol import request_seed

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Saved TB2 eval jsonl for the original run")
    parser.add_argument("--trajectory-root", required=True, help="Retained trajectory archive root")
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-trials", type=int, default=REGISTERED_ORIGINAL_TRIALS)
    args = parser.parse_args(argv)
    configure_coreweave_s3()
    source = args.source
    trajectory_root = args.trajectory_root

    def extract(path):
        result = []
        with zipfile.ZipFile(io.BytesIO(path.read_bytes())) as archive:
            for name in archive.namelist():
                if not name.startswith("records/") or not name.endswith(".json.gz"):
                    continue
                record = json.loads(gzip.decompress(archive.read(name)))
                extras = record["trajectory"]["environment_extras"]
                if extras.get("data_source") != "DCAgent2/terminal_bench_2":
                    continue
                instance = record["trajectory"]["instance_id"]
                uid = instance.rsplit("-", 1)[-1]
                response = record["response"]
                verification = record.get("verification_result") or {}
                identity = (
                    uid,
                    digest(response["token_ids"]),
                    digest(response.get("messages")),
                    verification.get("status"),
                    record["disposition"].get("exception_type"),
                )
                result.append(
                    (
                        identity,
                        {
                            "record_id": record["record_id"],
                            "archive": str(path),
                            "instance_id": instance,
                            "uid": uid,
                            "repetition_id": record["trajectory"]["repetition_id"],
                            "sampling": record["provenance"]["sampling"],
                            "verification_status": verification.get("status"),
                            "exception_type": record["disposition"].get("exception_type"),
                        },
                    )
                )
        return result

    rows = list(read_rows(source))
    task_by_uid = {row["uid"]: row["task_id"] for row in rows}
    raw_groups = defaultdict(list)
    for row in rows:
        raw_groups[row_key(row)].append(row)
    paths = sorted((StoragePath(trajectory_root) / "schema_v7/archives/phase=eval/step=eval_only/*.zip").glob(), key=str)
    retained = defaultdict(list)
    with ThreadPoolExecutor(max_workers=8) as pool:
        for records in pool.map(extract, paths):
            for identity, metadata in records:
                metadata["task_id"] = task_by_uid[metadata["uid"]]
                retained[identity].append(metadata)
    if {identity: len(items) for identity, items in retained.items()} != {
        identity: len(items) for identity, items in raw_groups.items()
    }:
        raise ValueError("Retained/raw TB2 scientific-content multisets differ; do not infer repetition from completion ordinal")
    ledger = []
    for identity, metadata_list in retained.items():
        originals = raw_groups[identity]
        for item in metadata_list:
            expected = request_seed(MASTER_SEED, item["task_id"], item["repetition_id"])
            if item["sampling"].get("seed") != expected:
                raise ValueError("Recorded trial seed differs from frozen seed contract")
            item["scientifically_equivalent_dump_ordinals"] = sorted(row["ordinal"] for row in originals)
            item["join_key"] = list(identity)
            item["original_infrastructure_error"] = is_infrastructure_error(originals[0])
            ledger.append(item)
    counts = Counter((item["task_id"], item["repetition_id"]) for item in ledger)
    if len(ledger) != args.expected_trials or len(counts) != args.expected_trials or set(counts.values()) != {1}:
        raise ValueError("Retained repetitions are incomplete or duplicated")
    receipt = {
        "source": source,
        "trajectory_root": trajectory_root,
        "trial_count": len(ledger),
        "infrastructure_trials": sum(item["original_infrastructure_error"] for item in ledger),
        "frozen_seed_verified": True,
        "content_multisets_match": True,
        "trials": ledger,
    }
    StoragePath(args.output).write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({key: value for key, value in receipt.items() if key != "trials"}))


def recovery_selection(argv: list[str] | None = None) -> None:
    """Publish the recovery task selection and its registered recovery receipt."""
    import pyarrow.parquet as pq
    from rolloutengine.models import TASK_ROLLOUT_COLUMN, TaskRolloutSpec

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", required=True, help="Verified tb2-repetition-ledger receipt")
    parser.add_argument("--source", required=True, help="Frozen TB2 panel parquet the ledger came from")
    parser.add_argument("--output-root", required=True, help="Recovery selection prefix to publish")
    parser.add_argument("--expected-targets", type=int, default=REGISTERED_RECOVERY_TARGETS)
    parser.add_argument("--expected-tasks", type=int, default=REGISTERED_RECOVERY_TASKS)
    parser.add_argument("--selection-version", default=REGISTERED_SELECTION_VERSION)
    args = parser.parse_args(argv)
    configure_coreweave_s3()
    ledger = json.loads(StoragePath(args.ledger).read_text())
    if not ledger["frozen_seed_verified"] or not ledger["content_multisets_match"]:
        raise ValueError("Unverified source identities")
    failures = [row for row in ledger["trials"] if row["original_infrastructure_error"]]
    task_ids = {row["task_id"] for row in failures}
    if len(failures) != args.expected_targets or len(task_ids) != args.expected_tasks:
        raise ValueError("Recovery census differs from the registered rule")
    payload = StoragePath(args.source).read_bytes()
    table = pq.read_table(io.BytesIO(payload))
    values = table[TASK_ROLLOUT_COLUMN].to_pylist()
    indices = [i for i, value in enumerate(values) if TaskRolloutSpec.model_validate_json(value).task.id in task_ids]
    selected = table.take(indices)
    if len(selected) != args.expected_tasks:
        raise ValueError("Frozen task lookup incomplete")
    rows = selected[TASK_ROLLOUT_COLUMN].to_pylist()
    if rows != [values[i] for i in indices]:
        raise ValueError("TaskRolloutSpecs changed")
    root = StoragePath(args.output_root)
    root.mkdirs()
    buffer = io.BytesIO()
    pq.write_table(selected, buffer)
    data = buffer.getvalue()
    if (root / "tb2.parquet").exists() and (root / "tb2.parquet").read_bytes() != data:
        raise ValueError("Recovery selection already exists with different bytes")
    (root / "tb2.parquet").write_bytes(data)
    (root / "summary.json").write_text(
        json.dumps(
            {
                "grader_kind_counts": {"train": {}, "tb2": {}},
                "selection_version": args.selection_version,
                "tb2_rows": len(rows),
            },
            indent=2,
        )
        + "\n"
    )
    receipt = {
        "original_ledger": args.ledger,
        "original_source": args.source,
        "original_parquet_sha256": hashlib.sha256(payload).hexdigest(),
        "selection_sha256": hashlib.sha256(data).hexdigest(),
        "selected_task_ids": sorted(task_ids),
        "source_row_indices": indices,
        "selected_spec_sha256": [hashlib.sha256(value.encode()).hexdigest() for value in rows],
        "replayed_trials": len(rows) * REPETITIONS_PER_TASK,
        "replaced_original_infrastructure_trials": len(failures),
        "preserved_original_graded_trials": ledger["trial_count"] - len(failures),
        "extra_replayed_original_graded_trials_ignored": len(rows) * REPETITIONS_PER_TASK - len(failures),
        "repetitions_per_task": REPETITIONS_PER_TASK,
        "rule": "one replay; replace only frozen original ungraded task/repetition identities; preserve all original graded outcomes",
        "target_trials": failures,
    }
    (root / "recovery-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({key: value for key, value in receipt.items() if key not in ("target_trials", "selected_spec_sha256", "selected_task_ids", "source_row_indices")}))


def main(argv: list[str] | None = None) -> None:
    handlers = {
        "stitch-tb2-recovery": stitch,
        "tb2-repetition-ledger": repetition_ledger,
        "tb2-recovery-selection": recovery_selection,
    }
    handler = handlers.get((argv or [None])[0])
    if handler is None:
        raise SystemExit(f"expected one of {sorted(handlers)} as the first argument")
    handler(argv[1:])


if __name__ == "__main__":
    main()
