# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build whole-row task intersections from verified prebuilt arms, without rendering traces again."""

import argparse
import gzip
import hashlib
import io
import json
import tempfile
from pathlib import Path

from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath

from experiments.post_training.sft_shootout_rows.build import cut_replay, replay_rows, write_split
from experiments.post_training.sweep_data.prepare_agentic import output_hashes, read_component, upload_checked

FAMILIES = ("swesmith", "tmax")
SUBSET_RULE = "Intersect source and reference tasks by family; preserve whole rows and shifted masks; no subsampling"


def task_keys(index: list[dict]) -> set[tuple[str, str]]:
    """Return agentic task identities from epoch zero."""
    return {
        (entry["family"], entry["task_id"]) for entry in index if entry["family"] in FAMILIES and entry["epoch"] == 0
    }


def intersection_rows(entries: list[tuple[str, dict]], tasks: set[tuple[str, str]]) -> list[tuple[str, dict]]:
    """Select whole source rows, preserving their token and mask bytes."""
    selected = [pair for pair in entries if (pair[1]["family"], pair[1].get("task_id")) in tasks]
    found = [(entry["family"], entry["task_id"]) for _, entry in selected]
    if set(found) != tasks or len(found) != len(tasks):
        raise ValueError("Expected exactly one source row per shared task")
    return sorted(selected, key=lambda pair: (pair[1]["family"], pair[1]["task_id"]))


def source_index(uri: str, digest: str) -> tuple[dict, list[dict]]:
    root = StoragePath(uri)
    if not (root / "_SUCCESS").exists() or (root / "DRY_RUN_NOT_FOR_TRAINING").exists():
        raise ValueError(f"Not a completed production source: {uri}")
    name = "train-index.jsonl"
    manifest, raw, files = read_component(root, [name])
    if hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError(f"Manifest pin mismatch: {uri}")
    if manifest["dry_run"] or manifest["arm"] != "plain":
        raise ValueError("Wrong source arm or dry-run source")
    return manifest, [json.loads(line) for line in files[name].decode().splitlines()]


def read_split(uri: str, manifest: dict, index: list[dict], name: str, split_name: str) -> list[tuple[str, dict]]:
    _, raw, files = read_component(StoragePath(uri), [name])
    if json.loads(raw) != manifest:
        raise ValueError("Source manifest changed during the build")
    split = manifest["splits"][split_name]
    if len(index) != split["rows"]:
        raise ValueError("Source index and split disagree")
    digest = hashlib.sha256()
    entries = []
    count = 0
    with gzip.open(io.BytesIO(files[name]), "rb") as stream:
        for number, line in enumerate(stream):
            count += 1
            digest.update(line)
            if number >= len(index):
                continue
            entry = index[number]
            if entry["row"] != number:
                raise ValueError("Index row order differs from the source file")
            entries.append(
                (line.decode().rstrip("\n"), {key: value for key, value in entry.items() if key not in ("row", "epoch")})
            )
    if count != split["padded_rows"] or digest.hexdigest() != split["rows_sha256"]:
        raise ValueError("Source split rows hash or count mismatch")
    return entries


def build_subset(
    source: str,
    source_sha: str,
    reference: str,
    reference_sha: str,
    output: str,
    variant: str,
    replay: str,
    builder_commit: str,
) -> dict:
    """Write a fresh whole-row training subset with a seeded replay prefix."""
    root = StoragePath(output)
    if root.exists():
        raise ValueError("Use a fresh output prefix")
    manifest, index = source_index(source, source_sha)
    other, other_index = source_index(reference, reference_sha)
    for key in ("parent_model", "tokenizer_source", "pad_token_id", "max_sequence_tokens", "batch_size"):
        if manifest[key] != other[key]:
            raise ValueError(f"Source/reference differ in {key}")
    shared = task_keys(index) & task_keys(other_index)
    if not shared:
        raise ValueError("Task intersection is empty")
    heldout = {
        family: sorted(set(manifest["heldout_task_ids"][family]) | set(other["heldout_task_ids"][family]))
        for family in FAMILIES
    }
    if any(task in heldout[family] for family, task in shared):
        raise ValueError("Held-out task in the training intersection")
    entries = read_split(
        source, manifest, [entry for entry in index if entry["epoch"] == 0], "train-epoch-0-prebuilt.jsonl.gz", "epoch-0"
    )
    selected = intersection_rows(entries, shared)
    del entries
    family_tokens = {
        family: sum(entry["supervised_tokens"] for _, entry in selected if entry["family"] == family)
        for family in FAMILIES
    }
    budget = sum(family_tokens.values())
    lines, replay_index, replay_info = replay_rows(replay)
    if replay_info["manifest_sha256"] != manifest["replay"]["manifest_sha256"]:
        raise ValueError("Replay source differs from the source arm")
    replay_entries, cut_info = cut_replay(list(zip(lines, replay_index, strict=True)), budget, replay_info["base_rows"])
    replay_info.update(cut_info)
    del lines, replay_index
    shared_bytes = "".join(
        json.dumps(dict(family=family, task_id=task), sort_keys=True) + "\n" for family, task in sorted(shared)
    ).encode()
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        output_index, splits = [], {}
        for epoch in range(3):
            seed = f"20261023:{variant}:epoch-{epoch}:{hashlib.sha256(shared_bytes).hexdigest()}"
            epoch_index, splits[f"epoch-{epoch}"] = write_split(
                directory / f"train-epoch-{epoch}-prebuilt.jsonl.gz", selected + replay_entries, seed
            )
            output_index += [dict(epoch=epoch, **entry) for entry in epoch_index]
        index_name = "train-index.jsonl"
        (directory / index_name).write_text("".join(json.dumps(entry) + "\n" for entry in output_index))
        (directory / "selection.jsonl").write_text("".join(json.dumps(entry) + "\n" for _, entry in selected))
        (directory / "shared-task-ids.jsonl").write_bytes(shared_bytes)
        result = {
            key: manifest[key]
            for key in (
                "parent_model",
                "tokenizer_source",
                "pad_token_id",
                "max_sequence_tokens",
                "batch_size",
                "pool_manifest_sha256",
                "harness",
                "pool_notes",
                "excluded_trials",
            )
            if key in manifest
        }
        result.update(
            arm="plain",
            dry_run=False,
            task="279",
            heldout_task_ids=heldout,
            B_supervised_tokens=budget,
            splits=splits,
            replay=replay_info,
            validation="disabled: no held-out loss on either side",
            task_slots=sorted(task for _, task in shared),
            builder_commit=builder_commit,
            builder_sources_sha256={"subset.py": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
            builder_changelog=[*manifest.get("builder_changelog", []), SUBSET_RULE],
            subset=dict(
                rule=SUBSET_RULE,
                variant=variant,
                source=source,
                source_sha256=source_sha,
                reference=reference,
                reference_sha256=reference_sha,
                shared_task_ids_sha256=hashlib.sha256(shared_bytes).hexdigest(),
                family_tokens=family_tokens,
                source_agentic_tasks=len(task_keys(index)),
                retained_tasks=len(shared),
                dropped_tasks=len(task_keys(index) - shared),
            ),
        )
        replay_info["share_of_supervised"] = {
            name: replay_info["supervised_tokens"] / split["supervised_total"] for name, split in splits.items()
        }
        result["output_hashes"] = output_hashes(directory)
        upload_checked(directory, root, result)
        (root / "_SUCCESS").write_text("")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("source", "source-sha", "reference", "reference-sha", "output", "builder-commit"):
        parser.add_argument(f"--{option}", required=True)
    parser.add_argument("--variant", choices=("lowshared", "contribshared"), required=True)
    parser.add_argument("--replay", required=True)
    args = parser.parse_args()
    configure_coreweave_s3()
    result = build_subset(
        args.source,
        args.source_sha,
        args.reference,
        args.reference_sha,
        args.output,
        args.variant,
        args.replay,
        args.builder_commit,
    )
    print("SUBSET_BUILT " + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
