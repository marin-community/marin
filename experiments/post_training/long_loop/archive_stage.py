# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Archive-only continuation: no dependency can launch an earlier training stage."""

import hashlib
import json
import re
from pathlib import Path

import click
from fray.types import ResourceConfig
from marin.evaluation.records import EvalRunRecord
from marin.execution.lazy import OUT, ArtifactStep, apply
from marin.execution.remote import remote
from marin.rl.skyrl import SkyRLRun
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.storage_path import prefix_join

from experiments.post_training.long_loop.archive import ArchiveInput, checkpoint_cleanup, seal_archive, write_once
from experiments.post_training.long_loop.eval_archive import publish_evaluation
from experiments.post_training.long_loop.hf_archive import (
    HfUpload,
    file_identity,
    upload_checkpoint,
    verify_hf_checkpoint,
)

REQUIRED_CATEGORIES = {"source", "config", "dataset", "timing", "eval_trace", "wandb"}


def archive_run(*, output: str, plan: dict, completed_report: str) -> None:
    """Seal an explicit reviewed inventory, then clean only its superseded checkpoints."""
    fs, key = filesystem_for(completed_report + "/evaluation.json")
    evaluation = json.loads(fs.cat_file(key))
    if not evaluation["all_verified"]:
        raise ValueError("Completed report has unverified evaluation paths")
    inputs = [ArchiveInput(**item) for item in plan["inputs"]]
    missing = REQUIRED_CATEGORIES - {item.category for item in inputs}
    if missing:
        raise ValueError(f"Archive lacks required evidence categories: {sorted(missing)}")
    provenance = dict(plan["provenance"])
    revision = provenance["marin_commit"]
    if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        raise ValueError("Archive requires an exact Marin commit")
    if not provenance["evaluation_complete"]:
        raise ValueError("Archival post-script requires completed evaluation")
    receipt_fs, receipt_key = filesystem_for(output + "/hf-checkpoint.json")
    if receipt_fs.exists(receipt_key):
        receipt = json.loads(receipt_fs.cat_file(receipt_key))
        upload = plan["hf_upload"]
        if any(receipt[key] != upload[key] for key in ("source", "repo_id", "native_checkpoint")):
            raise ValueError("Existing HF receipt belongs to a different archive plan")
        verify_hf_checkpoint(receipt)
    else:
        receipt = upload_checkpoint(HfUpload(**plan["hf_upload"]), timing_destination=output + "/timing/hf-upload.jsonl")
        write_once(output + "/hf-checkpoint.json", json.dumps(receipt, sort_keys=True, indent=2).encode())
    provenance["hf_checkpoint"] = receipt
    publication = plan["evaldash"]
    record = EvalRunRecord.model_validate(publication["record"]).model_copy(
        update={"results_path": output + "/evaluation"}
    )
    record_uri = publish_evaluation(
        source=publication["source"], record=record, records_prefix=publication["records_prefix"]
    )
    inputs.append(ArchiveInput(record_uri, "evaldash/record.json", "eval_trace"))
    eval_fs, eval_root = filesystem_for(record.results_path)
    for key in sorted(eval_fs.find(eval_root)):
        inputs.append(
            ArchiveInput(
                eval_fs.unstrip_protocol(key), "evaldash/store/" + key.removeprefix(eval_root + "/"), "eval_trace"
            )
        )
    seal_archive(output, inputs, provenance)
    inventory = provenance["checkpoint_inventory"]
    checkpoint_cleanup(
        output,
        inventory["root"],
        inventory["selected"],
        [item["source"] for item in inventory["disposable_files"]],
    )


def archive_step(version: str, plan: dict, completed_report: ArtifactStep) -> ArtifactStep:
    return apply(
        "long-loop/archive", archive_run, version=version, output=OUT, plan=plan, completed_report=completed_report
    )


def archive_selected_run(*, output: str, plan: dict, trained: str, completed_report: str) -> None:
    """Resolve the selected export from the committed RL result, never a fixed smoke step."""
    run = SkyRLRun.raw_load(trained)
    if run.hf_model_uri is None or run.global_step is None:
        raise ValueError("Selected RL run has no committed HF export")
    fs, root = filesystem_for(run.checkpoint_root)
    best_uri = prefix_join(run.checkpoint_root, "best_checkpoint.json")
    best = json.loads(fs.cat_file(prefix_join(root, "best_checkpoint.json")))
    selected = prefix_join(run.checkpoint_root, f"global_step_{best['step']}")
    if (
        best["step"] != run.global_step
        or best["model_uri"] != prefix_join(selected, "policy")
        or best["ema_window"] != 5
    ):
        raise ValueError("Exported RL step differs from the five-step validation selection")
    request_sha = hashlib.sha256(
        json.dumps({"plan": plan, "trained": trained, "report": completed_report}, sort_keys=True).encode()
    ).hexdigest()
    archive_fs, archive_key = filesystem_for(output + "/resolved-plan.json")
    if archive_fs.exists(archive_key):
        resolved = json.loads(archive_fs.cat_file(archive_key))
        if (
            resolved["archive_request_sha256"] != request_sha
            or resolved["hf_upload"]["source"] != run.hf_model_uri
            or resolved["provenance"]["checkpoint_selection"]["result"] != best
        ):
            raise ValueError("Resolved archive plan changed on resume")
        archive_run(output=output, plan=resolved, completed_report=completed_report)
        return
    resolved = json.loads(json.dumps(plan))
    resolved["archive_request_sha256"] = request_sha
    resolved["hf_upload"].update(source=run.hf_model_uri, native_checkpoint=selected)
    disposable = []
    for key, info in sorted(fs.find(root, detail=True).items()):
        relative = key.removeprefix(root.rstrip("/") + "/")
        component = relative.split("/", 1)[0]
        if info.get("type") != "file" or re.fullmatch(r"global_step_[0-9]+", component) is None:
            continue
        if component == f"global_step_{best['step']}":
            continue
        disposable.append(
            {"source": prefix_join(run.checkpoint_root, relative), "storage_identity": file_identity(info)}
        )
    resolved["provenance"]["checkpoint_inventory"] = {
        "root": run.checkpoint_root,
        "selected": selected,
        "ttl": resolved["native_checkpoint_ttl"],
        "disposable_files": disposable,
    }
    for filename in ("best_checkpoint.json", "validation_selection_history.json"):
        resolved["inputs"].append(
            {"source": prefix_join(run.checkpoint_root, filename), "name": "selection/" + filename, "category": "config"}
        )
    resolved["provenance"]["checkpoint_selection"] = {"receipt": best_uri, "result": best}
    write_once(output + "/resolved-plan.json", json.dumps(resolved, sort_keys=True, indent=2).encode())
    archive_run(output=output, plan=resolved, completed_report=completed_report)


def selected_archive_step(
    version: str, plan: dict, trained: ArtifactStep[SkyRLRun], completed_report: ArtifactStep
) -> ArtifactStep:
    """Append a cluster-local HF upload and evidence archive after full evaluation."""
    return apply(
        "long-loop/archive",
        remote(
            archive_selected_run,
            name=f"long-loop-hf-archive-{version}",
            resources=ResourceConfig.with_gpu("H100", count=8, cpu=32, ram="64GB", disk="500GB", target_cluster=None),
        ),
        version=version,
        output=OUT,
        plan=plan,
        trained=trained,
        completed_report=completed_report,
    )


@click.command(help=__doc__)
@click.option("--plan", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option("--output", required=True, help="Durable destination without a TTL lifecycle.")
def main(plan: Path, output: str) -> None:
    inventory = json.loads(plan.read_text())
    archive_run(output=output, plan=inventory, completed_report=inventory["completed_report"])


if __name__ == "__main__":
    main()
