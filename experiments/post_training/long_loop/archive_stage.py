# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Archive-only continuation: no dependency can launch an earlier training stage."""

import json
from pathlib import Path

import click
from marin.evaluation.records import EvalRunRecord
from marin.execution.lazy import OUT, ArtifactStep, apply
from rigging.filesystem.buckets import filesystem_for

from experiments.post_training.long_loop.archive import ArchiveInput, checkpoint_cleanup, seal_archive
from experiments.post_training.long_loop.eval_archive import publish_evaluation

REQUIRED_CATEGORIES = {"source", "config", "dataset", "timing", "eval_trace", "checkpoint", "wandb"}


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
    provenance = plan["provenance"]
    revision = provenance["marin_commit"]
    if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        raise ValueError("Archive requires an exact Marin commit")
    if not provenance["evaluation_complete"]:
        raise ValueError("Archival post-script requires completed evaluation")
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


@click.command(help=__doc__)
@click.option("--plan", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option("--output", required=True, help="Durable destination without a TTL lifecycle.")
def main(plan: Path, output: str) -> None:
    inventory = json.loads(plan.read_text())
    archive_run(output=output, plan=inventory, completed_report=inventory["completed_report"])


if __name__ == "__main__":
    main()
