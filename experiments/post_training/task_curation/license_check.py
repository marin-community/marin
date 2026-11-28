# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run the dataset-level license rules; uncertain cases go to the user."""

import hashlib
import json
import logging
import subprocess
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from urllib.parse import urlsplit

import click
import yaml
from finestore.rl.licenses import (
    LicenseBan,
    LicenseCheck,
    LicenseCheckRule,
    LicenseCheckVerdict,
    LicenseReceipt,
    check_license,
)
from huggingface_hub.errors import HfHubHTTPError
from requests import RequestException

from experiments.post_training.task_curation.license_resolution import resolve_license
from experiments.post_training.task_curation.license_targets import LicenseTarget
from experiments.post_training.task_curation.sources import all_sources
from infra.marina.apps.rl_data_catalog.server.licensing import source_license_ban

CHECKER_VERSION = "atlas-license-check-v11"
logger = logging.getLogger(__name__)


def input_target(value: str) -> LicenseTarget:
    location, separator, revision = value.rpartition("@")
    if not separator or not revision:
        raise click.BadParameter("Use a dataset URL or HF ID followed by @<immutable pin>")
    if "://" not in location:
        location = "https://huggingface.co/datasets/" + location
    parts = urlsplit(location)
    components = parts.path.strip("/").split("/")
    if parts.hostname == "huggingface.co" and components[0] == "datasets":
        dataset_id = "/".join(components[1:3])
    elif parts.hostname == "github.com":
        dataset_id = "/".join(components[:2])
    else:
        dataset_id = location
    return LicenseTarget(dataset_id, dataset_id, revision, location)


def target_ban(target: LicenseTarget, receipt: LicenseReceipt | None = None) -> LicenseBan | None:
    return source_license_ban(target.source_id, target.dataset_id, receipt)


def resolve_target(target: LicenseTarget, cache: Path | None) -> LicenseReceipt:
    key = [CHECKER_VERSION, target.dataset_id, target.revision, target.url, target.license_url]
    path = cache / (hashlib.sha256(json.dumps(key).encode()).hexdigest() + ".json") if cache else None
    if path is not None and path.exists():
        receipt = LicenseReceipt.model_validate_json(path.read_text())
        if (receipt.dataset_id, receipt.revision) != (target.dataset_id, target.revision):
            raise ValueError("Cached license receipt belongs to another pin")
        return receipt.model_copy(update={"source_id": target.source_id})
    receipt = resolve_license(target)
    if path is not None:
        path.write_text(receipt.model_dump_json(indent=2) + "\n")
    return receipt


def check_target(target: LicenseTarget, cache: Path | None) -> dict:
    receipt = None
    try:
        receipt = resolve_target(target, cache)
    except (ValueError, RequestException, HfHubHTTPError, subprocess.CalledProcessError, yaml.YAMLError) as error:
        evidence_result = LicenseCheck(
            source_id=target.source_id,
            dataset_id=target.dataset_id,
            revision=target.revision,
            verdict=LicenseCheckVerdict.ESCALATE,
            rule_id=LicenseCheckRule.UNREACHABLE,
            quote=f"{type(error).__name__}: {str(error).splitlines()[0]}",
            evidence_url=target.url,
        )
    else:
        evidence_result = check_license(receipt)
    return _target_row(target, evidence_result, receipt)


def _target_row(target: LicenseTarget, evidence_result: LicenseCheck, receipt: LicenseReceipt | None) -> dict:
    """One source's ruled row: the ban overrides the evidence verdict, which it keeps beside it."""
    result = evidence_result
    if ban := target_ban(target, receipt):
        result = evidence_result.model_copy(
            update={
                "verdict": LicenseCheckVerdict.FAIL,
                "rule_id": LicenseCheckRule.BANNED,
                "quote": ban.reason,
                "evidence_url": ban.issue_url,
            }
        )
    return {
        **result.model_dump(mode="json"),
        "evidence_verdict": evidence_result.model_dump(mode="json"),
        "receipt": receipt.model_dump(mode="json") if receipt is not None else None,
    }


def check_targets(targets: list[LicenseTarget], cache: Path | None, workers: int) -> list[dict]:
    # One network resolution per dataset pin; the scoped source identity is restored below.
    groups = {}
    for target in targets:
        key = (target.dataset_id, target.revision, target.url, target.license_url)
        groups.setdefault(key, []).append(target)
    rows = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(check_target, group[0], cache): group for group in groups.values()}
        for future in as_completed(futures):
            group = futures[future]
            first = future.result()
            for target in group:
                # One network resolution per dataset pin; the scoped source identity
                # is restored here because F3 is source-specific even when multiple
                # sources share one repository.
                receipt_data = first["receipt"]
                if receipt_data is not None:
                    receipt = LicenseReceipt.model_validate({**receipt_data, "source_id": target.source_id})
                    row = _target_row(target, check_license(receipt), receipt)
                else:
                    evidence = LicenseCheck.model_validate(first["evidence_verdict"])
                    row = _target_row(
                        target,
                        evidence.model_copy(update={"source_id": target.source_id}),
                        None,
                    )
                rows[target.source_id] = row
                logger.info("%s %s %s", target.source_id, row["verdict"], row["rule_id"])
    return [rows[target.source_id] for target in targets]


@click.command()
@click.argument("inputs", nargs=-1)
@click.option("--source-id", help="Bind a single input's receipt to its declared Atlas source ID.")
@click.option("--catalog", is_flag=True, help="Check the current Atlas dataset pins, without following lineage.")
@click.option(
    "--input-manifest", type=click.Path(exists=True, path_type=Path), help="JSON array of LicenseTarget fields."
)
@click.option("--cache", type=click.Path(path_type=Path))
@click.option("--output", type=click.Path(path_type=Path))
@click.option(
    "--receipts-output", type=click.Path(path_type=Path), help="Write evidence receipts; does not approve or admit them."
)
@click.option("--workers", type=click.IntRange(1, 4), default=2)
def main(
    inputs: tuple[str, ...],
    source_id: str | None,
    catalog: bool,
    input_manifest: Path | None,
    cache: Path | None,
    output: Path | None,
    receipts_output: Path | None,
    workers: int,
) -> None:
    targets = [input_target(value) for value in inputs]
    if source_id is not None:
        if len(targets) != 1 or catalog or input_manifest is not None:
            raise click.UsageError("--source-id requires exactly one positional input")
        targets = [replace(targets[0], source_id=source_id)]
    if catalog:
        for source in all_sources().values():
            dataset = source.dataset
            if dataset is None or dataset.revision is None:
                raise ValueError(f"Catalog source has no pinned dataset: {source.info.id}")
            targets.append(LicenseTarget(source.info.id, dataset.name, dataset.revision, dataset.url))
    if input_manifest is not None:
        targets.extend(LicenseTarget(**row) for row in json.loads(input_manifest.read_text()))
    if not targets:
        raise click.UsageError("Provide <dataset>@<pin>, --catalog, or --input-manifest")
    if len({target.source_id for target in targets}) != len(targets):
        raise click.UsageError("Input source IDs must be unique")
    if cache is not None:
        cache.mkdir(parents=True, exist_ok=True)
    ordered = check_targets(targets, cache, workers)
    summary = dict(Counter(row["verdict"] for row in ordered))
    report = {"checker_version": CHECKER_VERSION, "counts": summary, "sources": ordered}
    if output is not None:
        output.write_text(json.dumps(report, indent=2) + "\n")
    if receipts_output is not None:
        receipts_output.write_text(
            json.dumps({row["source_id"]: row["receipt"] for row in ordered if row["receipt"] is not None}, indent=2)
            + "\n"
        )
    for row in ordered:
        click.echo(json.dumps({key: value for key, value in row.items() if key != "receipt"}))
    click.echo("\n| Verdict | Sources |\n|---|---:|")
    for verdict in LicenseCheckVerdict:
        click.echo(f"| {verdict.value} | {summary.get(verdict.value, 0)} |")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
