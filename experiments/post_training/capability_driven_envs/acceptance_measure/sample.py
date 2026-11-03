# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Draw the measurement's capabilities from the n=25 proposal run: one code-verified proposal per capability.

The population is every capability under ``--proposals`` whose ``result.json`` exists (the proposal shard wrote
it last, so the capability's slots are final) with at least one eligible slot: ``slot-NN.proposal.json`` whose
header says ``verification == "code"``, ``null_reason`` is null and ``composite`` is empty, and whose
``slot-NN.verdict.json`` decision is ``accept`` (the deterministic preflight; a refused proposal would never reach
an author).

Order (acceptance.md section 3.2, "by SHA256 order, stratified by domain prefix"): within each domain prefix
(``d01`` of ``d01.algebra.recurrences``) the capabilities sort by ``sha256("<seed>:<capability>")``; each one's
key is then ``((rank + 0.5) / domain size, sha)``, so every prefix of the global order (wave 1 is the first 32)
holds each domain close to its population share. Each capability contributes the eligible slot with the
smallest ``sha256("<seed>:<capability>:<slot>")``.

``sample.json`` records the population counts, a digest of the eligible set and, for each sampled capability, the
proposal's path and the sha256 of its bytes, which the driver checks before authoring. The run is still being
written while it is in flight: ``population.capabilities_with_result`` against ``capabilities_listed`` (and the
catalog's count with ``--catalog``) says how complete the snapshot is, and the driver refuses a sample whose
population was not complete unless told otherwise.

    uv run --frozen python -m experiments.post_training.capability_driven_envs.acceptance_measure.sample \\
        --proposals s3://marin-us-east-02a/marin/taskforge/proposals/curriculum-v3-n25-20261102/run \\
        --output ~/.cache/taskforge-acceptance-measure/sample.json
"""

import hashlib
import json
import re
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import click
import fsspec
from rigging.filesystem.s3_compat import configure_coreweave_s3

from experiments.post_training.capability_driven_envs.catalog import load_capability_ideas

DEFAULT_PROPOSALS = "s3://marin-us-east-02a/marin/taskforge/proposals/curriculum-v3-n25-20261102/run"
DEFAULT_SEED = "acceptance-measure-20261102"
SAMPLE_SIZE = 192
WAVE_SIZE = 32
RESULT_FILE = "result.json"
SLOT_PROPOSAL = re.compile(r"slot-(\d+)\.proposal\.json\Z")
DOMAIN = re.compile(r"\A(d\d+)\.")
READ_BATCH = 256


def sha256_hex(data: bytes | str) -> str:
    return hashlib.sha256(data.encode() if isinstance(data, str) else data).hexdigest()


@dataclass(frozen=True)
class Eligible:
    capability: str
    slot: int
    path: str
    sha256: str


@dataclass(frozen=True)
class Sampled:
    rank: int
    capability: str
    domain: str
    slot: int
    proposal: str
    sha256: str
    eligible_slots: int


def domain_of(capability: str) -> str:
    match = DOMAIN.match(capability)
    if match is None:
        raise ValueError(f"capability id {capability!r} has no dNN domain prefix")
    return match.group(1)


def eligible_slot(proposal: Mapping, verdict: Mapping) -> bool:
    header = proposal["header"]
    return (
        header["verification"] == "code"
        and header["null_reason"] is None
        and not header.get("composite")
        and verdict["decision"] == "accept"
    )


def stratified_order(capabilities: Sequence[str], seed: str) -> list[str]:
    """``capabilities`` in the stratified SHA256 order of the module docstring."""
    by_domain: dict[str, list[str]] = defaultdict(list)
    for capability in capabilities:
        by_domain[domain_of(capability)].append(capability)
    keys: dict[str, tuple[float, str]] = {}
    for members in by_domain.values():
        members.sort(key=lambda c: sha256_hex(f"{seed}:{c}"))
        for rank, capability in enumerate(members):
            keys[capability] = ((rank + 0.5) / len(members), sha256_hex(f"{seed}:{capability}"))
    return sorted(capabilities, key=keys.__getitem__)


def draw(eligible: Sequence[Eligible], seed: str, size: int) -> list[Sampled]:
    """One proposal per capability for the first ``size`` capabilities of the stratified order."""
    slots: dict[str, list[Eligible]] = defaultdict(list)
    for item in eligible:
        slots[item.capability].append(item)
    order = stratified_order(sorted(slots), seed)[:size]
    sampled = []
    for rank, capability in enumerate(order):
        choice = min(slots[capability], key=lambda e: sha256_hex(f"{seed}:{capability}:{e.slot}"))
        sampled.append(
            Sampled(
                rank, capability, domain_of(capability), choice.slot, choice.path, choice.sha256, len(slots[capability])
            )
        )
    return sampled


def population_digest(eligible: Sequence[Eligible]) -> str:
    return sha256_hex(json.dumps(sorted((e.capability, e.slot, e.sha256) for e in eligible)))


DRIVER_ENTRIES = frozenset({"shards"})


def scan(root: str, read: Callable[[list[str]], dict[str, bytes]], listing: Callable[[str], list[str]]) -> dict:
    """The eligible slots under ``root`` and the counts that say how complete the run was."""
    entries = listing(root)
    # The driver also writes ``shards/`` and ``control.json`` under the root; neither is a capability.
    names = (p.rstrip("/").rsplit("/", 1)[-1] for p in entries)
    capability_dirs = sorted(n for n in names if n not in DRIVER_ENTRIES and not n.endswith(".json"))
    with_result, eligible, failures = [], [], 0
    for capability in capability_dirs:
        files = {p.rsplit("/", 1)[-1] for p in listing(f"{root}/{capability}")}
        if RESULT_FILE not in files:
            continue
        with_result.append(capability)
        slots = sorted(int(m.group(1)) for f in files if (m := SLOT_PROPOSAL.match(f)))
        paths = []
        for slot in slots:
            verdict = f"slot-{slot:02d}.verdict.json"
            if verdict not in files:
                failures += 1
                continue
            paths += [f"{root}/{capability}/slot-{slot:02d}.proposal.json", f"{root}/{capability}/{verdict}"]
        blobs = {}
        for start in range(0, len(paths), READ_BATCH):
            blobs |= read(paths[start : start + READ_BATCH])
        for slot in slots:
            path = f"{root}/{capability}/slot-{slot:02d}.proposal.json"
            if path not in blobs:
                continue
            raw = blobs[path]
            verdict = json.loads(blobs[f"{root}/{capability}/slot-{slot:02d}.verdict.json"])
            if eligible_slot(json.loads(raw), verdict):
                eligible.append(Eligible(capability, slot, path, sha256_hex(raw)))
    return {
        "capabilities_listed": len(capability_dirs),
        "capabilities_with_result": len(with_result),
        "proposals_without_verdict": failures,
        "eligible": eligible,
    }


def sample_document(
    scanned: dict, *, root: str, seed: str, size: int, wave: int, catalog_capabilities: int | None
) -> dict:
    eligible: list[Eligible] = scanned["eligible"]
    if not eligible:
        raise click.ClickException(
            f"no eligible proposal under {root}: wrong keys, wrong prefix or an empty run; refusing an empty sample"
        )
    sampled = draw(eligible, seed, size)
    eligible_capabilities = len({e.capability for e in eligible})
    complete = scanned["capabilities_with_result"] == scanned["capabilities_listed"] and (
        catalog_capabilities is None or catalog_capabilities == scanned["capabilities_listed"]
    )
    return {
        "format": "acceptance-measure-sample-v1",
        "seed": seed,
        "proposals_root": root,
        "drawn_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "population": {
            "capabilities_listed": scanned["capabilities_listed"],
            "capabilities_with_result": scanned["capabilities_with_result"],
            "catalog_capabilities": catalog_capabilities,
            "proposals_without_verdict": scanned["proposals_without_verdict"],
            "eligible_capabilities": eligible_capabilities,
            "eligible_proposals": len(eligible),
            "eligible_digest": population_digest(eligible),
            "complete": complete,
        },
        "size_requested": size,
        "size": len(sampled),
        "wave_size": wave,
        "sample": [asdict(item) for item in sampled],
    }


def wave_capabilities(document: Mapping, wave: int) -> list[Mapping]:
    """The sampled entries of wave ``wave`` (1-based): wave 1 is the first ``wave_size``, wave 2 the rest."""
    entries = document["sample"]
    size = document["wave_size"]
    if wave == 1:
        return entries[:size]
    if wave == 2:
        return entries[size:]
    raise ValueError(f"wave is 1 or 2, got {wave}")


def _filesystem(url: str) -> tuple[fsspec.AbstractFileSystem, str]:
    if url.startswith("s3://"):
        configure_coreweave_s3()
    return fsspec.core.url_to_fs(url.rstrip("/"))


@click.command()
@click.option("--proposals", default=DEFAULT_PROPOSALS, show_default=True, help="The n=25 run root.")
@click.option("--catalog", help="The capability catalog, to check the run covered every capability.")
@click.option("--seed", default=DEFAULT_SEED, show_default=True)
@click.option("--size", type=click.IntRange(min=1), default=SAMPLE_SIZE, show_default=True)
@click.option("--wave-size", type=click.IntRange(min=1), default=WAVE_SIZE, show_default=True)
@click.option("--output", type=click.Path(dir_okay=False, path_type=Path), required=True)
def main(proposals: str, catalog: str | None, seed: str, size: int, wave_size: int, output: Path) -> None:
    """Draw the sample and write ``--output``; print the eligible-capability count."""
    fs, root = _filesystem(proposals)
    protocol = "s3://" if proposals.startswith("s3://") else ""
    catalog_capabilities = None
    if catalog is not None:
        cfs, cpath = _filesystem(catalog)
        local = output.parent / "catalog.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        local.write_bytes(cast(bytes, cfs.cat_file(cpath)))
        catalog_capabilities = len(load_capability_ideas(local))
    scanned = scan(root, lambda paths: dict(fs.cat(paths)), lambda path: fs.ls(path, detail=False))
    scanned["eligible"] = [Eligible(e.capability, e.slot, protocol + e.path, e.sha256) for e in scanned["eligible"]]
    document = sample_document(
        scanned, root=proposals, seed=seed, size=size, wave=wave_size, catalog_capabilities=catalog_capabilities
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document, indent=2) + "\n")
    population = document["population"]
    click.echo(
        f"eligible capabilities {population['eligible_capabilities']} (proposals {population['eligible_proposals']}); "
        f"capabilities with result.json {population['capabilities_with_result']} of {population['capabilities_listed']} "
        f"listed{'' if catalog_capabilities is None else f' and {catalog_capabilities} in the catalog'}; "
        f"complete={population['complete']}; sampled {document['size']} of {size}; wave 1 = first "
        f"{min(wave_size, document['size'])}; sha256 {sha256_hex(output.read_bytes())}"
    )


if __name__ == "__main__":
    main()
