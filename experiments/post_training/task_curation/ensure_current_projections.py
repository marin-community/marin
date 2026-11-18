# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Ensure every current declaration pin carries its serving projections.

The Atlas Grader and curriculum-area columns project receipts against the exact current
declaration pin: a derived-contract census receipt matching the actor census input, and a
curriculum alignment matching the dataset pin and pipeline. Any flow that changes a source's
pin (a re-pin, a retained or remediated release, a new source) must produce both receipts in
the same landing, or the source silently loses its grader profile and curriculum area. This
stage measures the gaps, emits the idempotent work plan, submits the missing in-region CPU
jobs, and refuses a catalog publication while a current pin lacks either projection.
"""

import argparse
import json
import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from finestore.rl.curation import ActorCensusInput
from marin.execution.artifact import read_record
from marina.client import applet_query

from experiments.post_training.task_curation.datasets.derived_contract_measurements import (
    derived_contract_receipts,
)
from experiments.post_training.task_curation.derived_contract_census import (
    DerivedContractCensusInput,
    DerivedContractReceipt,
)
from experiments.post_training.task_curation.lexical_artifacts import SourceTextPopulation
from infra.marina.apps.rl_data_catalog.server.source_identity import canonical_source_id

LOGGER = logging.getLogger(__name__)

CURRICULUM_DECLARATION_FIELDS = ("dataset_id", "dataset_revision", "pipeline")

ATLAS_APPLET_ID = "fb11c931-5861-4878-8bb5-a964d652b45f"
MARINA_URL = "https://marina.oa.dev"


@dataclass(frozen=True)
class DeclarationPin:
    """The current declaration pin a projection must match."""

    source_id: str
    dataset_id: str | None
    dataset_revision: str | None
    pipeline: dict | None
    actor_binding: Any  # the ActorCensusInput the declaration binds, or None

    @property
    def measured(self) -> bool:
        return self.actor_binding is not None


@dataclass(frozen=True)
class ProjectionGap:
    """One current pin whose serving projection is missing or computed at an older pin."""

    source_id: str
    projection: Literal["derived-contract", "curriculum", "served-catalog"]
    state: Literal["missing", "stale-pin"]
    last_computed_revision: str | None


def contract_state(declaration: DeclarationPin, receipts: dict[str, Any]) -> tuple[str, str | None]:
    """Whether a derived-contract census receipt matches this declaration's actor census input.

    The serving projection compares the six measured-input fields and ignores evidence
    locations, so the audit replays exactly that comparison.
    """
    if not declaration.measured:
        return "unmeasured", ""
    receipt = receipts.get(canonical_source_id(declaration.source_id)) or receipts.get(declaration.source_id)
    if receipt is None:
        return "missing", ""
    binding = declaration.actor_binding
    if receipt.input.model_dump(exclude={"evidence_url"}) == binding.model_dump(exclude={"evidence_url"}):
        return "current", ""
    return "stale-pin", receipt.input.dataset_revision


def curriculum_state(declaration: DeclarationPin, curriculum_rows: dict[str, dict]) -> tuple[str, str | None]:
    """Whether a curriculum alignment row matches this declaration's dataset pin and pipeline."""
    if not declaration.measured:
        return "unmeasured", ""
    row = curriculum_rows.get(canonical_source_id(declaration.source_id))
    if row is None:
        return "missing", None
    stored = row.get("declaration") or {}
    triple_equal = all(
        stored.get(key) == getattr(declaration, key if key != "pipeline" else "pipeline")
        for key in CURRICULUM_DECLARATION_FIELDS
    )
    if triple_equal and row.get("alignment") is not None:
        return "current", None
    return "stale-pin", stored.get("dataset_revision")


def projection_gaps(
    declarations: list[DeclarationPin],
    contract_receipts: dict[str, Any],
    curriculum_rows: dict[str, dict],
    served_pins: dict[str, DeclarationPin] | None = None,
) -> tuple[list[ProjectionGap], list[str]]:
    """Classify every current pin's projection state; return gaps then informational notes.

    With ``served_pins`` the audit also compares the served catalog's declaration against the
    canonical pin: a served row that predates the pin (no census binding, an older revision)
    makes every exact-pin projection pending no matter which receipts are loaded, so it is a
    gap of its own.
    """
    gaps: list[ProjectionGap] = []
    notes: list[str] = []
    for declaration in declarations:
        if served_pins is not None and declaration.measured:
            served = served_pins.get(canonical_source_id(declaration.source_id))
            if (
                served is None
                or served.actor_binding is None
                or (
                    served.actor_binding.model_dump(exclude={"evidence_url"})
                    != declaration.actor_binding.model_dump(exclude={"evidence_url"})
                )
            ):
                gaps.append(
                    ProjectionGap(
                        source_id=declaration.source_id,
                        projection="served-catalog",
                        state="stale-pin" if served is not None else "missing",
                        last_computed_revision=served.dataset_revision if served is not None else None,
                    )
                )
                continue
        state, last_computed = contract_state(declaration, contract_receipts)
        if state in {"missing", "stale-pin"}:
            gaps.append(
                ProjectionGap(
                    source_id=declaration.source_id,
                    projection="derived-contract",
                    state=state,
                    last_computed_revision=last_computed,
                )
            )
        elif state == "unmeasured":
            notes.append(
                f"derived-contract: {declaration.source_id} has no actor census binding yet; "
                "stage its population and census it before the projections can run"
            )
        state, last_computed = curriculum_state(declaration, curriculum_rows)
        if state in {"missing", "stale-pin"}:
            gaps.append(
                ProjectionGap(
                    source_id=declaration.source_id,
                    projection="curriculum",
                    state=state,
                    last_computed_revision=last_computed,
                )
            )
        elif state == "unmeasured":
            notes.append(
                f"curriculum: {declaration.source_id} has no actor census binding yet; "
                "align it after its intake census"
            )
    return gaps, notes


def summary(gaps: list[ProjectionGap], notes: list[str]) -> dict[str, Any]:
    counts: Counter = Counter((gap.projection, gap.state) for gap in gaps)
    return {
        "counts": {f"{projection}.{state}": count for (projection, state), count in sorted(counts.items())},
        "gaps": [
            {
                "source_id": gap.source_id,
                "projection": gap.projection,
                "state": gap.state,
                "last_computed_revision": gap.last_computed_revision,
            }
            for gap in sorted(gaps, key=lambda gap: (gap.projection, gap.source_id))
        ],
        "notes": notes,
        "state": "ok" if not gaps else "incomplete",
    }


def served_pin_map(served_pins: list[DeclarationPin]) -> dict[str, DeclarationPin]:
    return {canonical_source_id(pin.source_id): pin for pin in served_pins}


def _load_population(population_path: str):
    return SourceTextPopulation.raw_load(population_path)


def census_input_rows(declarations: list[DeclarationPin], sources: dict[str, Any]) -> tuple[list[dict], list[str]]:
    """Build the derived-contract census inputs over each measured source's bound population.

    The actor census binding's evidence artifact records the exact population the census
    measured (``input_artifact``), so the derived-contract census reuses that frozen
    population with the same binding — idempotent inputs for the in-region CPU job. Every
    row is validated against the census's own input model before it leaves this stage, so a
    population that disagrees with its binding can never reach a job; the mismatched sources
    are returned for the report instead.
    """
    receipts = derived_contract_receipts()
    rows = []
    unrunnable = []
    for declaration in declarations:
        state, _ = contract_state(declaration, receipts)
        if state not in {"missing", "stale-pin"}:
            continue
        source = sources[declaration.source_id]
        binding = source.info.actor_census_input
        actor_artifact = str(binding.evidence_url).removesuffix("/.artifact.json")
        record = read_record(actor_artifact)
        population = _load_population(record.result["input_artifact"])
        row = {"population": population.model_dump(mode="json"), "input": binding.model_dump(mode="json")}
        try:
            DerivedContractCensusInput.model_validate(row)
        except Exception as error:
            unrunnable.append(
                f"derived-contract census: {declaration.source_id} population "
                f"{population.dataset_id}@{population.dataset_revision} disagrees with its census binding "
                f"{binding.dataset_id}@{binding.dataset_revision} ({type(error).__name__}); "
                "re-measure the population at the declared pin before the census can run"
            )
            continue
        rows.append(row)
    rows.sort(key=lambda row: -row["population"]["task_count"])
    return rows, unrunnable


def publication_preflight(
    declarations: list[DeclarationPin],
    contract_receipts: dict[str, Any],
    curriculum_rows: dict[str, dict],
    waivers: frozenset[str] = frozenset(),
    served_pins: dict[str, DeclarationPin] | None = None,
) -> list[str]:
    """Refuse a preview or production release while a current pin lacks either projection.

    ``waivers`` names "source:projection" pairs (or bare source ids) deliberately published
    without one of the projections; a waiver covers only what it names and is recorded in the
    release manifest. Returns the informational notes so the caller can record them.
    """
    gaps, notes = projection_gaps(declarations, contract_receipts, curriculum_rows, served_pins=served_pins)
    blocking = [
        gap for gap in gaps if f"{gap.source_id}:{gap.projection}" not in waivers and gap.source_id not in waivers
    ]
    if blocking:
        counts: Counter = Counter((gap.projection, gap.state) for gap in blocking)
        detail = ", ".join(f"{projection}.{state}={count}" for (projection, state), count in sorted(counts.items()))
        named = "\n".join(
            f"  {gap.projection}/{gap.state}: {gap.source_id}"
            + (f" (last computed at {gap.last_computed_revision})" if gap.last_computed_revision else "")
            for gap in sorted(blocking, key=lambda gap: (gap.projection, gap.source_id))[:40]
        )
        raise ValueError(
            f"Atlas publication preflight refused: {len(blocking)} current pins lack a serving "
            f"projection ({detail}). Run ensure_current_projections for the missing pins or add "
            f"typed waivers:\n{named}"
        )
    return notes


def atlas_applet_query():
    return applet_query(MARINA_URL, ATLAS_APPLET_ID)


def atlas_curriculum_rows() -> dict[str, dict]:
    """The served curriculum rows from the live atlas applet (the current release)."""
    result = atlas_applet_query()(
        "SELECT source_id, declaration, alignment IS NOT NULL AS has_alignment FROM curriculum_sources "
        "WHERE load_id=(SELECT load_id FROM curriculum_current WHERE name='latest')",
        {},
    )
    return {row["source_id"]: row for row in result["rows"]}


def atlas_contract_receipts() -> dict[str, Any]:
    """The derived-contract receipts the served applet carries (the registry family before the first load)."""
    try:
        result = atlas_applet_query()("SELECT source_id, receipt FROM atlas_derived_contracts", {})
    except Exception as error:
        LOGGER.warning("atlas_derived_contracts is not loaded yet (%s); using the registry family", error)
        return derived_contract_receipts()
    return {
        row["source_id"]: DerivedContractReceipt.model_validate(
            row["receipt"] if isinstance(row["receipt"], dict) else json.loads(row["receipt"])
        )
        for row in result["rows"]
    }


def served_declaration_pins() -> list[DeclarationPin]:
    """The served catalog's declaration pins (the live applet's atlas_sources rows)."""
    result = atlas_applet_query()("SELECT id, declaration FROM atlas_sources", {})
    pins = []
    for row in result["rows"]:
        served = row.get("declaration") or {}
        binding = served.get("actor_census_input")
        pins.append(
            DeclarationPin(
                source_id=row["id"],
                dataset_id=served.get("dataset_id"),
                dataset_revision=served.get("dataset_revision"),
                pipeline=served.get("pipeline"),
                actor_binding=_binding_from_served(binding),
            )
        )
    return pins


def _binding_from_served(binding: dict | None):
    return ActorCensusInput.model_validate(binding) if binding else None


def assembled_declaration_pins() -> tuple[list[DeclarationPin], dict[str, Any]]:
    """The canonical layer's declaration pins plus the assembled sources for the census plan."""
    from experiments.post_training.task_curation.sources import (  # noqa: PLC0415 — the assembly pulls the registry families and the S3 credential wiring.
        all_sources,
    )

    sources = all_sources()
    pins = []
    for source_id, source in sources.items():
        dataset = source.dataset
        pipeline = source.pipeline_reference.model_dump(mode="json") if source.pipeline_reference is not None else None
        pins.append(
            DeclarationPin(
                source_id=source_id,
                dataset_id=dataset.name if dataset else None,
                dataset_revision=dataset.revision if dataset else None,
                pipeline=pipeline,
                actor_binding=source.info.actor_census_input,
            )
        )
    return pins, sources


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Print the gap report over the served catalog")
    parser.add_argument(
        "--assembled", action="store_true", help="Audit the assembled layer instead of the served catalog"
    )
    parser.add_argument("--plan", type=Path, help="Write the derived-contract census inputs manifest here")
    parser.add_argument("--waivers", type=Path, help="JSON list of source:projection waivers for the preflight")
    parser.add_argument("--preflight", action="store_true", help="Refuse while a current pin lacks a projection")
    args = parser.parse_args()

    if args.plan is not None:
        pins, sources = assembled_declaration_pins()
        gaps, notes = projection_gaps(pins, derived_contract_receipts(), {})
        rows, unrunnable = census_input_rows(pins, sources)
        args.plan.write_text(
            json.dumps({"report": summary(gaps, notes), "census_inputs": rows, "unrunnable": unrunnable}, indent=2)
            + "\n"
        )
        print(
            json.dumps(
                {
                    "plan": str(args.plan),
                    "census_inputs": len(rows),
                    "unrunnable": len(unrunnable),
                    "counts": summary(gaps, notes)["counts"],
                },
                indent=2,
            )
        )
        return
    curriculum_rows = atlas_curriculum_rows()
    if args.assembled:
        pins, _ = assembled_declaration_pins()
        contracts = derived_contract_receipts()
        served = None
    else:
        assembled_pins, _ = assembled_declaration_pins()
        served_list = served_declaration_pins()
        pins, served = assembled_pins, served_pin_map(served_list)
        contracts = atlas_contract_receipts()
    gaps, notes = projection_gaps(pins, contracts, curriculum_rows, served_pins=served)
    report = summary(gaps, notes)
    print(json.dumps(report, indent=2))
    if args.preflight:
        waivers = frozenset(json.loads(args.waivers.read_text())) if args.waivers is not None else frozenset()
        preflight_notes = publication_preflight(pins, contracts, curriculum_rows, waivers, served_pins=served)
        for note in preflight_notes:
            LOGGER.info("staged for intake: %s", note)
        return
    if not args.check:
        parser.error("choose --check, --assembled, --plan or --preflight")


if __name__ == "__main__":
    main()
