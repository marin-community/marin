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
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from finestore.rl.curation import ActorCensusInput
from marina.client import applet_query
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from experiments.post_training.task_curation.datasets.derived_contract_measurements import (
    derived_contract_receipts,
)
from experiments.post_training.task_curation.derived_contract_census import (
    DerivedContractCensusInput,
    DerivedContractReceipt,
)
from experiments.post_training.task_curation.lexical_artifacts import population_for_census
from infra.marina.apps.rl_data_catalog.server.contract_selection import INPUT_FIELDS, select_contract_receipts
from infra.marina.apps.rl_data_catalog.server.schema import PROJECTION_REFRESH_SEAL_SQL
from infra.marina.apps.rl_data_catalog.server.source_identity import canonical_source_id

LOGGER = logging.getLogger(__name__)

DECLARATION_PIN_PAGE_SIZE = 50
APPLET_PAGE_SIZE = 128
"""Receipt rows can be large; a bounded key window keeps every query under the response cap."""
PROJECTION_REFRESH_SEAL_NAME = "ruling-60-2026-11-19"
PROJECTION_WAIVER_CONFIG = Path(__file__).with_name("declaration-projection-waivers.json")
PROJECTION_WAIVER_REASON = "preview projection lag; declaration drift 0; ruling 2026-11-19"

CURRICULUM_DECLARATION_FIELDS = ("dataset_id", "dataset_revision", "pipeline")

ATLAS_APPLET_ID = "fb11c931-5861-4878-8bb5-a964d652b45f"
MARINA_URL = "https://marina.oa.dev"
TARGET_CONFIG = Path(__file__).with_name("atlas-target-applet.json")


def target_applet() -> tuple[str, str]:
    """The served-catalog target applet from config, with production as the default.

    While the production catalog is on the coordinator's hold, the config points the
    served-catalog checks at the preview applet the lanes keep current; lifting the
    hold flips the config back to production.
    """
    reference = json.loads(TARGET_CONFIG.read_text())
    return reference["marina_url"], reference["applet_id"]


def atlas_applet_query():
    marina_url, applet_id = target_applet()
    return applet_query(marina_url, applet_id)


@dataclass(frozen=True)
class DeclarationPin:
    """The current declaration pin a projection must match."""

    source_id: str
    dataset_id: str | None
    dataset_revision: str | None
    pipeline: dict | None
    actor_binding: ActorCensusInput | None

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


@dataclass(frozen=True)
class CatalogSelection:
    marina_url: str
    applet_id: str
    load_id: str
    refresh_sealed: bool


class ProjectionWaiver(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    source_id: str = Field(min_length=1)
    projection: Literal["derived-contract", "curriculum", "served-catalog"]
    state: Literal["missing", "stale-pin"]
    reason: str

    @property
    def key(self) -> str:
        return f"{self.source_id}:{self.projection}"


class ProjectionWaivers(BaseModel):
    """Preview exceptions that expire only after a successful strict refresh seal."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    marina_url: str
    applet_id: str
    expires_when: Literal["preview_refresh_strict_audit_passes"] = "preview_refresh_strict_audit_passes"
    rule: Literal[
        "stale-or-missing-serving-projection-without-declaration-drift",
        "post-seal-declaration-missing-served-projection",
    ]
    reason: Literal[
        "preview projection lag; declaration drift 0; ruling 2026-11-19",
        "source declared after the sealed refresh; served-catalog projection pending; ruling 2026-11-20",
        "admitted; awaiting r39 serving (curriculum embeddings in progress); ruling 2026-11-22",
    ]
    source_ids: tuple[str, ...] = ()
    """Ruling 68: the exact declared source IDs the post-seal waiver covers; empty for the
    ruling-60b preview-lag rule (which waives only non-drifted projection gaps)."""

    def active(self, selection: CatalogSelection) -> bool:
        return (
            self.marina_url == selection.marina_url
            and self.applet_id == selection.applet_id
            and not selection.refresh_sealed
        )


@dataclass(frozen=True)
class ProjectionWaiverResult:
    blocking: list[ProjectionGap]
    waived: list[ProjectionWaiver]
    notes: list[str]


def apply_projection_waivers(
    gaps: list[ProjectionGap],
    waivers: ProjectionWaivers | None,
    selection: CatalogSelection | None,
    drift_sources: frozenset[str] | None,
) -> ProjectionWaiverResult:
    """Evaluate the temporary rule against gaps and declaration drift from this audit run."""
    if waivers is None:
        return ProjectionWaiverResult(gaps, [], [])
    if waivers.rule == "post-seal-declaration-missing-served-projection":
        # Ruling 68: a source declared after the sealed refresh. The named-source waiver is
        # independent of declaration drift, so it runs before the selection/drift check.
        # A stale-pin gap is the same sealed-state class for a re-pinned source: the served
        # projection still records the prior pin and catches up at the next refresh, which
        # the same expiry forces. The curriculum and derived-contract projections of the
        # same named sources are covered by the INTERIM coordinator waiver 2026-11-25
        # 19:40 EST, pending user ratification and expiring with ruling 68's named-source
        # expiry: they are the same sealed-refresh pending state and catch up in the one
        # projection pass the expiry forces. Recorded as PENDING in the atlas rulings
        # (experiments/active/atlas-rulings/interim-projection-waiver-pending-20261125.md).
        blocking, waived, notes = [], [], []
        sealed_pending_projections = {"served-catalog", "curriculum", "derived-contract"}
        if not waivers.source_ids:
            raise ValueError("Ruling 68 waiver must name the exact declared source IDs")
        for gap in gaps:
            if (
                gap.projection in sealed_pending_projections
                and gap.state in {"missing", "stale-pin"}
                and gap.source_id in waivers.source_ids
            ):
                record = ProjectionWaiver(
                    source_id=gap.source_id,
                    projection=gap.projection,
                    state=gap.state,
                    reason=waivers.reason,
                )
                waived.append(record)
                notes.append(
                    f"waived {record.key}/{gap.state}: {record.reason}; "
                    "expires at the next strict refresh seal (which must include the named sources)"
                )
            else:
                blocking.append(gap)
        return ProjectionWaiverResult(blocking, waived, notes)
    if selection is None or drift_sources is None:
        raise ValueError("Projection waiver rules require catalog selection and declaration drift evidence")
    if not waivers.active(selection):
        return ProjectionWaiverResult(
            gaps, [], ["Projection waivers expired or target differs; all projection checks remain blocking"]
        )
    blocking, waived, notes = [], [], []
    for gap in gaps:
        if gap.source_id not in drift_sources and gap.state in {"missing", "stale-pin"}:
            record = ProjectionWaiver(
                source_id=gap.source_id,
                projection=gap.projection,
                state=gap.state,
                reason=waivers.reason,
            )
            waived.append(record)
            notes.append(f"waived {record.key}/{gap.state}: {record.reason}; expires when preview refresh seals")
        else:
            blocking.append(gap)
    return ProjectionWaiverResult(blocking, waived, notes)


def atlas_catalog_selection() -> CatalogSelection:
    marina_url, applet_id = target_applet()
    query = atlas_applet_query()
    rows = query(
        "SELECT load_id, to_regclass(:seal_table) IS NOT NULL AS has_seal_table "
        "FROM atlas_current WHERE name='latest'",
        {"seal_table": "atlas_projection_refresh_seals"},
    )["rows"]
    if len(rows) != 1 or not rows[0]["load_id"]:
        raise ValueError("The served catalog has no selected load")
    # The event table is created by finalization; absence before its first event is unsealed.
    sealed = False
    if rows[0]["has_seal_table"]:
        sealed = query(
            "SELECT EXISTS(SELECT 1 FROM atlas_projection_refresh_seals WHERE name=:name) AS sealed",
            {"name": PROJECTION_REFRESH_SEAL_NAME},
        )["rows"][0]["sealed"]
    return CatalogSelection(marina_url, applet_id, rows[0]["load_id"], refresh_sealed=sealed)


def record_projection_refresh_seal(
    selection: CatalogSelection, scope: ProjectionWaivers, audit: dict[str, Any]
) -> CatalogSelection:
    """Persist the one-way refresh event only for a successful strict preview audit."""
    if (selection.marina_url, selection.applet_id) != (scope.marina_url, scope.applet_id):
        raise ValueError("Refresh seals are restricted to the preview applet")
    if (
        audit["state"] != "ok"
        or audit["findings"]
        or audit["projection_waivers"]
        or audit["waived_projection_count"] != 0
        or audit["waiver_scope"] is not None
        or audit["declaration_drift_sources"]
        or audit["catalog_load_id"] != selection.load_id
    ):
        raise ValueError("Refresh sealing requires a passing strict audit without waivers")
    query = atlas_applet_query()
    query(PROJECTION_REFRESH_SEAL_SQL, {})
    query(
        "INSERT INTO atlas_projection_refresh_seals(name,load_id,audit) "
        "SELECT :name,load_id,CAST(:audit AS JSONB) FROM atlas_current "
        "WHERE name='latest' AND load_id=:load_id ON CONFLICT(name) DO NOTHING",
        {
            "name": PROJECTION_REFRESH_SEAL_NAME,
            "load_id": selection.load_id,
            "audit": json.dumps({**audit, "refresh_sealed": True}),
        },
    )
    observed = atlas_catalog_selection()
    if observed.load_id != selection.load_id or not observed.refresh_sealed:
        raise ValueError("Catalog selection changed during refresh sealing; retry")
    return observed


def contract_receipt_map(receipts: Iterable[DerivedContractReceipt]) -> dict[str, list[DerivedContractReceipt]]:
    grouped: dict[str, list[DerivedContractReceipt]] = defaultdict(list)
    for receipt in receipts:
        grouped[canonical_source_id(receipt.source_id)].append(receipt)
    return dict(grouped)


def contract_state(
    declaration: DeclarationPin, receipts: dict[str, list[DerivedContractReceipt]]
) -> tuple[str, str | None]:
    """Match any archived receipt against the six fields used by the serving projection."""
    if not declaration.measured:
        return "unmeasured", ""
    candidates = receipts.get(canonical_source_id(declaration.source_id), [])
    if not candidates:
        return "missing", ""
    binding = declaration.actor_binding
    matches = [
        receipt
        for receipt in candidates
        if all(getattr(receipt.input, key) == getattr(binding, key) for key in INPUT_FIELDS)
    ]
    if matches:
        if any(receipt.summary != matches[0].summary for receipt in matches):
            raise ValueError(f"Conflicting derived contract summaries: {declaration.source_id}")
        return "current", ""
    return "stale-pin", candidates[-1].input.dataset_revision


def curriculum_state(declaration: DeclarationPin, curriculum_rows: dict[str, dict]) -> tuple[str, str | None]:
    """Whether a curriculum alignment row matches this declaration's dataset pin and pipeline."""
    if not declaration.measured:
        return "unmeasured", ""
    row = curriculum_rows.get(canonical_source_id(declaration.source_id))
    if row is None:
        return "missing", None
    stored = row.get("declaration") or {}
    triple_equal = all(stored.get(key) == getattr(declaration, key) for key in CURRICULUM_DECLARATION_FIELDS)
    if triple_equal and row["has_alignment"]:
        return "current", None
    return "stale-pin", stored.get("dataset_revision")


def projection_gaps(
    declarations: list[DeclarationPin],
    contract_receipts: dict[str, list[DerivedContractReceipt]],
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
        state, last_computed = contract_state(declaration, contract_receipts)
        if served_pins is not None and declaration.measured:
            served = served_pins.get(canonical_source_id(declaration.source_id))
            if (
                served is None
                or any(getattr(served, key) != getattr(declaration, key) for key in CURRICULUM_DECLARATION_FIELDS)
                or served.actor_binding is None
                or any(
                    getattr(served.actor_binding, key) != getattr(declaration.actor_binding, key) for key in INPUT_FIELDS
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


def census_input_rows(declarations: list[DeclarationPin], sources: dict[str, Any]) -> tuple[list[dict], list[str]]:
    """Build the derived-contract census inputs over each measured source's bound population.

    The actor census binding's evidence artifact records the exact population the census
    measured (``input_artifact``), so the derived-contract census reuses that frozen
    population with the same binding — idempotent inputs for the in-region CPU job. Every
    row is validated against the census's own input model before it leaves this stage, so a
    population that disagrees with its binding can never reach a job; the mismatched sources
    are returned for the report instead.
    """
    receipts = contract_receipt_map(derived_contract_receipts().values())
    rows = []
    unrunnable = []
    for declaration in declarations:
        state, _ = contract_state(declaration, receipts)
        if state not in {"missing", "stale-pin"}:
            continue
        source = sources[declaration.source_id]
        binding = source.info.actor_census_input
        try:
            population = population_for_census(binding.evidence_url, declaration.source_id)
        except ValueError as error:
            # A staged source whose census evidence is not an artifact record yet (an
            # intake lane mid-flight) cannot feed a census job: record it for the report
            # instead of failing the whole plan over another lane's pending intake.
            unrunnable.append(
                f"derived-contract census: {declaration.source_id} census evidence does not resolve "
                f"({type(error).__name__}: {error}); stage its population and census it before the "
                "projections can run"
            )
            continue
        row = {"population": population.model_dump(mode="json"), "input": binding.model_dump(mode="json")}
        try:
            DerivedContractCensusInput.model_validate(row)
        except ValidationError as error:
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
    contract_receipts: dict[str, list[DerivedContractReceipt]],
    curriculum_rows: dict[str, dict],
    waivers: ProjectionWaivers | None = None,
    served_pins: dict[str, DeclarationPin] | None = None,
    *,
    selection: CatalogSelection | None = None,
    drift_sources: frozenset[str] | None = None,
) -> list[str]:
    """Refuse a preview or production release while a current pin lacks either projection.

    The rule requires current declaration drift evidence and expires outside its frozen
    preview selections. Returns notes recording each source:projection pair actually waived.
    """
    gaps, notes = projection_gaps(declarations, contract_receipts, curriculum_rows, served_pins=served_pins)
    result = apply_projection_waivers(gaps, waivers, selection, drift_sources)
    if waivers is not None and drift_sources:
        raise ValueError("Declaration drift blocks publication: " + ", ".join(sorted(drift_sources)))
    blocking = result.blocking
    notes.extend(result.notes)
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


def atlas_curriculum_rows() -> dict[str, dict]:
    """Read the consumed pin fields in bounded pages from one selected curriculum load."""
    query = atlas_applet_query()
    selected = query("SELECT load_id FROM curriculum_current WHERE name='latest'", {})["rows"]
    if not selected:
        return {}
    load_id = selected[0]["load_id"]
    rows = {}
    cursor = ""
    while True:
        page = query(
            "SELECT source_id, jsonb_build_object("
            "'dataset_id', declaration->'dataset_id', "
            "'dataset_revision', declaration->'dataset_revision', "
            "'pipeline', declaration->'pipeline') AS declaration, "
            "alignment IS NOT NULL AS has_alignment FROM curriculum_sources "
            "WHERE load_id=:load_id AND source_id>:cursor ORDER BY source_id LIMIT :limit",
            {"load_id": load_id, "cursor": cursor, "limit": DECLARATION_PIN_PAGE_SIZE},
        )["rows"]
        rows.update((row["source_id"], row) for row in page)
        if len(page) < DECLARATION_PIN_PAGE_SIZE:
            return rows
        cursor = page[-1]["source_id"]


def atlas_contract_receipts() -> dict[str, list[DerivedContractReceipt]]:
    """Read the archive in bounded key windows; the receipts outgrew one query's response cap."""
    query = atlas_applet_query()
    rows: list[dict] = []
    after = ""
    while True:
        page = query(
            "SELECT receipt_id, source_id, receipt FROM atlas_derived_contracts "
            "WHERE receipt_id > :after ORDER BY receipt_id LIMIT :batch",
            {"after": after, "batch": APPLET_PAGE_SIZE},
        )["rows"]
        rows.extend(page)
        if len(page) < APPLET_PAGE_SIZE:
            break
        after = page[-1]["receipt_id"]
    # Select exact archived JSON identities before typed parsing adds legacy defaults.
    populations = defaultdict(list)
    for row in rows:
        receipt = row["receipt"] if isinstance(row["receipt"], dict) else json.loads(row["receipt"])
        source = canonical_source_id(row["source_id"])
        identity = (source, *(receipt["input"][key] for key in INPUT_FIELDS))
        populations[identity].append(receipt)
    return contract_receipt_map(
        DerivedContractReceipt.model_validate(receipt)
        for identity, receipts in populations.items()
        for receipt in select_contract_receipts(identity[0], receipts)
    )


def served_declaration_pins(selection: CatalogSelection) -> list[DeclarationPin]:
    """Page the four consumed fields from one frozen active catalog load."""
    query = atlas_applet_query()
    pins = []
    offset = 0
    while True:
        rows = query(
            "SELECT id, jsonb_build_object("
            "'dataset_id', declaration->'dataset_id', "
            "'dataset_revision', declaration->'dataset_revision', "
            "'pipeline', declaration->'pipeline', "
            "'actor_census_input', declaration->'actor_census_input') AS declaration "
            "FROM atlas_sources WHERE load_id=:load_id ORDER BY id LIMIT :limit OFFSET :offset",
            {"load_id": selection.load_id, "limit": DECLARATION_PIN_PAGE_SIZE, "offset": offset},
        )["rows"]
        for row in rows:
            served = row.get("declaration") or {}
            pins.append(
                DeclarationPin(
                    source_id=row["id"],
                    dataset_id=served.get("dataset_id"),
                    dataset_revision=served.get("dataset_revision"),
                    pipeline=served.get("pipeline"),
                    actor_binding=_binding_from_served(served.get("actor_census_input")),
                )
            )
        if len(rows) < DECLARATION_PIN_PAGE_SIZE:
            return pins
        offset += len(rows)


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
    parser.add_argument(
        "--waivers", type=Path, help="JSON source:projection waivers with ruling reason and preview refresh expiry"
    )
    parser.add_argument("--preflight", action="store_true", help="Refuse while a current pin lacks a projection")
    args = parser.parse_args()
    if args.waivers is not None and not args.preflight:
        parser.error("--waivers requires --preflight")

    if args.plan is not None:
        pins, sources = assembled_declaration_pins()
        gaps, notes = projection_gaps(pins, contract_receipt_map(derived_contract_receipts().values()), {})
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
    if args.assembled and args.waivers is not None:
        parser.error("Projection waivers apply only to the served preview")
    selection = atlas_catalog_selection() if not args.assembled else None
    curriculum_rows = atlas_curriculum_rows()
    if args.assembled:
        pins, _ = assembled_declaration_pins()
        contracts = contract_receipt_map(derived_contract_receipts().values())
        served = None
    else:
        assembled_pins, _ = assembled_declaration_pins()
        served_list = served_declaration_pins(selection)
        pins, served = assembled_pins, served_pin_map(served_list)
        contracts = atlas_contract_receipts()
    if selection is not None and atlas_catalog_selection() != selection:
        raise ValueError("Catalog selection changed during projection audit; retry")
    gaps, notes = projection_gaps(pins, contracts, curriculum_rows, served_pins=served)
    report = summary(gaps, notes)
    print(json.dumps(report, indent=2))
    if args.preflight:
        waivers = ProjectionWaivers.model_validate_json(args.waivers.read_bytes()) if args.waivers else None
        drift_sources = None
        if waivers is not None:
            # The declaration audit imports this module; import here to break that cycle.
            from experiments.post_training.task_curation.declaration_audit import (  # noqa: PLC0415
                audit_declarations,
                canonical_layer,
            )
            from experiments.post_training.task_curation.sources import configured_sources  # noqa: PLC0415

            layer, families, releases, _ = canonical_layer()
            declaration_audit = audit_declarations(
                families,
                layer=layer,
                releases=releases,
                configured_ids=frozenset(source.info.id for source in configured_sources()),
            )
            drift_sources = declaration_audit.drift_sources
        preflight_notes = publication_preflight(
            pins,
            contracts,
            curriculum_rows,
            waivers,
            served_pins=served,
            selection=selection,
            drift_sources=drift_sources,
        )
        for note in preflight_notes:
            LOGGER.info("%s", note)
        return
    if not args.check:
        parser.error("choose --check, --assembled, --plan or --preflight")


if __name__ == "__main__":
    main()
