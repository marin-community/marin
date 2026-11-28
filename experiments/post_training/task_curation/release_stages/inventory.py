# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Inventory out-of-repo Atlas preview-release scripts and map them to in-repo stages.

The R40 gate (BRIEF.md, 2026-11-23) forbids r40 from using any out-of-tree script. This
module is the auditable record of that retirement: it walks the historical preview-rebuild
directory, classifies every file, and maps every release-stage script to the in-repo
``release_stages`` module that replaces it (or marks it RETIRED as evidence/analysis).

It reads the rebuild directory and writes one INVENTORY.md next to it. It never modifies
or deletes anything in the rebuild directory; those files are evidence.

Usage::

    uv run python -m experiments.post_training.task_curation.release_stages.inventory \
        --rebuild-root ~/Documents/experiments/active/atlas-preview-rebuild
"""

import argparse
import ast
import dataclasses
import re
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

# Virtualenv trees: environments, not scripts and not bundled repo copies.
_ENV_SEGMENT = re.compile(r"^(.*-venv.*|venv.*|site-packages|__pycache__)$")

# Working-package trees whose Python files are copies of in-repo modules or tests.
_BUNDLED_SEGMENT = re.compile(r"^(server|journeys|dist|bundled|screenshots|bin|lib|include|tests|fixtures)$")
_BUNDLED_TOP_DIR = re.compile(r"^app-")

# Stems that exist as modules or test helpers in the in-repo rl_data_catalog app; a file
# with one of these stems anywhere in the rebuild tree is a bundled copy of that module.
_APP_STEMS = frozenset(
    {
        "confidence",
        "historical_assessments",
        "load",
        "load_browse_evidence",
        "load_counts",
        "load_curriculum",
        "load_derived_contract",
        "load_difficulty",
        "load_identity",
        "load_reviews",
        "load_rollouts",
        "parallel_history",
        "prepare_preview",
        "readability",
        "release_browse",
        "release_browser",
        "release_check",
        "release_curriculum",
        "release_evidence",
        "release_gateway",
        "release_receipts",
        "review_mapping",
        "role_projection",
        "stage_browse_evidence",
    }
)


def _is_env(relative: Path) -> bool:
    return any(_ENV_SEGMENT.match(part) for part in relative.parts)


def _is_bundled(relative: Path) -> bool:
    parts = relative.parts
    if not parts:
        return False
    if _BUNDLED_TOP_DIR.match(parts[0]):
        return True
    return any(_BUNDLED_SEGMENT.match(part) for part in parts[1:])


@dataclasses.dataclass(frozen=True)
class StageSpec:
    """One maintained release-pipeline stage module under ``release_stages/``."""

    module: str
    title: str
    description: str


STAGES: dict[str, StageSpec] = {
    spec.module: spec
    for spec in [
        StageSpec(
            "declarations.py",
            "Declarations",
            "Extract, compare, freeze and apply source declarations and supersessions.",
        ),
        StageSpec(
            "catalog_stage.py", "Catalog staging", "Stage an isolated catalog from declarations and exact-pin reviews."
        ),
        StageSpec(
            "census.py", "Census", "Population census: SampleInventory receipts, oracle/static counts, count binding."
        ),
        StageSpec(
            "corpus_assembly.py", "Corpus assembly", "Regional prepare_sources corpus build with inventory verification."
        ),
        StageSpec(
            "public_input_reuse.py",
            "Public input reuse",
            "Verify complete task correspondence and seal curriculum reuse evidence from a fresh census.",
        ),
        StageSpec(
            "embed.py", "Embedding", "Curriculum embedding; GPU only with a valid runtime parity receipt, else CPU."
        ),
        StageSpec(
            "curriculum_compose.py",
            "Curriculum compose",
            "Compose curriculum populations while preserving other assignments.",
        ),
        StageSpec(
            "identity_backend_compose.py",
            "Identity/backend compose",
            "Identity surveys, backend validation censuses, counts, oracle ledger.",
        ),
        StageSpec(
            "history_staging.py",
            "History staging",
            "Parallel retained-history resume with a shared bounded retry budget.",
        ),
        StageSpec("text_scans.py", "Text scans", "Full-release text scan submission and merge."),
        StageSpec("qc_browse.py", "QC browse", "QC browse plan build and browse worker; sample QC proofs."),
        StageSpec(
            "snapshot_publication.py",
            "Snapshot publication",
            "Publish snapshots, seal contracts, archive superseded versions.",
        ),
        StageSpec("catalog_load.py", "Catalog load", "Catalog/difficulty load into local apps or the hosted preview."),
        StageSpec(
            "curriculum_load.py",
            "Curriculum load",
            "Curriculum import into the preview with history preservation proof.",
        ),
        StageSpec(
            "strict_audit.py", "Strict audit", "Strict final-catalog/registry audit and history-preservation proof."
        ),
        StageSpec(
            "package_preview.py", "Package preview", "Bundle preview dependencies and preflight entrypoint imports."
        ),
        StageSpec(
            "release_check.py",
            "Release check",
            "Release-check gates (preflight/hosted/postpublish) over the in-repo module.",
        ),
        StageSpec("marina_release.py", "Marina release", "Marina app release and release-record sealing."),
        StageSpec("hosted_check.py", "Hosted check", "Hosted/local browser and API verification, target-parameterized."),
        StageSpec(
            "exposure_probe.py", "Exposure probe", "Anonymous exposure probe and private-content guard verification."
        ),
        StageSpec("graph.py", "Release graph", "Lazy ArtifactStep graph, typed manifest, Iris dispatch via remote()."),
        StageSpec("manifest.py", "Typed manifest", "ReleaseManifest and per-stage receipt Artifact types."),
        StageSpec("preflight.py", "Preflight", "Per-stage type-check + small-slice dry run; preflight receipts."),
        StageSpec("sizing.py", "Sizing", "Workers/memory/timeouts derived from population size; device selection."),
        StageSpec("alerts.py", "Alerts", "Stage start/finish/fail/stall events and watchdog registration."),
        StageSpec("guard.py", "Guard", "Submission guard: landed-SHA module check + release-literal scan."),
        StageSpec("inventory.py", "Inventory", "This inventory generator."),
    ]
}

_GRAPH = "graph.py"
_PREFLIGHT = "preflight.py"

_STATUS_REPLACED = "REPLACED"
_STATUS_RETIRED_ANALYSIS = "RETIRED-ANALYSIS"
_STATUS_BUNDLED_COPY = "BUNDLED-COPY"
_STATUS_EVIDENCE = "RETIRED-EVIDENCE"
_STATUS_ENV = "ENVIRONMENT"
_STATUS_UNCLASSIFIED = "NEEDS-REVIEW"

# Ordered family rules: first fullmatch on the file stem wins. ``stage`` is the replacing
# release_stages module; ``status`` distinguishes mapping from retirement.
_RULES: tuple[tuple[str, str, str | None, str, str], ...] = (
    # -- Launchers: per-release copies of one dispatch family; the graph dispatches. --------
    (r"launch[-_].*census.*", "launcher", "census.py", "census submission via graph dispatch", _STATUS_REPLACED),
    (
        r"launch[-_].*corpus.*",
        "launcher",
        "corpus_assembly.py",
        "corpus submission via graph dispatch",
        _STATUS_REPLACED,
    ),
    (
        r"launch[-_].*curriculum[-_]embed.*",
        "launcher",
        "embed.py",
        "embed submission via graph dispatch",
        _STATUS_REPLACED,
    ),
    (
        r"launch[-_].*curriculum[-_]compose.*",
        "launcher",
        "curriculum_compose.py",
        "compose submission via graph dispatch",
        _STATUS_REPLACED,
    ),
    (
        r"launch[-_].*curriculum[-_]load.*",
        "launcher",
        "curriculum_load.py",
        "load submission via graph dispatch",
        _STATUS_REPLACED,
    ),
    (
        r"launch[-_].*catalog[-_]load.*",
        "launcher",
        "catalog_load.py",
        "load submission via graph dispatch",
        _STATUS_REPLACED,
    ),
    (
        r"launch[-_].*history[-_]stage.*",
        "launcher",
        "history_staging.py",
        "staging submission via graph dispatch",
        _STATUS_REPLACED,
    ),
    (
        r"launch[-_].*backend.*",
        "launcher",
        "identity_backend_compose.py",
        "backend submission via graph dispatch",
        _STATUS_REPLACED,
    ),
    (
        r"launch[-_].*qc[-_](browse|stage).*",
        "launcher",
        "qc_browse.py",
        "QC submission via graph dispatch",
        _STATUS_REPLACED,
    ),
    (r"launch[-_].*textscan.*", "launcher", "text_scans.py", "scan submission via graph dispatch", _STATUS_REPLACED),
    (
        r"launch[-_].*compose[-_]preflight.*",
        "launcher",
        _PREFLIGHT,
        "compose preflight via preflight stage",
        _STATUS_REPLACED,
    ),
    (r"launch[-_].*inventory.*", "launcher", "census.py", "inventory submission via graph dispatch", _STATUS_REPLACED),
    (r"launch[-_].*", "launcher", _GRAPH, "typed graph dispatch (release_stages/graph.py)", _STATUS_REPLACED),
    (
        r"driver[-_].*curriculum[-_]load.*",
        "launcher",
        "curriculum_load.py",
        "load submission via graph dispatch",
        _STATUS_REPLACED,
    ),
    (r"driver[-_].*", "launcher", _GRAPH, "typed graph dispatch", _STATUS_REPLACED),
    # -- Shell job wrappers for one stage each. -----------------------------------------------
    (
        r".*curriculum[-_]load.*job.*",
        "launcher",
        "curriculum_load.py",
        "load submission via graph dispatch",
        _STATUS_REPLACED,
    ),
    (
        r".*curriculum[-_](project|preparation).*job.*",
        "launcher",
        "curriculum_compose.py",
        "compose submission via graph dispatch",
        _STATUS_REPLACED,
    ),
    (
        r".*catalog[-_]refresh.*job.*",
        "launcher",
        "snapshot_publication.py",
        "snapshot refresh via graph dispatch",
        _STATUS_REPLACED,
    ),
    (r".*job.*", "launcher", _GRAPH, "typed graph dispatch", _STATUS_REPLACED),
    # -- Workers: one maintained module per stage replaces all per-release copies. ------------
    (r".*census.*worker.*", "worker", "census.py", "release_stages/census.py worker", _STATUS_REPLACED),
    (
        r".*corpus.*worker.*",
        "worker",
        "corpus_assembly.py",
        "release_stages/corpus_assembly.py worker",
        _STATUS_REPLACED,
    ),
    (
        r".*history[-_]stage.*worker.*",
        "worker",
        "history_staging.py",
        "release_stages/history_staging.py worker",
        _STATUS_REPLACED,
    ),
    (
        r".*catalog[-_]load.*worker.*",
        "worker",
        "catalog_load.py",
        "release_stages/catalog_load.py worker",
        _STATUS_REPLACED,
    ),
    (
        r".*curriculum[-_]load.*worker.*",
        "worker",
        "curriculum_load.py",
        "release_stages/curriculum_load.py worker",
        _STATUS_REPLACED,
    ),
    (r".*qc[-_]browse.*worker.*", "worker", "qc_browse.py", "release_stages/qc_browse.py worker", _STATUS_REPLACED),
    (r".*backend.*worker.*", "worker", "identity_backend_compose.py", "backend validation worker", _STATUS_REPLACED),
    (r".*inventory[-_]census.*", "worker", "census.py", "release_stages/census.py worker", _STATUS_REPLACED),
    (r".*worker.*", "worker", _GRAPH, "stage worker (one maintained module per stage)", _STATUS_REPLACED),
    # -- Compose family. ------------------------------------------------------------------------
    (
        r".*compose.*curriculum.*",
        "compose",
        "curriculum_compose.py",
        "release_stages/curriculum_compose.py",
        _STATUS_REPLACED,
    ),
    (
        r"atlas[-_]consumer[-_]compose[-_]curriculum",
        "compose",
        "curriculum_compose.py",
        "release_stages/curriculum_compose.py",
        _STATUS_REPLACED,
    ),
    (
        r"atlas[-_]consumer[-_]curriculum[-_]composition[-_]support",
        "compose",
        "curriculum_compose.py",
        "composition support module",
        _STATUS_REPLACED,
    ),
    (r".*curriculum[-_]project.*", "compose", "curriculum_compose.py", "curriculum projection", _STATUS_REPLACED),
    (r".*compose.*backend.*", "compose", "identity_backend_compose.py", "backend compose", _STATUS_REPLACED),
    (
        r"compose[-_].*",
        "compose",
        "identity_backend_compose.py",
        "release_stages/identity_backend_compose.py",
        _STATUS_REPLACED,
    ),
    # -- Plans, prep, proofs, scans. --------------------------------------------------------------
    (
        r"build[-_].*qc[-_]browse[-_]plan.*",
        "plan",
        "qc_browse.py",
        "plan built inside release_stages/qc_browse.py",
        _STATUS_REPLACED,
    ),
    (r"build[-_].*plan.*", "plan", _GRAPH, "plan as typed graph output", _STATUS_REPLACED),
    (r"prepare[-_].*job[-_]inputs.*", "prep", _GRAPH, "typed manifest + remote() dispatch", _STATUS_REPLACED),
    (r"prepare[-_].*", "prep", _GRAPH, "typed manifest + remote() dispatch", _STATUS_REPLACED),
    (r".*sample.*(proof|fixture).*", "preflight", "qc_browse.py", "sample QC proof/fixture", _STATUS_REPLACED),
    (r".*review.*(proof|readback).*", "preflight", "qc_browse.py", "review restoration proof", _STATUS_REPLACED),
    (
        r"local[-_].*proof.*",
        "preflight",
        _PREFLIGHT,
        "small-slice preflight (release_stages/preflight.py)",
        _STATUS_REPLACED,
    ),
    (r".*proof.*", "preflight", _PREFLIGHT, "small-slice proof (release_stages/preflight.py)", _STATUS_REPLACED),
    (r"prove[-_].*", "preflight", _PREFLIGHT, "small-slice proof (release_stages/preflight.py)", _STATUS_REPLACED),
    (r".*preflight.*", "preflight", _PREFLIGHT, "release_stages/preflight.py", _STATUS_REPLACED),
    (r".*text[-_]scan.*", "scan", "text_scans.py", "release_stages/text_scans.py", _STATUS_REPLACED),
    (r"merge[-_].*text[-_]scan.*", "scan", "text_scans.py", "release_stages/text_scans.py merge", _STATUS_REPLACED),
    (r".*textscan.*", "scan", "text_scans.py", "release_stages/text_scans.py", _STATUS_REPLACED),
    (r"^scan([-_]v\d+)?$", "scan", "text_scans.py", "release_stages/text_scans.py", _STATUS_REPLACED),
    # -- Snapshot publication and record sealing. ---------------------------------------------------
    (
        r".*publish.*snapshot.*",
        "publish",
        "snapshot_publication.py",
        "release_stages/snapshot_publication.py",
        _STATUS_REPLACED,
    ),
    (r"publish[-_].*", "publish", "snapshot_publication.py", "release_stages/snapshot_publication.py", _STATUS_REPLACED),
    (r".*publish.*", "publish", "snapshot_publication.py", "release_stages/snapshot_publication.py", _STATUS_REPLACED),
    (
        r"seal[-_].*contracts.*",
        "publish",
        "snapshot_publication.py",
        "release_stages/snapshot_publication.py",
        _STATUS_REPLACED,
    ),
    (
        r"seal[-_].*",
        "publish",
        "marina_release.py",
        "record sealing in release_stages/marina_release.py",
        _STATUS_REPLACED,
    ),
    (
        r"finalize[-_].*record.*",
        "publish",
        "marina_release.py",
        "record sealing in release_stages/marina_release.py",
        _STATUS_REPLACED,
    ),
    (r"postpublish[-_].*", "publish", "release_check.py", "postpublish gate phase", _STATUS_REPLACED),
    (r".*promote.*", "publish", "snapshot_publication.py", "candidate promotion", _STATUS_REPLACED),
    (r"archive[-_].*", "publish", "snapshot_publication.py", "version archival", _STATUS_REPLACED),
    (r".*guarded.*", "publish", "snapshot_publication.py", "guarded-candidate publication lifecycle", _STATUS_REPLACED),
    (r".*private3[45].*", "publish", "snapshot_publication.py", "private-release lifecycle", _STATUS_REPLACED),
    (r".*public3[45].*", "publish", "exposure_probe.py", "public exposure/promotion probe", _STATUS_REPLACED),
    (r"finalize[-_].*", "publish", "snapshot_publication.py", "build finalization/archival", _STATUS_REPLACED),
    # -- Declarations and catalog staging. ------------------------------------------------------------
    (
        r"extract[-_]declarations.*",
        "declarations",
        "declarations.py",
        "release_stages/declarations.py",
        _STATUS_REPLACED,
    ),
    (r".*declarations.*", "declarations", "declarations.py", "release_stages/declarations.py", _STATUS_REPLACED),
    (
        r"apply[-_].*supersessions.*",
        "declarations",
        "declarations.py",
        "receipt application in declarations stage",
        _STATUS_REPLACED,
    ),
    (
        r"compare[-_].*pins.*",
        "declarations",
        "declarations.py",
        "pin comparison in declarations stage",
        _STATUS_REPLACED,
    ),
    (
        r"build[-_](catalog|maps|role[-_]preparation[-_]maps).*",
        "staging",
        "catalog_stage.py",
        "release_stages/catalog_stage.py",
        _STATUS_REPLACED,
    ),
    (r"stage[-_]catalog.*", "staging", "catalog_stage.py", "release_stages/catalog_stage.py", _STATUS_REPLACED),
    (
        r"stage[-_]source[-_]areas.*",
        "staging",
        "curriculum_compose.py",
        "source-area staging for composition",
        _STATUS_REPLACED,
    ),
    # -- Identity/backend compose. -----------------------------------------------------------------------
    (
        r"build[-_].*identit(y|ies).*",
        "compose",
        "identity_backend_compose.py",
        "identity survey binding",
        _STATUS_REPLACED,
    ),
    (r"build[-_].*backend.*", "compose", "identity_backend_compose.py", "backend validation binding", _STATUS_REPLACED),
    (r"combine[-_].*backend.*", "compose", "identity_backend_compose.py", "backend census combine", _STATUS_REPLACED),
    (r"stage[-_]identit.*", "compose", "identity_backend_compose.py", "identity binding worker", _STATUS_REPLACED),
    (r"stage[-_]backend.*", "compose", "identity_backend_compose.py", "backend validation worker", _STATUS_REPLACED),
    (
        r"regional[-_].*backend.*census.*",
        "compose",
        "identity_backend_compose.py",
        "regional backend census",
        _STATUS_REPLACED,
    ),
    (r".*surveys?.*", "compose", "identity_backend_compose.py", "survey collection/binding", _STATUS_REPLACED),
    # -- Census and counts. ---------------------------------------------------------------------------------
    (r"build[-_].*counts?", "census", "census.py", "count binding in census stage", _STATUS_REPLACED),
    (r"build[-_].*count[-_]?inputs.*", "census", "census.py", "count binding in census stage", _STATUS_REPLACED),
    (r"build[-_].*inventory.*", "census", "census.py", "inventory binding in census stage", _STATUS_REPLACED),
    (r"fetch[-_].*inventory.*", "census", "census.py", "inventory fetch", _STATUS_REPLACED),
    (r".*restore-(counts|oracle).*", "census", "census.py", "receipt restore into census stage", _STATUS_REPLACED),
    (r"restore[-_].*", "census", "census.py", "receipt restore into census stage", _STATUS_REPLACED),
    (r"stage[-_]oracle[-_]inventory.*", "census", "census.py", "oracle census binding", _STATUS_REPLACED),
    (r"stage[-_]progress.*", "census", "census.py", "content-progress binding", _STATUS_REPLACED),
    (r"project[-_].*actorless.*", "census", "census.py", "actorless exemption projection", _STATUS_REPLACED),
    (r"collect[-_].*", "census", "census.py", "regional collection into census stage", _STATUS_REPLACED),
    (r"export[-_].*", "census", "census.py", "input export via census stage", _STATUS_REPLACED),
    (r"select[-_].*inputs.*", "census", "census.py", "input selection", _STATUS_REPLACED),
    (r".*census.*", "census", "census.py", "release_stages/census.py", _STATUS_REPLACED),
    # -- Loads. -----------------------------------------------------------------------------------------------
    (r"load[-_].*curriculum.*", "load", "curriculum_load.py", "release_stages/curriculum_load.py", _STATUS_REPLACED),
    (
        r"copy[-_]existing[-_]curriculum.*",
        "load",
        "curriculum_load.py",
        "migration fixtures via preflight small-slice inputs",
        _STATUS_REPLACED,
    ),
    (r".*submit[-_].*curriculum.*", "load", "curriculum_load.py", "curriculum import submission", _STATUS_REPLACED),
    (r"load[-_].*difficulty.*", "load", "catalog_load.py", "canonical loader in catalog load stage", _STATUS_REPLACED),
    (r"load[-_].*catalog.*", "load", "catalog_load.py", "release_stages/catalog_load.py", _STATUS_REPLACED),
    (r"load[-_].*", "load", "catalog_load.py", "release_stages/catalog_load.py", _STATUS_REPLACED),
    # -- Audits, checks, gates, verification. --------------------------------------------------------------------
    (r"audit[-_].*", "audit", "strict_audit.py", "release_stages/strict_audit.py", _STATUS_REPLACED),
    (
        r"check[-_].*(catalog|registry|history|inputs).*",
        "check",
        "strict_audit.py",
        "catalog/registry/history proof in strict audit",
        _STATUS_REPLACED,
    ),
    (r"check[-_].*", "check", "hosted_check.py", "target-parameterized browser/API check", _STATUS_REPLACED),
    (r"recheck[-_].*", "check", "release_check.py", "release-check re-verification", _STATUS_REPLACED),
    (r"gates[-_].*", "gates", "release_check.py", "release-check gates", _STATUS_REPLACED),
    (r".*verify.*actorless.*", "check", "strict_audit.py", "actorless contract proof", _STATUS_REPLACED),
    (r".*verify.*", "check", "strict_audit.py", "read-only verification proof", _STATUS_REPLACED),
    (r".*readback.*", "check", "hosted_check.py", "served-state readback verification", _STATUS_REPLACED),
    (r".*continuity.*", "check", "strict_audit.py", "continuity proof", _STATUS_REPLACED),
    (r".*hold[-_]check.*", "check", "strict_audit.py", "final hold check", _STATUS_REPLACED),
    (r"inspect[-_].*", "check", "strict_audit.py", "read-only inspection", _STATUS_REPLACED),
    (r"capture[-_].*", "check", "hosted_check.py", "hosted/browser evidence capture", _STATUS_REPLACED),
    (r"hosted[-_].*", "check", "hosted_check.py", "hosted verification", _STATUS_REPLACED),
    (r".*probe.*", "check", "hosted_check.py", "probe check (target-parameterized)", _STATUS_REPLACED),
    (r"candidate[-_]api[-_]bridge.*", "check", "hosted_check.py", "served-API bridge check", _STATUS_REPLACED),
    (r"resume[-_].*review.*", "check", "qc_browse.py", "review resume", _STATUS_REPLACED),
    (r"review[-_]presentation.*", "check", "qc_browse.py", "review presentation binding", _STATUS_REPLACED),
    (r"update[-_]history.*", "check", "history_staging.py", "history update via history staging", _STATUS_REPLACED),
    (r"scan[-_]source[-_]api.*", "check", "hosted_check.py", "source API scan", _STATUS_REPLACED),
    # -- Packaging. -------------------------------------------------------------------------------------------------
    (r"bundle[-_].*preview.*", "package", "package_preview.py", "release_stages/package_preview.py", _STATUS_REPLACED),
    (r".*job[-_]preflight.*", "package", "package_preview.py", "entrypoint import preflight", _STATUS_REPLACED),
    # -- Exposure. ---------------------------------------------------------------------------------------------------
    (r".*exposure.*", "exposure", "exposure_probe.py", "release_stages/exposure_probe.py", _STATUS_REPLACED),
    # -- Performance, diagnostic and profiling one-offs (evidence only). ----------------------------------------------
    (
        r"profile[-_].*",
        "one-off",
        None,
        "performance profiling one-off; findings live in receipts/docs",
        _STATUS_RETIRED_ANALYSIS,
    ),
    (
        r"reproduce[-_].*",
        "one-off",
        None,
        "failure reproduction one-off; superseded by typed preflight",
        _STATUS_RETIRED_ANALYSIS,
    ),
    (r"diagnose[-_].*", "one-off", None, "diagnostic one-off; superseded by typed receipts", _STATUS_RETIRED_ANALYSIS),
    (r"debug[-_].*", "one-off", None, "debug one-off", _STATUS_RETIRED_ANALYSIS),
    (
        r"measure[-_].*",
        "one-off",
        None,
        "measurement one-off; superseded by sizing measurements",
        _STATUS_RETIRED_ANALYSIS,
    ),
    (r"benchmark[-_].*", "one-off", None, "benchmark one-off", _STATUS_RETIRED_ANALYSIS),
    (r"perf[-_].*", "one-off", None, "performance one-off", _STATUS_RETIRED_ANALYSIS),
    (
        r"watch[-_].*",
        "one-off",
        None,
        "manual watch one-off; superseded by watchdog registration",
        _STATUS_RETIRED_ANALYSIS,
    ),
    (r"pause[-_].*", "one-off", None, "manual cluster control one-off", _STATUS_RETIRED_ANALYSIS),
    (r"serve[-_]replay", "one-off", None, "replay server one-off", _STATUS_RETIRED_ANALYSIS),
    (
        r"retry[-_]cold[-_]table.*",
        "one-off",
        None,
        "cold-table retry one-off; superseded by sizing/retry policy",
        _STATUS_RETIRED_ANALYSIS,
    ),
    (r"cold[-_]table.*", "one-off", None, "cold-table reproduction one-off", _STATUS_RETIRED_ANALYSIS),
    # -- One-off content edits, reports and runners. -------------------------------------------------------------------
    (
        r"refresh[-_]guide",
        "one-off",
        None,
        "guide content edit; content lives in the in-repo guide",
        _STATUS_RETIRED_ANALYSIS,
    ),
    (
        r"report[-_].*",
        "one-off",
        None,
        "comparison report; superseded by strict-audit receipts",
        _STATUS_RETIRED_ANALYSIS,
    ),
    (
        r"v93[-_]typed[-_]backend[-_]fixtures.*",
        "one-off",
        None,
        "typed fixtures; subsumed by typed manifest artifacts",
        _STATUS_RETIRED_ANALYSIS,
    ),
    (r"run[-_].*tests", "one-off", None, "test runner; tests live in-repo", _STATUS_RETIRED_ANALYSIS),
    (r"run[-_]proof[-_]with[-_]cleanup", "preflight", _PREFLIGHT, "proof with cleanup", _STATUS_REPLACED),
    (r"land[-_]tests", "one-off", None, "one-off test landing helper; superseded by repo CI", _STATUS_RETIRED_ANALYSIS),
    (r"drop[-_]duplicate[-_]projection", "one-off", None, "one-off projection fix", _STATUS_RETIRED_ANALYSIS),
    (
        r"normalize[-_]catalog[-_]judge[-_]reasons.*",
        "one-off",
        None,
        "one-off data normalization",
        _STATUS_RETIRED_ANALYSIS,
    ),
    (
        r".*app[-_]archived",
        "one-off",
        None,
        "archived app server entrypoint; serving lives in the in-repo app",
        _STATUS_RETIRED_ANALYSIS,
    ),
    (
        r"unlanded[-_]source[-_]inventory[-_]prototype[-_]v\d+",
        "census",
        "census.py",
        "inventory prototype",
        _STATUS_REPLACED,
    ),
    (r".*exact[-_]inventory[-_]split.*", "census", "census.py", "inventory-only split", _STATUS_REPLACED),
    (r"preview[-_]load[-_]job.*", "load", "catalog_load.py", "load job via graph dispatch", _STATUS_REPLACED),
    # -- Remaining release-iteration helpers. ---------------------------------------------
    (r".*readonly[-_]check.*", "check", "strict_audit.py", "read-only binding check", _STATUS_REPLACED),
    (r"coverage[-_]control", "check", "qc_browse.py", "sample QC coverage control", _STATUS_REPLACED),
    (r"compare[-_].*hosted.*", "check", "hosted_check.py", "hosted comparison", _STATUS_REPLACED),
    (r"download[-_]campaigns", "census", "census.py", "campaign input download", _STATUS_REPLACED),
    (
        r"resolve[-_].*backend.*",
        "compose",
        "identity_backend_compose.py",
        "backend location resolution",
        _STATUS_REPLACED,
    ),
    (r"relaunch[-_].*backend.*", "compose", "identity_backend_compose.py", "backend relaunch", _STATUS_REPLACED),
    (
        r"complete[-_]identity[-_]index.*",
        "compose",
        "identity_backend_compose.py",
        "identity index completion",
        _STATUS_REPLACED,
    ),
    (r"preserve[-_].*", "census", "census.py", "neighbor preservation into census stage", _STATUS_REPLACED),
    (r"recover[-_].*", "census", "census.py", "overlap recovery into census stage", _STATUS_REPLACED),
    (r"backfill[-_].*", "census", "census.py", "cache backfill into census stage", _STATUS_REPLACED),
    (r"prevalidate[-_].*", "preflight", _PREFLIGHT, "full-load prevalidation (preflight stage)", _STATUS_REPLACED),
    (r"^screenshots$", "check", "hosted_check.py", "screenshot capture", _STATUS_REPLACED),
    (
        r"apply[-_].*schema.*",
        "one-off",
        None,
        "one-off preview schema migration; schema lives in-repo",
        _STATUS_RETIRED_ANALYSIS,
    ),
    (
        r"pending[-_]overlay.*",
        "one-off",
        None,
        "release qualification-hold overlay; hold classification becomes typed in the ingest-pipeline follow-on",
        _STATUS_RETIRED_ANALYSIS,
    ),
    (r"summarize[-_].*", "one-off", None, "one-off cut summary", _STATUS_RETIRED_ANALYSIS),
)

_FALLBACK_NOTE = "no r40 pipeline role identified; docstring recorded for reviewer"


@dataclasses.dataclass(frozen=True)
class Row:
    """One classified file."""

    relative: str
    tier: str
    family: str
    stage: str | None
    replacement: str
    status: str
    note: str
    docstring: str
    size_bytes: int
    modified: str


def _first_docstring(path: Path) -> str:
    """Best-effort module docstring; empty string when absent or unparseable."""
    try:
        if path.suffix != ".py":
            return ""
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        doc = ast.get_docstring(tree)
    except (OSError, SyntaxError, ValueError):
        return ""
    if not doc:
        return ""
    return " ".join(doc.split())[:300]


def _classify_file(relative: Path, path: Path) -> Row:
    stat = path.stat()
    size = stat.st_size
    modified = datetime.fromtimestamp(stat.st_mtime, tz=UTC).isoformat(timespec="seconds")
    stem = path.stem
    if _is_env(relative):
        return Row(
            relative=str(relative),
            tier="environment",
            family=path.suffix.lstrip(".") or "none",
            stage=None,
            replacement="(none — virtualenv tree)",
            status=_STATUS_ENV,
            note="virtualenv/environment tree; regenerated from repo dependencies",
            docstring="",
            size_bytes=size,
            modified=modified,
        )
    if _is_bundled(relative) or stem in _APP_STEMS or stem.startswith("test_") or stem.endswith("_fixtures"):
        in_repo = stem in _APP_STEMS
        note = (
            f"bundled copy of in-repo rl_data_catalog module {stem}.py"
            if in_repo
            else "bundled copy of in-repo code inside a working package"
        )
        return Row(
            relative=str(relative),
            tier="bundled",
            family="bundle",
            stage=None,
            replacement=f"{stem}.py" if in_repo else "in-repo source module; regenerated by package_preview stage",
            status=_STATUS_BUNDLED_COPY,
            note=note,
            docstring="",
            size_bytes=size,
            modified=modified,
        )
    if path.suffix not in (".py", ".sh"):
        return Row(
            relative=str(relative),
            tier="evidence",
            family=path.suffix.lstrip(".") or "none",
            stage=None,
            replacement="(none — run log/evidence)",
            status=_STATUS_EVIDENCE,
            note="run log or evidence artifact; never deleted",
            docstring="",
            size_bytes=size,
            modified=modified,
        )
    for pattern, family, stage, replacement, status in _RULES:
        if re.fullmatch(pattern, stem):
            return Row(
                relative=str(relative),
                tier="script",
                family=family,
                stage=stage,
                replacement=replacement,
                status=status,
                note=replacement if status == _STATUS_REPLACED else _FALLBACK_NOTE,
                docstring=_first_docstring(path),
                size_bytes=size,
                modified=modified,
            )
    return Row(
        relative=str(relative),
        tier="script",
        family="unknown",
        stage=None,
        replacement="(none identified)",
        status=_STATUS_UNCLASSIFIED,
        note=_FALLBACK_NOTE,
        docstring=_first_docstring(path),
        size_bytes=size,
        modified=modified,
    )


def build_rows(root: Path) -> list[Row]:
    return [_classify_file(path.relative_to(root), path) for path in sorted(root.rglob("*")) if path.is_file()]


def _summary(rows: list[Row]) -> str:
    lines = ["| Tier | Status | Files |", "| --- | --- | ---: |"]
    counter = Counter((row.tier, row.status) for row in rows)
    for (tier, status), count in sorted(counter.items()):
        lines.append(f"| {tier} | {status} | {count} |")
    lines.append(f"| | **total** | **{len(rows)}** |")
    return "\n".join(lines)


def _stage_table(rows: list[Row]) -> str:
    script_rows = [row for row in rows if row.tier == "script"]
    lines = [
        "Stage scripts are replaced by one maintained module per stage under",
        "`experiments/post_training/task_curation/release_stages/` (see DESIGN.md).",
        "",
        "| Replacing stage module | Title | Scripts replaced |",
        "| --- | --- | ---: |",
    ]
    per_stage: Counter = Counter(row.stage for row in script_rows if row.stage)
    for module, count in sorted(per_stage.items(), key=lambda item: (-item[1], item[0])):
        spec = STAGES[module]
        lines.append(f"| `release_stages/{module}` | {spec.title} | {count} |")
    retired = Counter(row.status for row in script_rows if row.stage is None)
    for status, count in sorted(retired.items()):
        lines.append(f"| (none) | {status} | {count} |")
    return "\n".join(lines)


def _script_table(rows: list[Row]) -> str:
    script_rows = [row for row in rows if row.tier == "script"]
    lines = [
        "Every non-bundled script, individually mapped. Docstrings are the scripts' own",
        "first docstring line (evidence for the r39 builder's review).",
        "",
        "| Script | Family | Replacing stage | Status | First docstring line |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in script_rows:
        stage = f"`{row.stage}`" if row.stage else "(retired)"
        doc = row.docstring.replace("|", "\\|") or "(none)"
        lines.append(f"| `{row.relative}` | {row.family} | {stage} | {row.status} | {doc} |")
    return "\n".join(lines)


def render(rows: list[Row], root: Path) -> str:
    generated = datetime.now(tz=UTC).isoformat(timespec="seconds")
    bundles = Counter(row.relative.split("/")[0] for row in rows if row.tier == "bundled")
    envs = Counter(row.relative.split("/")[0] for row in rows if row.tier == "environment")
    evidence_exts = Counter(Path(row.relative).suffix for row in rows if row.tier == "evidence")
    parts = [
        "# INVENTORY — atlas-preview-rebuild scripts → in-repo release stages (R40 gate)",
        "",
        f"Generated {generated} by `experiments/post_training/task_curation/release_stages/inventory.py`",
        f"over `{root}`. Read-only: nothing here was modified or deleted; these files are evidence.",
        "",
        "## Scoping",
        "",
        "- **script** — a hand-written `.py`/`.sh` outside bundled and virtualenv trees: mapped",
        "  individually to the `release_stages` module that replaces it, or marked",
        "  RETIRED-ANALYSIS / NEEDS-REVIEW with its docstring recorded.",
        "- **bundled** — copies of in-repo `rl_data_catalog` modules and tests inside working",
        "  packages (app bundles, journeys, candidate/package/dist/server trees): BUNDLED-COPY,",
        "  regenerated by the packaging stage; the source of truth is the repo module.",
        "- **environment** — virtualenv trees: ENVIRONMENT, regenerated from repo dependencies.",
        "- **evidence** — logs, JSON receipts, screenshots, docs, patches: RETIRED-EVIDENCE.",
        "",
        "R40 uses none of these files: every stage runs from a landed in-repo module through the",
        "typed release graph (see `DESIGN.md` shipped alongside `inventory.py`).",
        "",
        "## Summary",
        "",
        _summary(rows),
        "",
        "## Stage mapping",
        "",
        _stage_table(rows),
        "",
        "## Bundled-copy directories",
        "",
        "| Working directory | Bundled files |",
        "| --- | ---: |",
    ]
    for directory, count in sorted(bundles.items()):
        parts.append(f"| `{directory}/` | {count} |")
    parts += [
        "",
        "## Environment (virtualenv) directories",
        "",
        "| Directory | Files |",
        "| --- | ---: |",
    ]
    for directory, count in sorted(envs.items()):
        parts.append(f"| `{directory}/` | {count} |")
    parts += [
        "",
        "## Evidence files by extension",
        "",
        "| Extension | Files |",
        "| --- | ---: |",
    ]
    for suffix, count in sorted(evidence_exts.items(), key=lambda item: (-item[1], item[0])):
        parts.append(f"| {suffix or '(none)'} | {count} |")
    parts += [
        "",
        "## Full script mapping",
        "",
        _script_table(rows),
        "",
        "## NEEDS-REVIEW / RETIRED-ANALYSIS rows (r39 builder review requested)",
        "",
    ]
    needs_review = [row for row in rows if row.status in (_STATUS_UNCLASSIFIED, _STATUS_RETIRED_ANALYSIS)]
    if not needs_review:
        parts.append("(none)")
    else:
        for row in needs_review:
            parts.append(f"- `{row.relative}` — {row.docstring or '(no docstring)'}")
    return "\n".join(parts) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rebuild-root",
        type=Path,
        default=Path.home() / "Documents/experiments/active/atlas-preview-rebuild",
        help="Historical preview-rebuild directory to inventory (read-only).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output path (default: <rebuild-root>/INVENTORY.md).",
    )
    args = parser.parse_args()
    root = args.rebuild_root.expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"rebuild root does not exist: {root}")
    rows = build_rows(root)
    out = (args.out.expanduser() if args.out else root / "INVENTORY.md").resolve()
    out.write_text(render(rows, root), encoding="utf-8")
    print(f"wrote {out} ({len(rows)} files classified)")


if __name__ == "__main__":
    main()
