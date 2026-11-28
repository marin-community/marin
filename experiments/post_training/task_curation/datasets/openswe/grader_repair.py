# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Deterministic classification and repair of OpenSWE graders that pass a no-op agent.

Coordinator ruling 2026-11-27: QC verdicts never block admission, but a grader
that passes a no-op agent is a broken grader, and graders fail closed. The
upstream OpenSWE eval_scripts embed the golden (reference) patch and apply it to
``/testbed`` before running the tests, so the graded state is the reference
solution and an empty-diff agent passes. This module classifies every row's
eval_script by byte-matching the script's embedded diffs against the row's
recorded golden patch and test patch, repairs the repairable graders (drop the
golden-patch application so the agent's diff is what gets tested; scope any
unscoped working-tree reset to the test files so the agent's diff survives), and
types the unrepairable rows ``grader passes no-op (golden patch applied)``.

The classifier never guesses: a heredoc body counts as the golden patch only on
byte evidence (an exact match, or a prefix of the recorded patch — the upstream
harness truncated some embedded patches). Anything the transform cannot excise
as a clean block leaves the row unrepairable; the fidelity gate's no-op control
then adjudicates the residual classes behaviorally on its sample.
"""

import hashlib
import itertools
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from experiments.post_training.task_curation.datasets.openswe.sources import OpenSweRow

EXCLUSION_REASON = "grader passes no-op (golden patch applied)"

_HEREDOC_START = re.compile(r"<<-?\s*(['\"]?)(?P<delimiter>[A-Za-z_][A-Za-z_0-9]*)\1")
_APPLY_LINE = re.compile(r"^\s*(sudo\s+)?git apply\b")
_ASSIGNMENT_LINE = re.compile(r"^\s*(?:local\s+)?(?P<name>[A-Za-z_][A-Za-z_0-9]*)=.*<<")
_FULL_TREE_RESET = re.compile(
    r"^(?P<indent>\s*)(sudo\s+)?(?P<command>git (checkout -- \.|checkout HEAD -- \.|checkout -f \.|reset\s+[^#\n]*--hard\S*))(?P<suffix>[^\n]*)$",
    re.MULTILINE,
)
# The exit-code capture line a golden application's status handler reads right
# after the excised block (`patch_rc=$?`): once the block is gone the capture
# would record some unrelated command's status, so the repair pins it to 0 —
# the removed application's success value — and the script's own gating on it
# keeps its meaning.
_STATUS_CAPTURE = re.compile(r"^(?P<indent>[ \t]*)(?:local\s+)?(?P<name>[A-Za-z_][A-Za-z_0-9]*)=\$\?\s*$", re.MULTILINE)


@dataclass(frozen=True)
class DiffSite:
    """One heredoc-embedded diff in the eval_script and what its bytes match."""

    start: int
    end: int
    delimiter: str
    kind: str  # "golden" | "test" | "unknown"
    match: str  # "exact" | "prefix" | "none"
    body_lines: int
    consuming: str  # "git_apply" | "assignment" | "other"


@dataclass(frozen=True)
class ResetSite:
    """One unscoped working-tree reset in the graded flow."""

    start: int
    end: int
    line: str
    indent: str = ""
    suffix: str = ""


@dataclass
class GraderClassification:
    """The scan verdict for one row's grader, with the repair material."""

    instance_id: str
    verdict: str  # "clean" | "repairable" | "unrepairable"
    reason: str | None
    detail: str
    golden_sites: tuple[DiffSite, ...] = field(default_factory=tuple)
    test_sites: tuple[DiffSite, ...] = field(default_factory=tuple)
    unknown_sites: tuple[DiffSite, ...] = field(default_factory=tuple)
    reset_sites: tuple[ResetSite, ...] = field(default_factory=tuple)
    repaired_script: str | None = None
    golden_digest: str | None = None
    test_digest: str | None = None

    def as_dict(self) -> dict:
        return {
            "instance_id": self.instance_id,
            "verdict": self.verdict,
            "reason": self.reason,
            "detail": self.detail,
            "golden_sites": len(self.golden_sites),
            "test_sites": len(self.test_sites),
            "unknown_sites": len(self.unknown_sites),
            "reset_sites": len(self.reset_sites),
            "golden_digest": self.golden_digest,
            "test_digest": self.test_digest,
        }


def _heredoc_blocks(script: str) -> list[tuple[int, int, str, str]]:
    """Every heredoc block as (start, end, delimiter, body); never nested.

    A block starts at the ``<<`` that opens it and ends at the delimiter line.
    Parsing is sequential: a delimiter line closes exactly one open block, so
    scripts that reuse one delimiter for several patches (the upstream harness
    does) still parse.
    """
    blocks: list[tuple[int, int, str, str]] = []
    lines = script.splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))
    open_block = None  # (delimiter, start_offset, body_lines)
    for index, line in enumerate(lines):
        offset = offsets[index]
        if open_block is not None:
            delimiter, start, body = open_block
            if line.strip() == delimiter:
                blocks.append((start, offsets[index + 1], delimiter, "".join(body)))
                open_block = None
            else:
                body.append(line)
            continue
        match = _HEREDOC_START.search(line)
        if match is None:
            continue
        open_block = (match.group("delimiter"), offset, [])
    return blocks


def _diff_file_paths(diff: str) -> set[str]:
    """The files a unified diff touches (the ``+++ b/`` and ``--- a/`` targets)."""
    paths: set[str] = set()
    for line in diff.splitlines():
        for prefix in ("--- a/", "+++ b/", "--- ", "+++ "):
            if line.startswith(prefix):
                path = line[len(prefix) :].strip()
                if path and path not in ("/dev/null", ""):
                    paths.add(path)
    return paths


def _match_kind(body: str, golden: str, test: str) -> tuple[str, str]:
    """What one heredoc body's bytes are: golden, test, or unknown."""
    stripped = body.rstrip("\n")
    for name, reference in (("golden", golden), ("test", test)):
        expected = reference.rstrip("\n")
        if not expected.strip():
            continue
        if stripped == expected:
            return name, "exact"
        if len(stripped) > 40 and expected.startswith(stripped):
            return name, "prefix"
    return "unknown", "none"


def _consuming_kind(opening_line: str) -> str:
    if _APPLY_LINE.search(opening_line):
        return "git_apply"
    if _ASSIGNMENT_LINE.search(opening_line):
        return "assignment"
    return "other"


def _line_end(script: str, offset: int) -> int:
    end = script.find("\n", offset)
    return len(script) if end == -1 else end + 1


def classify_grader(row: OpenSweRow) -> GraderClassification:
    """Classify one row's grader deterministically from its own bytes."""
    script = row.eval_script
    golden = row.patch or ""
    test = row.test_patch or ""
    golden_digest = hashlib.sha256(golden.encode()).hexdigest() if golden.strip() else None
    test_digest = hashlib.sha256(test.encode()).hexdigest() if test.strip() else None
    classification = GraderClassification(
        instance_id=row.instance_id,
        verdict="clean",
        reason=None,
        detail="",
        golden_digest=golden_digest,
        test_digest=test_digest,
    )
    # A grader that never runs cannot be repaired into one that grades: the
    # verifier executes the upstream bytes verbatim, so a script bash refuses
    # to parse scores 0 for every agent state, no-op and oracle alike.
    parse = subprocess.run(["bash", "-n"], input=script.encode(), capture_output=True, timeout=60)
    if parse.returncode != 0:
        classification.verdict = "unrepairable"
        classification.reason = "eval_script_syntax_error"
        classification.detail = (
            "the eval_script does not parse under bash: "
            + parse.stderr.decode(errors="replace").strip()[-400:]
        )
        return classification
    # The verifier's ambiguity rule scores a run 0 when more than one
    # OPENSWE_EXIT_CODE line reaches its output; a script that echoes the
    # verdict several times (phased upstream scripts do) therefore can never
    # award 1, whatever the agent does.
    markers = re.findall(r"^\s*(?:echo|printf)\b[^\n]*OPENSWE_EXIT_CODE", script, re.MULTILINE)
    if len(markers) > 1:
        classification.verdict = "unrepairable"
        classification.reason = "ambiguous_exit_code"
        classification.detail = (
            f"the eval_script echoes the OPENSWE_EXIT_CODE verdict {len(markers)} times; the verifier's "
            "ambiguity rule scores every such run 0"
        )
        return classification
    golden_sites: list[DiffSite] = []
    test_sites: list[DiffSite] = []
    unknown_sites: list[DiffSite] = []
    for start, end, delimiter, body in _heredoc_blocks(script):
        kind, match = _match_kind(body, golden, test)
        # The block's start is the opening line's own offset.
        consuming = _consuming_kind(script[start : _line_end(script, start)])
        site = DiffSite(
            start=start,
            end=end,
            delimiter=delimiter,
            kind=kind,
            match=match,
            body_lines=body.count("\n"),
            consuming=consuming,
        )
        if kind == "golden":
            golden_sites.append(site)
        elif kind == "test":
            test_sites.append(site)
        else:
            unknown_sites.append(site)
    reset_sites = [
        ResetSite(
            start=match.start(),
            end=_line_end(script, match.start()),
            line=match.group("command").strip(),
            indent=match.group("indent"),
            suffix=match.group("suffix"),
        )
        for match in _FULL_TREE_RESET.finditer(script)
    ]
    classification.golden_sites = tuple(golden_sites)
    classification.test_sites = tuple(test_sites)
    classification.unknown_sites = tuple(unknown_sites)
    classification.reset_sites = tuple(reset_sites)
    if not golden_sites:
        file_applies = [
            match.group(0).strip()[:110]
            for match in re.finditer(r"^\s*(sudo\s+)?git apply\b(?![^\n]*<<)[^\n]*$", script, re.MULTILINE)
        ]
        detail = (
            f"no golden patch application; {len(test_sites)} test diff site(s), "
            f"{len(unknown_sites)} unknown diff site(s), {len(reset_sites)} unscoped reset(s)"
        )
        if file_applies:
            detail += f"; file-based apply line(s) recorded for the behavioral gate: {'; '.join(file_applies)}"
        classification.detail = detail
        # Fail closed on an embedded diff that is neither the recorded golden
        # nor the recorded test patch but touches the golden patch's own files:
        # a modified golden application is a no-op-passing grader the byte
        # matcher cannot certify. Diffs on other files are the eval scripts'
        # setup patches and stay.
        golden_paths = _diff_file_paths(golden)
        for site in unknown_sites:
            body = next(
                block_body for block_start, _, _, block_body in _heredoc_blocks(script) if block_start == site.start
            )
            if site.body_lines >= 4 and _diff_file_paths(body) & golden_paths:
                classification.verdict = "unrepairable"
                classification.reason = EXCLUSION_REASON
                classification.detail = (
                    f"an embedded diff at offset {site.start} matches neither the recorded golden patch nor "
                    "a prefix of it but touches the golden patch's own files; the row fails closed"
                )
                return classification
        return classification
    if any(site.consuming == "other" for site in golden_sites):
        unexcisable = next(site for site in golden_sites if site.consuming == "other")
        classification.verdict = "unrepairable"
        classification.reason = EXCLUSION_REASON
        classification.detail = (
            f"the golden patch is applied from a non-heredoc site at offset {unexcisable.start}; "
            "the application cannot be excised as a clean block"
        )
        return classification
    if any(site.match == "none" for site in golden_sites):
        weak = next(site for site in golden_sites if site.match == "none")
        classification.verdict = "unrepairable"
        classification.reason = EXCLUSION_REASON
        classification.detail = (
            f"a golden-patch heredoc at offset {weak.start} matches neither the recorded golden patch "
            "nor a prefix of it; removing it could not be verified against the row's own bytes"
        )
        return classification
    classification.verdict = "repairable"
    classification.detail = (
        f"{len(golden_sites)} golden site(s) to excise "
        f"({', '.join(sorted({site.match for site in golden_sites}))}), "
        f"{len(test_sites)} test diff site(s), {len(unknown_sites)} unknown diff site(s), "
        f"{len(reset_sites)} unscoped reset(s) to scope"
    )
    return classification


def _test_file_paths(test_patch: str) -> list[str]:
    """The test files a recorded test patch touches, for the scoped reset."""
    paths: list[str] = []
    for line in test_patch.splitlines():
        if line.startswith("+++ b/"):
            path = line[len("+++ b/") :].strip()
            if path and path != "/dev/null" and path not in paths:
                paths.append(path)
    return paths


def repaired_script(row: OpenSweRow, classification: GraderClassification) -> str:
    """The repaired eval_script: the golden application dropped, resets scoped.

    A golden patch applied through a direct stdin heredoc is excised together
    with immediately preceding comment or echo lines that announce it. A golden
    patch assigned through a command substitution (the upstream moded scripts'
    ``local golden=$(cat <<'EOF' ...)``) keeps its wrapper and loses only its
    body: the variable becomes the empty patch, the branch structure stays
    valid, and the golden application becomes a no-op. An unscoped working-tree
    reset becomes a scoped reset of the test files (the tamper guard the scoped
    SWE-bench reset provides); with no test files to name, the reset line is
    dropped — the row's tamper surface is then the agent-visible tree, which the
    SAMPLE evidence records.
    """
    if classification.verdict != "repairable":
        raise ValueError(f"the grader is not repairable: {classification.instance_id}")
    script = row.eval_script
    excisions: list[tuple[int, int]] = []
    replacements: list[tuple[int, int, str]] = []
    test_paths = _test_file_paths(row.test_patch or "")
    for site in classification.golden_sites:
        if site.consuming == "assignment":
            body_start = _line_end(script, site.start)
            terminator_start = script.rfind(f"\n{site.delimiter}", body_start, site.end) + 1
            if terminator_start <= body_start:
                raise ValueError(f"the golden heredoc body is not locatable in {classification.instance_id}")
            excisions.append((body_start, terminator_start))
            continue
        cut_start = site.start
        # Pull in immediately preceding comment/echo lines that announce the
        # golden application, so the repaired flow carries no stale references.
        while cut_start > 0:
            line_start = script.rfind("\n", 0, cut_start - 1) + 1
            line = script[line_start:cut_start].strip()
            if (line.startswith("#") or line.startswith("echo ")) and "golden" in line.lower():
                cut_start = line_start
                continue
            break
        cut_end = site.end
        excisions.append((cut_start, cut_end))
        # The status capture that read the excised application's exit code
        # would now record some unrelated command's status; pin it to 0 — the
        # removed application's success value — so the script's own gating on
        # it keeps its meaning.
        capture = _STATUS_CAPTURE.search(script, cut_end)
        if capture is not None and all(
            not script[offset].strip() for offset in range(cut_end, min(capture.start(), len(script)))
        ):
            line_end = script.find("\n", capture.end())
            line_end = len(script) if line_end == -1 else line_end + 1
            replacements.append(
                (
                    capture.start(),
                    line_end,
                    f"{capture.group('indent')}{capture.group('name')}=0\n",
                )
            )
    for site in classification.reset_sites:
        if test_paths:
            scoped = f"{site.indent}git checkout HEAD -- " + " ".join(test_paths) + f"{site.suffix}\n"
            replacements.append((site.start, site.end, scoped))
            continue
        cut_start = site.start
        # Preceding echo lines announce the reset; drop them with it.
        while cut_start > 0:
            previous_start = script.rfind("\n", 0, cut_start - 1) + 1
            previous = script[previous_start:cut_start].strip()
            if previous.startswith("echo ") and "reset" in previous.lower():
                cut_start = previous_start
                continue
            break
        excisions.append((cut_start, site.end))
    all_cuts = sorted([(start, end) for start, end in excisions] + [(start, end) for start, end, _ in replacements])
    for (_start, end), (next_start, _next_end) in itertools.pairwise(all_cuts):
        if next_start < end:
            raise ValueError(f"overlapping repair cuts in {classification.instance_id}")
    # One descending pass: every cut's offsets are valid in the original bytes.
    edits = sorted(
        [(start, end, None) for start, end in excisions] + [(start, end, text) for start, end, text in replacements],
        key=lambda cut: cut[0],
        reverse=True,
    )
    result = script
    for start, end, text in edits:
        result = result[:start] + (text if text is not None else "") + result[end:]
    return result


def scan_rows(rows: list[OpenSweRow]) -> dict:
    """Classify every row; return the scan counts and the per-row records."""
    counts = {"scanned": len(rows), "clean": 0, "repairable": 0, "unrepairable": 0}
    by_reason: dict[str, int] = {}
    records: list[dict] = []
    for row in rows:
        classification = classify_grader(row)
        counts[classification.verdict] += 1
        if classification.reason is not None:
            by_reason[classification.reason] = by_reason.get(classification.reason, 0) + 1
        records.append(classification.as_dict())
    return {
        "counts": counts,
        "unrepairable_by_reason": dict(sorted(by_reason.items())),
        "records": records,
    }


def main() -> None:
    import argparse  # noqa: PLC0415
    import json  # noqa: PLC0415
    import logging  # noqa: PLC0415

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True, help="Write the scan report JSON here.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    LOGGER = logging.getLogger(__name__)

    from experiments.post_training.task_curation.datasets.openswe.sources import keep_release_rows  # noqa: PLC0415

    rows = keep_release_rows()
    LOGGER.info("scanning %d rows' graders", len(rows))
    report = scan_rows(rows)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    LOGGER.info("scan counts: %s | unrepairable by reason: %s", report["counts"], report["unrepairable_by_reason"])
    print(json.dumps({"counts": report["counts"], "unrepairable_by_reason": report["unrepairable_by_reason"]}))


if __name__ == "__main__":
    main()
