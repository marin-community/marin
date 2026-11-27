# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Release a policy version: pin each baseline's run of each evaluation in a manifest, and tag its commit.

The manifest is written once. The annotated tag ``eval-policy-<version>`` on ``origin``, at the manifest's
commit, makes it the release; a version without that tag is unreleased, whatever manifest exists.
"""

import hashlib
from collections.abc import Collection
from dataclasses import dataclass, replace
from pathlib import Path

from evalrecords.policy_release import PolicyRelease, ReleasedRun, record_sha256, release_path, release_tag
from evalrecords.records import EvalRunRecord
from finestore.reader import ReadView
from rigging.filesystem.conditional_object import ConditionalWriteError, conditional_object
from rigging.filesystem.factory import open_url
from rigging.provenance import git_output

from experiments.evaluation.policy.launch import ModelEvaluations, PolicyModel, model_evaluations, policy_model
from experiments.evaluation.policy.runs import ExpectedRun, compliant_run, expected_runs, found_runs, run_suffix
from experiments.evaluation.policy.schema import Policy

EVALUATION_CODE = (
    "experiments/evaluation",
    "lib/evalrecords",
    "lib/finestore",
    "lib/marin/src/marin/evaluation",
    "lib/marin/src/marin/inference",
    "lib/marin/src/marin/rollouts",
    "lib/rolloutengine",
    "lib/taskcompendium",
    "lib/verifyit",
)
"""Paths whose changes since a released run's commit the release's reviewer reads for side effects."""


def tagged_commit(version: str, root: Path) -> str | None:
    """The commit ``origin``'s release tag for ``version`` points at, or None if it is unreleased."""
    tag = f"refs/tags/{release_tag(version)}"
    lines = dict(
        reversed(line.split("\t")) for line in git_output(["ls-remote", "origin", tag, f"{tag}^{{}}"], root).splitlines()
    )
    return lines.get(f"{tag}^{{}}", lines.get(tag))


def read_release(version: str, root: Path) -> PolicyRelease:
    """The release of ``version``: its manifest, which must be at the commit ``origin``'s tag names."""
    commit = tagged_commit(version, root)
    if commit is None:
        raise ValueError(f"{version} is not released: origin has no tag {release_tag(version)}")
    with open_url(release_path(version), "rb") as handle:
        release = PolicyRelease.model_validate_json(handle.read())
    if (release.version, release.commit) != (version, commit):
        raise ValueError(f"{release_path(version)} is {release.version} at {release.commit}, not the tagged {commit}")
    return release


def carried_runs(policy: Policy, model: PolicyModel, plan: ModelEvaluations, root: Path) -> dict[str, ReleasedRun]:
    """The base release's runs a patch keeps for one baseline: those whose inputs, apart from the version, are what
    this checkout would run."""
    if policy.base is None or not model.baseline or not plan.evaluations:
        return {}
    released = {run.evaluation: run for run in read_release(policy.base, root).runs if run.model == model.name}
    base = replace(policy, version=policy.base, base=None)
    suffixes = {evaluation.name: run_suffix("0" * 12, 1) for evaluation in plan.evaluations}
    return {
        expected.name: run.model_copy(update={"carried_from": run.carried_from or policy.base})
        for expected in expected_runs(base, plan, limit=None, run_suffixes=suffixes)
        if (run := released.get(expected.name)) is not None and run.run_id.startswith(f"{expected.base}-")
    }


def without(plan: ModelEvaluations, names: Collection[str]) -> ModelEvaluations:
    return replace(
        plan, evaluations=tuple(evaluation for evaluation in plan.evaluations if evaluation.name not in names)
    )


@dataclass(frozen=True)
class ReleaseCandidate:
    """What a release at this checkout would pin, what it lacks, and what its reviewer must read."""

    release: PolicyRelease
    missing: tuple[str, ...]
    """Each baseline and evaluation without a compliant full run of the inputs this checkout would run."""
    warnings: tuple[str, ...]
    changes: dict[str, str]
    """For each older commit a pinned run comes from, the evaluation code commits since it."""


def _released_run(model: str, expected: ExpectedRun, record: EvalRunRecord, scored: EvalRunRecord) -> ReleasedRun:
    """Pin a run: ``record`` as stored, ``scored`` with the metrics its archive gives."""
    token = ReadView(record.results_path).token
    assert token is not None, "a compliant run has an archive"
    return ReleasedRun(
        model=model,
        evaluation=expected.name,
        run_id=record.run_id,
        records_prefix=expected.batch.records_prefix,
        commit=record.provenance.git_sha,
        record_sha256=record_sha256(expected.batch.records_prefix, record.run_id),
        archive_version=token.version,
        scores=scored.metrics,
    )


def release_candidate(policy: Policy, root: Path) -> ReleaseCandidate:
    """Pick, for every baseline and evaluation, the run a release at HEAD would pin.

    A patch keeps each base release run whose inputs are unchanged apart from the version. Every other pair needs a
    compliant full run of the inputs HEAD would run, at HEAD or an earlier commit in its history.
    """
    head = git_output(["rev-parse", "HEAD"], root)
    runs, missing, warnings, since = [], [], [], {}
    for name in policy.baselines:
        model = policy_model(policy, policy.baseline(name), baseline=True)
        plan = model_evaluations(policy, model)
        carried = carried_runs(policy, model, plan, root)
        runs.extend(carried.values())
        rest = without(plan, carried)
        suffixes = {evaluation.name: run_suffix(head[:12], 1) for evaluation in rest.evaluations}
        for expected in expected_runs(policy, rest, limit=None, run_suffixes=suffixes):
            in_history = tuple(
                run
                for run in found_runs(expected)
                if run.record is not None and is_ancestor(run.record.provenance.git_sha, head, root)
            )
            if (hit := compliant_run(in_history, expected)) is None:
                missing.append(f"{name} {expected.name}: no compliant full run in HEAD's history")
                continue
            found, check = hit
            assert found.record is not None
            runs.append(_released_run(name, expected, found.record, check.record))
            warnings.extend(f"{name} {expected.name}: {warning}" for warning in check.warnings)
    for commit in sorted({run.commit for run in runs} - {head}):
        since[commit] = git_output(["log", "--oneline", f"{commit}..{head}", "--", *EVALUATION_CODE], root)
    release = PolicyRelease(version=policy.version, commit=head, base=policy.base, runs=tuple(runs))
    return ReleaseCandidate(release=release, missing=tuple(missing), warnings=tuple(warnings), changes=since)


def is_ancestor(commit: str, head: str, root: Path) -> bool:
    """Whether ``commit`` is in ``head``'s history; an unknown commit is not."""
    try:
        git_output(["merge-base", "--is-ancestor", commit, head], root)
    except RuntimeError:
        return False
    return True


def publish(release: PolicyRelease, root: Path) -> str:
    """Write the manifest once and push the annotated tag naming it; return the manifest's digest.

    A retry with the same manifest finishes an interrupted release. A different manifest already written for the
    version, or a tag at another commit, fails without changing either.
    """
    content = release.content()
    digest = hashlib.sha256(content).hexdigest()
    path, tag = release_path(release.version), release_tag(release.version)
    tagged = tagged_commit(release.version, root)
    if tagged is not None and tagged != release.commit:
        raise ValueError(f"{tag} already names {tagged}")
    manifest = conditional_object(path)
    try:
        manifest.write(content, expected_version=None)
    except ConditionalWriteError:
        existing = manifest.read()
        if existing is None or existing.data != content:
            raise ValueError(f"{path} already holds another manifest for {release.version}") from None
    if tagged is None:
        local = git_output(["rev-parse", "-q", "--verify", f"refs/tags/{tag}^{{commit}}"], root, check=False)
        if local and local != release.commit:
            raise ValueError(f"a local tag {tag} names {local}; delete it")
        if not local:
            message = f"Eval policy {release.version}\n\n{path}\nsha256 {digest}"
            git_output(["tag", "-a", tag, release.commit, "-m", message], root)
        git_output(["push", "origin", f"refs/tags/{tag}"], root)
    return digest
