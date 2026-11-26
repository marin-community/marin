# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The OpenSWE dedicated converter profile: exit-code grading fails closed, the
upstream eval_script rides verbatim, and the Dockerfile translation never invents
bytes (ruling 2026-11-26 6:03 AM, option a)."""

import base64
import json

import pytest
from taskcompendium.models import NoGrader, ScriptGrader, TaskSpec
from taskcompendium.pipeline.inputs import ConversionContext

from experiments.post_training.task_curation.datasets.openswe import bases, convert, grader, launch_base_build
from experiments.post_training.task_curation.datasets.openswe.dockerfile_translation import (
    DockerfileRejected,
    dockerfile_instructions,
    translate_dockerfile,
)
from experiments.post_training.task_curation.datasets.openswe.sources import (
    KEEP_RELEASE_REPO,
    UPSTREAM_REVISION,
    OpenSweRow,
)

CONDENSED = {"3.11": "ghcr.io/marin-community/rl-data-openswe@sha256:" + "a" * 64}

CANONICAL_DOCKERFILE = """FROM openswe-python-3.11

COPY repo /testbed

WORKDIR /testbed

RUN bash -lc 'pip install -e ".[test]"'
"""

LICENSE_PARTITION = {
    "repo_rule": "allow",
    "repo_verdict": "covered",
    "repo_license_ids": ["MIT"],
    "repo_pinned_commit": "b" * 40,
    "verification": "base_commit_byte_identical",
}


def row(**overrides) -> OpenSweRow:
    payload = {
        "instance_id": "acme__widget-1",
        "repo": "acme/widget",
        "base_commit": "b" * 40,
        "problem_statement": "Fix the widget.",
        "patch": "diff --git a/w.py b/w.py\n--- a/w.py\n+++ b/w.py\n",
        "test_patch": "diff --git a/test_w.py b/test_w.py\n",
        "eval_script": '#!/bin/bash\ncd /testbed\npytest -q\necho "OPENSWE_EXIT_CODE=$rc"\n',
        "dockerfile": CANONICAL_DOCKERFILE,
        "image_name": "openswe--acme__widget-1",
        "license_partition": LICENSE_PARTITION,
    }
    payload.update(overrides)
    return OpenSweRow(**payload)


def test_reward_fails_closed_on_every_nonzero_or_ambiguous_verdict():
    """The ruling's fail-closed contract: one exact OPENSWE_EXIT_CODE=0 line scores 1;
    a missing marker, a nonzero exit, a malformed value, or an ambiguous second
    marker line all score 0."""
    assert grader.reward(">>>>> Start\nOPENSWE_EXIT_CODE=0\n>>>>> End") == 1
    assert grader.reward("OPENSWE_EXIT_CODE=1") == 0
    assert grader.reward("no marker at all") == 0
    assert grader.reward("OPENSWE_EXIT_CODE=0 ") == 0  # trailing space: not the exact verdict
    assert grader.reward("OPENSWE_EXIT_CODE=00") == 0
    assert grader.reward("OPENSWE_EXIT_CODE=0x1") == 0
    assert grader.reward("OPENSWE_EXIT_CODE=1\nOPENSWE_EXIT_CODE=0") == 0  # ambiguous
    assert grader.reward("OPENSWE_EXIT_CODE=0\nOPENSWE_EXIT_CODE=0") == 0  # doubled echo: ambiguous


def test_verifier_script_runs_the_eval_script_verbatim_and_starts_at_zero():
    """The verifier's own bytes fail closed: reward 0 before anything runs, the
    upstream script executed as-is, the verdict parsed from its output only."""
    script = grader.verifier_script().decode()
    assert "echo 0 > /logs/verifier/reward.txt" in script
    assert "bash /tests/eval_script.sh > /logs/verifier/eval-output.txt 2>&1" in script
    assert script.index("echo 0 >") < script.index("bash /tests/eval_script.sh")
    assert "grep -c '^OPENSWE_EXIT_CODE=' /logs/verifier/eval-output.txt" in script
    assert "grep -c '^OPENSWE_EXIT_CODE=0$' /logs/verifier/eval-output.txt" in script
    assert script.rstrip().endswith("exit 0")
    assert 'total" -eq 1 ] && [ "$zero" -eq 1' in script


def test_dockerfile_instructions_skip_heredoc_bodies_and_join_continuations():
    """Machine-authored recipes embed shell heredocs whose bodies carry
    instruction-looking lines; those are shell data, never instructions."""
    text = (
        "RUN cat > /tmp/t.py <<'PYEOF'\n"
        "from typing import NamedTuple\n"
        "FROM nothing at all\n"
        "PYEOF\n"
        "RUN printf '%s' \\\n"
        "  'continued'\n"
        "COPY repo /testbed\n"
        "# a comment line\n"
        "WORKDIR /testbed\n"
        "RUN cat <<EOF\n"
        "FROM also-inside\n"
        "EOF\n"
    )
    instructions = dockerfile_instructions(text)
    names = [(instruction.name, instruction.argument) for instruction in instructions]
    assert ("RUN", "cat > /tmp/t.py <<'PYEOF'") in names
    assert ("RUN", "printf '%s' 'continued'") in names
    assert ("COPY", "repo /testbed") in names
    assert ("WORKDIR", "/testbed") in names
    assert ("RUN", "cat <<EOF") in names
    assert all(
        "typing" not in argument and "nothing" not in argument and "also-inside" not in argument for _, argument in names
    )


def test_translate_dockerfile_binds_the_condensed_base_and_clones_the_repo():
    translation = translate_dockerfile(
        CANONICAL_DOCKERFILE, repo="acme/widget", base_commit="b" * 40, condensed_bases=CONDENSED
    )
    assert "FROM " + CONDENSED["3.11"] in translation.text
    assert "openswe-python-3.11" not in translation.text
    assert (
        "RUN git clone https://github.com/acme/widget.git '/testbed' \\\n"
        " && git -C '/testbed' checkout --quiet '" + "b" * 40 + "' \\\n"
        " && git -C '/testbed' remote remove origin"
    ) in translation.text
    assert "COPY repo /testbed" not in translation.text
    assert "RUN bash -lc 'pip install -e \".[test]\"'" in translation.text  # untouched bytes
    assert translation.base_group == "3.11"
    assert translation.declared_bases == ("openswe-python-3.11",)
    assert translation.workdir == "/testbed"


def test_translate_dockerfile_resolves_other_registry_bases_through_the_shared_resolver():
    text = "FROM continuumio/miniconda3:25.3.1-1\nCOPY repo /testbed\n"
    pinned = "continuumio/miniconda3@sha256:" + "c" * 64

    def resolve(reference: str) -> str:
        assert reference == "continuumio/miniconda3:25.3.1-1"
        return pinned

    translation = translate_dockerfile(text, repo="r/x", base_commit="b" * 40, condensed_bases={}, resolve_base=resolve)
    assert f"FROM {pinned}" in translation.text
    assert translation.base_group is None


def test_translate_dockerfile_rejects_unresolvable_and_variable_bases():
    with pytest.raises(DockerfileRejected, match="does not resolve"):
        translate_dockerfile(
            "FROM quay.io/missing/image:1\n",
            repo="r/x",
            base_commit="b" * 40,
            condensed_bases={},
            resolve_base=lambda reference: (_ for _ in ()).throw(ValueError("nope")),
        )
    with pytest.raises(DockerfileRejected, match="not a literal reference"):
        translate_dockerfile("FROM $BASE\nCOPY repo /testbed\n", repo="r/x", base_commit="b" * 40, condensed_bases={})
    with pytest.raises(DockerfileRejected, match="no digest-pinned condensed base"):
        translate_dockerfile(CANONICAL_DOCKERFILE, repo="r/x", base_commit="b" * 40, condensed_bases={})
    with pytest.raises(DockerfileRejected, match="no resolver is staged"):
        translate_dockerfile(
            "FROM python:3.12-slim\nCOPY repo /testbed\n", repo="r/x", base_commit="b" * 40, condensed_bases={}
        )


def test_translate_dockerfile_passes_stage_references_through_and_pins_only_roots():
    text = "FROM openswe-python-3.10 AS builder\n" "COPY repo /build\n" "RUN make\n" "FROM builder\n" "WORKDIR /build\n"
    python310 = {"3.10": "ghcr.io/marin-community/rl-data-openswe@sha256:" + "d" * 64}
    translation = translate_dockerfile(text, repo="r/x", base_commit="b" * 40, condensed_bases=python310)
    assert translation.text.count("FROM ") == 2
    assert translation.text.splitlines()[0] == "FROM " + python310["3.10"] + " AS builder"
    assert "FROM builder" in translation.text  # the stage re-reference stays verbatim
    assert translation.declared_bases == ("openswe-python-3.10",)


def test_translate_dockerfile_rewrites_partial_repo_copies_with_a_staging_clone():
    """The layer-caching shape: requirements copied before the full repo copy."""
    text = (
        "FROM openswe-python-3.11\n"
        "COPY repo/requirements.txt /tmp/requirements.txt\n"
        "RUN pip install -r /tmp/requirements.txt\n"
        "COPY repo /testbed\n"
        "WORKDIR /testbed\n"
        "RUN pip install -e .\n"
    )
    translation = translate_dockerfile(text, repo="acme/widget", base_commit="b" * 40, condensed_bases=CONDENSED)
    lines = translation.text.splitlines()
    assert lines[0] == "FROM " + CONDENSED["3.11"]
    # the staging clone is inserted before the first repo copy; both copies read from it
    assert (
        "RUN git clone https://github.com/acme/widget.git '/tmp/openswe-repo-staging' \\\n"
        " && git -C '/tmp/openswe-repo-staging' checkout --quiet '" + "b" * 40 + "' \\\n"
        " && git -C '/tmp/openswe-repo-staging' remote remove origin"
    ) in translation.text
    assert (
        "mkdir -p '/tmp' && cp -a '/tmp/openswe-repo-staging/requirements.txt' '/tmp/requirements.txt'"
        in translation.text
    )
    assert "mkdir -p '/testbed' && cp -a '/tmp/openswe-repo-staging/.' '/testbed'" in translation.text
    assert "COPY repo" not in translation.text


def test_translate_dockerfile_rewrites_multi_source_repo_copies():
    text = "FROM openswe-python-3.11\n" "COPY repo/pyproject.toml repo/poetry.lock* /testbed/\n" "COPY repo /testbed\n"
    translation = translate_dockerfile(text, repo="acme/widget", base_commit="b" * 40, condensed_bases=CONDENSED)
    assert "cp -a '/tmp/openswe-repo-staging/pyproject.toml' '/testbed'" in translation.text
    assert "cp -a '/tmp/openswe-repo-staging/poetry.lock*' '/testbed'" in translation.text
    assert "COPY repo" not in translation.text


def test_translate_dockerfile_rejects_flagged_repo_copies():
    with pytest.raises(DockerfileRejected, match="COPY flags"):
        translate_dockerfile(
            "FROM openswe-python-3.11\nCOPY --chmod=755 repo /testbed\n",
            repo="r/x",
            base_commit="b" * 40,
            condensed_bases=CONDENSED,
        )


def test_package_files_keeps_the_upstream_bytes_and_renders_the_fail_closed_grader():
    files = convert.package_files(row(), CONDENSED)
    assert set(files) == {
        "task.toml",
        "instruction.md",
        "environment/Dockerfile",
        "tests/test.sh",
        "tests/eval_script.sh",
        "solution/gold.patch",
        "solution/solve.sh",
    }
    assert files["tests/eval_script.sh"] == row().eval_script.encode()  # verbatim authority
    assert files["instruction.md"] == row().problem_statement.encode()
    assert files["solution/gold.patch"] == row().patch.encode()
    assert files["tests/test.sh"] == grader.verifier_script()
    assert files["solution/solve.sh"] == grader.solve_script("/testbed")


def test_package_files_excludes_unconvertible_rows_with_typed_evidence():
    for overrides, reason in (
        ({"dockerfile": "   "}, "empty_dockerfile"),
        ({"eval_script": "   "}, "empty_eval_script"),
        ({"patch": ""}, "missing_gold_patch"),
        ({"eval_script": "pytest -q\n"}, "ungradeable_eval_script"),
        ({"dockerfile": "FROM openswe-python-99.9\nCOPY repo /testbed\n"}, "dockerfile_rejected"),
    ):
        with pytest.raises(convert.OpenSweRowExcluded) as raised:
            convert.package_files(row(**overrides), CONDENSED)
        assert raised.value.reason == reason
        assert raised.value.detail


def test_convert_row_produces_the_pinned_cohort_task_spec():
    context = ConversionContext({}, None)
    outcome = convert.convert_row(
        row(),
        CONDENSED,
        context,
        resolve=lambda reference: (_ for _ in ()).throw(
            AssertionError(f"the canonical row needs no registry resolution: {reference}")
        ),
    )
    task = outcome.task if hasattr(outcome, "task") else outcome
    assert isinstance(task, TaskSpec)
    assert not isinstance(task.grader, NoGrader)
    assert isinstance(task.grader, ScriptGrader)
    # the environment build carries the translated, digest-pinned Dockerfile
    build = task.environment_requirements.docker_build
    assert build is not None
    dockerfile = next(resource for resource in build.files if resource.path == "Dockerfile")
    content = dockerfile.source.content_base64
    text = base64.b64decode(content).decode()
    assert "FROM " + CONDENSED["3.11"] in text
    assert "git clone https://github.com/acme/widget.git" in text
    # the oracle carries the gold patch and its solve script
    oracle_paths = {resource.path for resource in task.resources.oracle}
    assert {"solution/gold.patch", "solution/solve.sh"} <= oracle_paths
    # execution profile: the Daytona cap the lane runs SWE waves at
    requirements = task.execution_requirements.model_dump()
    assert requirements["cpus"] == 4
    assert requirements["memory_mb"] == 8192
    assert requirements["storage_mb"] == 10240
    assert task.source.dataset == KEEP_RELEASE_REPO
    assert task.source.row == "acme__widget-1"


def test_the_condensed_base_recipe_matches_the_upstream_template():
    recipe = bases.base_dockerfile("3.11")
    assert recipe.startswith("# The condensed OpenSWE base")
    assert "FROM continuumio/miniconda3:25.3.1-1" in recipe
    assert "conda create -n testbed python=3.11 -y" in recipe
    assert 'echo "conda activate testbed" >> ~/.bashrc' in recipe
    assert "pip config set global.index-url https://pypi.org/simple/" in recipe
    assert bases.PYTHON_VERSIONS == ("2.7", *(f"3.{minor}" for minor in range(5, 15)))


def test_load_condensed_bases_requires_digest_pins_for_the_pinned_revision(tmp_path):
    good = {"upstream_revision": UPSTREAM_REVISION, "images": {"3.11": "ghcr.io/x/y@sha256:" + "a" * 64}}
    (tmp_path / bases.BINDINGS_FILENAME).write_text(json.dumps(good))
    assert bases.load_condensed_bases(str(tmp_path)) == {"3.11": "ghcr.io/x/y@sha256:" + "a" * 64}
    mutable = {"upstream_revision": UPSTREAM_REVISION, "images": {"3.11": "ghcr.io/x/y:latest"}}
    (tmp_path / bases.BINDINGS_FILENAME).write_text(json.dumps(mutable))
    with pytest.raises(ValueError, match="not a digest-pinned reference"):
        bases.load_condensed_bases(str(tmp_path))
    foreign = {"upstream_revision": "f" * 40, "images": {}}
    (tmp_path / bases.BINDINGS_FILENAME).write_text(json.dumps(foreign))
    with pytest.raises(ValueError, match="do not cover the pinned"):
        bases.load_condensed_bases(str(tmp_path))


def test_base_build_collector_records_only_built_versions(tmp_path, monkeypatch):
    """A version whose build failed leaves no digest file and stays out of the
    bindings, so the converter's rows naming it become typed exclusions."""
    script = launch_base_build.bindings_collector(("3.11", "2.7"), digest_dir=str(tmp_path))
    compile(script, "collector.py", "exec")  # the generated source must be valid Python
    (tmp_path / "python3_11-digest.txt").write_text(
        "ghcr.io/marin-community/rl-data-openswe:python-3.11@sha256:" + "a" * 64 + "\n"
    )
    monkeypatch.chdir(tmp_path)
    exec(compile(script, "collector.py", "exec"), {})
    bindings = json.loads((tmp_path / "condensed-base-bindings.json").read_text())
    assert bindings["images"] == {
        "3.11": "ghcr.io/marin-community/rl-data-openswe:python-3.11@sha256:" + "a" * 64
    }  # 2.7 built nothing and is absent
    assert bindings["upstream_revision"] == UPSTREAM_REVISION


def test_base_build_commands_pin_the_destination_and_isolate_failures():
    command = launch_base_build.builds_command(("3.11",))
    assert "--destination=ghcr.io/marin-community/rl-data-openswe:python-3.11" in command
    assert "--image-name-with-digest-file=/tmp/python3_11-digest.txt" in command
    assert "BASE_BUILD_FAILED python-3.11" in command  # a failed base does not stop the job
