# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The OpenSWE fidelity gate: deterministic per-base sampling, the upstream arm's
verbatim bytes, fail-closed verdict reading, and the typed exclusion decisions
(ruling 2026-11-26 6:03 AM, option a)."""

import asyncio
import json

import pytest
from shellbox.image import DockerfileSource
from shellbox.machine import ExitReason, MachineSpec, NetworkPolicy, Result

from experiments.post_training.task_curation.datasets.openswe import fidelity_gate
from experiments.post_training.task_curation.datasets.openswe.dockerfile_translation import (
    dockerfile_instructions,
)
from experiments.post_training.task_curation.datasets.openswe.fidelity_gate import (
    ARM_CONDENSED,
    ARM_UPSTREAM,
    FidelityExclusion,
    TaskRecord,
    decide,
    evaluate_task,
    run_gate,
    run_verifier,
    sample_by_base,
    upstream_dockerfile,
)
from experiments.post_training.task_curation.datasets.openswe.sources import UPSTREAM_REVISION, OpenSweRow

CONDENSED = {"3.11": "ghcr.io/marin-community/rl-data-openswe@sha256:" + "a" * 64}
UPSTREAM = {"3.11": "ghcr.io/marin-community/rl-data-openswe:upstream-python-3.11@sha256:" + "c" * 64}

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


@pytest.fixture(autouse=True)
def fast_create(monkeypatch):
    monkeypatch.setattr(fidelity_gate, "CREATE_BACKOFF_SECONDS", (0.0, 0.0))


@pytest.fixture(autouse=True)
def fake_clone_context(monkeypatch):
    """The upstream context builder clones from GitHub; tests stage a local stub."""

    def clone_context(target_row, root):
        checkout = root / fidelity_gate._instance_slug(target_row.instance_id) / "repo"
        checkout.mkdir(parents=True, exist_ok=True)
        return checkout.parent

    monkeypatch.setattr(fidelity_gate, "clone_context", clone_context)


class FakeMachine:
    """One simulated task machine: records commands, serves the verifier's files."""

    def __init__(self, name: str, scenario: dict):
        self.name = name
        self.scenario = dict(scenario)
        self.commands: list[str] = []
        self.files: dict[str, bytes] = {}
        self.closed = False

    async def run(self, command) -> Result:
        joined = " ".join(command.argv)
        self.commands.append(joined)
        if joined.startswith("bash /tests/test.sh"):
            # A run after the oracle's solve produces the oracle's verifier files.
            if any(earlier.startswith("bash /solution/solve.sh") for earlier in self.commands[:-1]):
                self.scenario = {**self.scenario, **oracle_files(self.scenario)}
            for path, content in self.scenario.items():
                if path.startswith("/logs/"):
                    self.files[path] = content
            return Result(0, b"verifier ran\n", b"", False, False, ExitReason.EXITED)
        if "rev-parse" in joined:
            if self.scenario.get("git_root") is False:
                return Result(1, b"", b"not a git repository", False, False, ExitReason.EXITED)
            return Result(0, b"/testbed\n", b"", False, False, ExitReason.EXITED)
        if "reset --hard" in joined:
            if self.scenario.get("reset_fails"):
                return Result(1, b"", b"reset refused", False, False, ExitReason.EXITED)
            return Result(0, b"", b"", False, False, ExitReason.EXITED)
        if joined.startswith("bash /solution/solve.sh"):
            if self.scenario.get("solve_fails"):
                return Result(3, b"", b"error: patch does not apply", False, False, ExitReason.EXITED)
            return Result(0, b"", b"", False, False, ExitReason.EXITED)
        return Result(0, b"", b"", False, False, ExitReason.EXITED)

    async def upload(self, source, target: str) -> None:
        self.files[target] = source.read_bytes()

    async def download(self, source: str, target) -> None:
        if source not in self.files:
            raise FileNotFoundError(source)
        target.write_bytes(self.files[source])

    async def close(self) -> None:
        self.closed = True


def oracle_files(scenario: dict) -> dict:
    """The verifier files an oracle run writes: the gold patch made the tests pass."""
    if scenario.get("solve_fails"):
        return {}
    if scenario.get("oracle_stays_buggy"):
        return verdict_files(reward=0)
    return verdict_files(reward=1, exit_status="0", marker="OPENSWE_EXIT_CODE=0")


def verdict_files(reward: int, exit_status: str = "1", marker: str = "OPENSWE_EXIT_CODE=1") -> dict:
    return {
        "/logs/verifier/reward.txt": str(reward).encode(),
        "/logs/verifier/eval-exit-status.txt": exit_status.encode(),
        "/logs/verifier/eval-output.txt": f">>>>> Start Test Output\n{marker}\n>>>>> End Test Output\n".encode(),
    }


class FakeFactory:
    """Builds the fake machines the arms ask for, keyed by the context path."""

    def __init__(self, machines_by_arm: dict[str, list[FakeMachine]], fail_first: int = 0):
        self.machines_by_arm = machines_by_arm
        self.fail_first = fail_first
        self.created_specs: list[MachineSpec] = []
        self.handed_out: list[FakeMachine] = []
        self.attempts = 0

    async def create(self, spec: MachineSpec) -> FakeMachine:
        self.attempts += 1
        if self.fail_first > 0:
            self.fail_first -= 1
            raise RuntimeError("daytona create blipped")
        self.created_specs.append(spec)
        arm = ARM_UPSTREAM if "upstream" in spec.source.context.parts else ARM_CONDENSED
        queue = self.machines_by_arm[arm]
        if not queue:
            raise AssertionError(f"no fake machine staged for arm {arm}")
        machine = queue.pop(0)
        self.handed_out.append(machine)
        return machine


def noop_record(arm: str, reward: int = 0) -> fidelity_gate.ControlRecord:
    return fidelity_gate.ControlRecord(
        control="noop", reward=reward, eval_exit_status="1", protocol="verifier", machine=arm, output_tail=""
    )


def oracle_record(arm: str, reward: int = 1) -> fidelity_gate.ControlRecord:
    return fidelity_gate.ControlRecord(
        control="oracle", reward=reward, eval_exit_status="0", protocol="in_place", machine=arm, output_tail=""
    )


def one_machine_per_arm(scenario: dict) -> dict[str, list[FakeMachine]]:
    return {ARM_CONDENSED: [FakeMachine("condensed-1", scenario)], ARM_UPSTREAM: [FakeMachine("upstream-1", scenario)]}


def test_sample_by_base_is_deterministic_and_covers_every_base():
    """At least --per-base rows per base, the same sample for the same seed, and
    the unbaseable rows never sample."""
    rows = [row(instance_id=f"acme__widget-{index}") for index in range(25)]
    rows.append(row(instance_id="acme__other-1", dockerfile="FROM python:3.11\nCOPY repo /testbed\n"))
    rows.append(row(instance_id="acme__var-1", dockerfile="FROM openswe-python-${PYTHON_VERSION}\n"))
    first, skipped = sample_by_base(rows, per_base=20, seed=7)
    second, _ = sample_by_base(rows, per_base=20, seed=7)
    assert list(first) == ["3.11"]
    assert len(first["3.11"]) == 20
    assert [item.instance_id for item in first["3.11"]] == [item.instance_id for item in second["3.11"]]
    assert skipped == {"other_base": 1, "variable_from": 1}
    _, skipped_other_seed = sample_by_base(rows, per_base=20, seed=8)
    assert skipped_other_seed == skipped


def test_sample_by_base_takes_the_whole_population_below_per_base():
    rows = [row(instance_id=f"acme__widget-{index}") for index in range(7)]
    samples, _ = sample_by_base(rows, per_base=20, seed=1)
    assert len(samples["3.11"]) == 7


def test_upstream_dockerfile_binds_only_the_conda_root_froms():
    """The upstream arm keeps the COPY repo build and every other byte verbatim;
    stage re-references, scratch, and registry bases pass through untouched."""
    dockerfile = (
        "FROM openswe-python-3.11 AS build\n"
        "COPY repo /testbed\n"
        "RUN bash -lc 'pip install -e .'\n"
        "FROM build AS final\n"
        "COPY --from=build /testbed /testbed\n"
    )
    bound = upstream_dockerfile(dockerfile, UPSTREAM)
    froms = [
        instruction.argument.split()[0] for instruction in dockerfile_instructions(bound) if instruction.name == "FROM"
    ]
    assert froms[0].startswith("ghcr.io/marin-community/rl-data-openswe:upstream-python-3.11@sha256:")
    assert froms[1] == "build"  # the stage re-reference stays verbatim
    assert "COPY repo /testbed" in bound


def test_upstream_dockerfile_typed_exclusions():
    with pytest.raises(FidelityExclusion, match="unbound_upstream_base"):
        upstream_dockerfile("FROM openswe-python-3.10\n", UPSTREAM)
    with pytest.raises(FidelityExclusion, match="variable_from"):
        upstream_dockerfile("FROM openswe-python-${V}\n", UPSTREAM)


def test_decide_qualifies_only_both_arms_scoring_as_expected():
    arms = {
        ARM_CONDENSED: fidelity_gate.ArmRecord(
            ARM_CONDENSED, noop=noop_record("condensed"), oracle=oracle_record("condensed")
        ),
        ARM_UPSTREAM: fidelity_gate.ArmRecord(
            ARM_UPSTREAM, noop=noop_record("upstream"), oracle=oracle_record("upstream")
        ),
    }
    assert decide("acme__widget-1", "3.11", arms).status == "qualified"


def test_decide_typed_exclusions_carry_the_arm_and_evidence():
    good_noop = noop_record("condensed")
    good_oracle = oracle_record("condensed")
    bad_noop = fidelity_gate.ControlRecord(
        control="noop", reward=1, eval_exit_status="0", protocol="verifier", machine="upstream", output_tail="x=0"
    )
    arms = {
        ARM_CONDENSED: fidelity_gate.ArmRecord(ARM_CONDENSED, noop=good_noop, oracle=good_oracle),
        ARM_UPSTREAM: fidelity_gate.ArmRecord(ARM_UPSTREAM, noop=bad_noop, oracle=oracle_record("upstream")),
    }
    record = decide("acme__widget-1", "3.11", arms)
    assert (record.status, record.exclusion_reason) == ("excluded", "upstream_noop_mismatch")
    assert record.exclusion_detail is not None and "scored 1" in record.exclusion_detail

    failed = {
        ARM_CONDENSED: fidelity_gate.ArmRecord(ARM_CONDENSED, noop=good_noop, oracle=good_oracle),
        ARM_UPSTREAM: fidelity_gate.ArmRecord(
            ARM_UPSTREAM, failure_reason="build_failed", failure_detail="snapshot build died"
        ),
    }
    record = decide("acme__widget-1", "3.11", failed)
    assert (record.status, record.exclusion_reason) == ("excluded", "upstream_build_failed")
    assert record.exclusion_detail == "snapshot build died"

    arms = {
        ARM_CONDENSED: fidelity_gate.ArmRecord(ARM_CONDENSED, noop=good_noop, oracle=good_oracle),
        ARM_UPSTREAM: fidelity_gate.ArmRecord(
            ARM_UPSTREAM, noop=noop_record("upstream"), oracle=oracle_record("upstream", reward=0)
        ),
    }
    record = decide("acme__widget-1", "3.11", arms)
    assert (record.status, record.exclusion_reason) == ("excluded", "upstream_oracle_mismatch")


def test_run_verifier_fails_closed_on_missing_and_divergent_verdicts():
    machine = FakeMachine("m", verdict_files(0))

    async def scenario():
        record = await run_verifier(machine, control="noop", machine_name="m")
        assert (record.reward, record.eval_exit_status) == (0, "1")

    asyncio.run(scenario())

    unscored = FakeMachine("m", {})

    async def scenario_unscored():
        with pytest.raises(FidelityExclusion, match="verifier_unscored"):
            await run_verifier(unscored, control="noop", machine_name="m")

    asyncio.run(scenario_unscored())

    divergent = FakeMachine(
        "m",
        {
            "/logs/verifier/reward.txt": b"1",
            "/logs/verifier/eval-exit-status.txt": b"0",
            "/logs/verifier/eval-output.txt": b"no marker here\n",
        },
    )

    async def scenario_divergent():
        with pytest.raises(FidelityExclusion, match="reward_divergence"):
            await run_verifier(divergent, control="oracle", machine_name="m")

    asyncio.run(scenario_divergent())


def test_evaluate_task_qualifies_matching_arms_in_place(tmp_path):
    factory = FakeFactory(one_machine_per_arm(verdict_files(0)))

    async def scenario():
        return await evaluate_task(row(), CONDENSED, UPSTREAM, factory, tmp_path)

    record = asyncio.run(scenario())
    assert record.status == "qualified", record.as_dict()
    for arm in (ARM_CONDENSED, ARM_UPSTREAM):
        assert record.arms[arm].oracle.protocol == "in_place"
        assert (record.arms[arm].noop.reward, record.arms[arm].oracle.reward) == (0, 1)
    # The in-place protocol restores the bug state while keeping ignored artifacts.
    resets = [joined for machine in factory.handed_out for joined in machine.commands if "clean -fd" in joined]
    assert resets and all(not joined.rstrip().endswith("clean -fdx") for joined in resets)
    assert all(machine.closed for machine in factory.handed_out)


def test_evaluate_task_falls_back_to_a_fresh_oracle_without_git(tmp_path):
    factory = FakeFactory(
        {
            ARM_CONDENSED: [
                FakeMachine("condensed-1", {**verdict_files(0), "git_root": False}),
                FakeMachine("condensed-2", verdict_files(0)),
            ],
            ARM_UPSTREAM: [
                FakeMachine("upstream-1", {**verdict_files(0), "git_root": False}),
                FakeMachine("upstream-2", verdict_files(0)),
            ],
        }
    )

    async def scenario():
        return await evaluate_task(row(), CONDENSED, UPSTREAM, factory, tmp_path)

    record = asyncio.run(scenario())
    assert record.status == "qualified", record.as_dict()
    for arm in (ARM_CONDENSED, ARM_UPSTREAM):
        assert record.arms[arm].oracle.protocol == "fresh"
    assert len(factory.created_specs) == 4  # two machines per arm


def test_evaluate_task_falls_back_to_a_fresh_oracle_when_solve_fails_in_place(tmp_path):
    factory = FakeFactory(
        {
            ARM_CONDENSED: [
                FakeMachine("condensed-1", {**verdict_files(0), "solve_fails": True}),
                FakeMachine("condensed-2", verdict_files(0)),
            ],
            ARM_UPSTREAM: [
                FakeMachine("upstream-1", {**verdict_files(0), "solve_fails": True}),
                FakeMachine("upstream-2", verdict_files(0)),
            ],
        }
    )

    async def scenario():
        return await evaluate_task(row(), CONDENSED, UPSTREAM, factory, tmp_path)

    record = asyncio.run(scenario())
    assert record.status == "qualified", record.as_dict()
    for arm in (ARM_CONDENSED, ARM_UPSTREAM):
        assert record.arms[arm].oracle.protocol == "fresh"


def test_evaluate_task_reports_the_converter_rejection(tmp_path):
    factory = FakeFactory({ARM_CONDENSED: [], ARM_UPSTREAM: []})

    async def scenario():
        return await evaluate_task(row(dockerfile=""), CONDENSED, UPSTREAM, factory, tmp_path)

    record = asyncio.run(scenario())
    assert (record.status, record.exclusion_reason) == ("excluded", "converter_empty_dockerfile")
    assert not factory.created_specs


def test_evaluate_task_types_a_build_failure_per_arm(tmp_path):
    factory = FakeFactory(
        one_machine_per_arm(verdict_files(0)),
        fail_first=fidelity_gate.CREATE_ATTEMPTS,
    )

    async def scenario():
        return await evaluate_task(row(), CONDENSED, UPSTREAM, factory, tmp_path)

    record = asyncio.run(scenario())
    assert (record.status, record.exclusion_reason) == ("excluded", "condensed_build_failed")
    # Both arms attempt: the condensed arm exhausts its retries, the upstream arm runs.
    assert factory.attempts == fidelity_gate.CREATE_ATTEMPTS + 1


def test_evaluate_task_types_a_noop_mismatch_with_output_evidence(tmp_path):
    factory = FakeFactory(
        {
            ARM_CONDENSED: [FakeMachine("condensed-1", verdict_files(1, exit_status="0", marker="OPENSWE_EXIT_CODE=0"))],
            ARM_UPSTREAM: [FakeMachine("upstream-1", verdict_files(0))],
        }
    )

    async def scenario():
        return await evaluate_task(row(), CONDENSED, UPSTREAM, factory, tmp_path)

    record = asyncio.run(scenario())
    assert (record.status, record.exclusion_reason) == ("excluded", "condensed_noop_mismatch")
    assert record.exclusion_detail is not None and "scored 1" in record.exclusion_detail


def test_evaluate_task_types_an_oracle_mismatch_when_the_patch_does_not_help(tmp_path):
    factory = FakeFactory(
        {
            ARM_CONDENSED: [FakeMachine("condensed-1", {**verdict_files(0), "oracle_stays_buggy": True})],
            ARM_UPSTREAM: [FakeMachine("upstream-1", verdict_files(0))],
        }
    )

    async def scenario():
        return await evaluate_task(row(), CONDENSED, UPSTREAM, factory, tmp_path)

    record = asyncio.run(scenario())
    assert (record.status, record.exclusion_reason) == ("excluded", "condensed_oracle_mismatch")


def test_run_gate_writes_rows_and_fails_closed(tmp_path):
    factory = FakeFactory(one_machine_per_arm(verdict_files(0)))

    async def scenario():
        return await run_gate(
            {"3.11": [row(instance_id="acme__widget-1")]},
            CONDENSED,
            UPSTREAM,
            factory,
            report_dir=tmp_path,
            concurrency=2,
        )

    summary = asyncio.run(scenario())
    assert summary["state"] == "ok"
    assert summary["qualified"] == 1
    rows = [json.loads(line) for line in (tmp_path / "rows.jsonl").read_text().splitlines()]
    assert rows[0]["status"] == "qualified"
    assert (tmp_path / "fidelity-gate-result.json").exists()

    factory = FakeFactory({ARM_CONDENSED: [], ARM_UPSTREAM: []})

    async def failing():
        return await run_gate(
            {"3.11": [row(instance_id="acme__widget-2", dockerfile="")]},
            CONDENSED,
            UPSTREAM,
            factory,
            report_dir=tmp_path / "failing",
        )

    summary = asyncio.run(failing())
    assert summary["state"] == "mismatched"
    assert summary["exclusions_by_reason"] == {"converter_empty_dockerfile": 1}


def test_task_record_shape():
    record = TaskRecord("acme__widget-1", "3.11", "qualified", None, None, {})
    payload = record.as_dict()
    assert payload["instance_id"] == "acme__widget-1"
    assert payload["arms"] == {}


def test_bindings_loader_rejects_unpinned_references(tmp_path):
    manifest = {"upstream_revision": UPSTREAM_REVISION, "miniconda_pin": "continuumio/miniconda3:25.3.1-1", "images": {}}
    path = tmp_path / "upstream-base-bindings.json"
    path.write_text(json.dumps(manifest))
    assert fidelity_gate.bases.load_bindings(tmp_path, "upstream-base-bindings.json") == {}
    manifest["images"]["3.11"] = "ghcr.io/x:tag"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="digest-pinned"):
        fidelity_gate.bases.load_bindings(tmp_path, "upstream-base-bindings.json")


def test_machine_spec_uses_the_profile_declarations(tmp_path):
    context = tmp_path / "condensed" / "acme__widget-1"
    context.mkdir(parents=True)
    (context / "Dockerfile").write_text("FROM x\n")
    spec = MachineSpec(
        source=DockerfileSource(context, context / "Dockerfile"),
        workdir="/testbed",
        network=NetworkPolicy.ALLOW,
        cpus=fidelity_gate.convert.CPUS,
        memory_mb=fidelity_gate.convert.MEMORY_MB,
        storage_mb=fidelity_gate.convert.STORAGE_MB,
        startup_timeout=fidelity_gate.convert.BUILD_TIMEOUT_SEC,
    )
    assert (spec.cpus, spec.memory_mb, spec.storage_mb) == (4, 8192, 10240)
    assert spec.startup_timeout == 3600.0
