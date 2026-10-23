# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Builds in a child process: the loader's test program, its model call served by ``fake_glm``."""

import asyncio
import json
import time
from dataclasses import asdict, replace

import pytest

from taskforge.ledger.jsonl import JsonlLedger, read_entries
from taskforge.ledger.records import EntryKind
from taskforge.llm.agent import ReasoningReplay
from taskforge.llm.glm_client import Pool
from taskforge.sandbox.factories import MachineHost
from taskforge.sandbox.image_builder import IrisImageBuilder, RegistryCredential
from taskforge.stage40_build_with_program.build_process import (
    LOG_FILE,
    REQUEST,
    RESULT,
    BuildModelConfig,
    BuildRequest,
    BuildServiceConfig,
    Built,
    Crashed,
    Expired,
    Failed,
    HostFailed,
    Raised,
    image_builder,
    run_build_in_process,
)
from taskforge.stage40_build_with_program.context import (
    BuildExpired,
    BuildFailure,
    BuildInfrastructureFailure,
    ExpiredClock,
)
from taskforge.stage40_build_with_program.infrastructure import InfrastructureCause
from taskforge.stage40_build_with_program.program_sdk import load_program
from taskforge.stage40_build_with_program.run import ResearchRequirement, run_build
from taskforge.stage40_build_with_program.step_store import CacheStatus
from tests.stage30_author.programs import PROGRAM as PLAIN_PROGRAM
from tests.stage40_build_with_program.conftest import LIMITS, SAMPLING, WORKPIECE

TOKEN_ENV = "TASKFORGE_TEST_GLM_TOKEN"
CONTROLLER_URL = "http://controller.example:10000"
TOKEN = "child-token"
REGISTRY_ENV = "TASKFORGE_TEST_REGISTRY_CREDENTIAL"
CREDENTIAL = RegistryCredential(registry="registry.example", user="publisher", password="registry-password")
WALL_TIMEOUT = 50.0
"""Long enough for the child to start and build; the tests expecting a kill set their own."""
SPIN_TIMEOUT = 8.0
QUESTION = '    return Question(text="What is 6 * 7?")'
ASKED = '    return await structured(context, [{"role": "user", "content": "Ask for 6 * 7."}], Question)'
PROGRAM = (
    PLAIN_PROGRAM.replace("from dataclasses import dataclass\n", "from pydantic import BaseModel\n")
    .replace("@dataclass(frozen=True)\nclass Question:", "class Question(BaseModel):")
    .replace("import ANSWER_FORMAT", "import ANSWER_FORMAT, structured")
    .replace(
        "import RLEnvBuildContext",
        "import BuildExpired, BuildFailure, BuildInfrastructureFailure, ExpiredClock, RLEnvBuildContext",
    )
    .replace(QUESTION, ASKED)
)
BUILD_INPUT_PROGRAM = PROGRAM.replace(
    "async def assemble(context: RLEnvBuildContext, question: Question, grader: Grader) -> TaskSpec:",
    "async def assemble(context: RLEnvBuildContext, question: Question, grader: Grader, "
    "build_input: object) -> TaskSpec:",
).replace("        context.workpiece_id,\n", '        build_input["task_id"],\n')
BUILD_INPUT = {"task_id": "task-from-build-input"}


def with_question(body: str) -> str:
    """``PROGRAM`` with the body of its ``question`` step replaced by ``body``."""
    assert ASKED in PROGRAM
    return PROGRAM.replace(ASKED, body)


@pytest.fixture(autouse=True)
def token(monkeypatch):
    monkeypatch.setenv(TOKEN_ENV, TOKEN)


@pytest.fixture
def make_request(tmp_path, proposal, fake_glm, fast_provider):
    def make(source: str) -> BuildRequest:
        return BuildRequest(
            workpiece_id=WORKPIECE,
            round=0,
            program_source=source,
            proposal=proposal,
            round_dir=tmp_path / "child" / "round",
            store_root=str(tmp_path / "child" / "store"),
            model=BuildModelConfig(fake_glm.base_url, Pool.HIGH, TOKEN_ENV, fast_provider),
            sampling=SAMPLING,
            research=ResearchRequirement.OFF,
            invalidate=frozenset(),
            model_digest=None,
            services=BuildServiceConfig(
                host=MachineHost.LAPTOP,
                controller_url=None,
                parallel_key_env=None,
                ledger_root=tmp_path / "child" / "ledger",
                reasoning_replay=ReasoningReplay.KEEP,
                agent_limits=LIMITS,
                image_cluster=None,
                registry_credential_env=None,
                decision=None,
            ),
        )

    return make


def on_iris(request: BuildRequest, image_cluster: str | None) -> BuildRequest:
    """``request`` for a child in an Iris task, publishing images on ``image_cluster`` when it is set."""
    services = replace(
        request.services,
        host=MachineHost.IRIS,
        controller_url=CONTROLLER_URL,
        image_cluster=image_cluster,
        registry_credential_env=None if image_cluster is None else REGISTRY_ENV,
    )
    return replace(request, services=services)


async def test_a_build_in_a_child_process_writes_the_draft_and_ledger_of_the_build_in_process(
    tmp_path, proposal, services, glm_client, fake_glm, make_request
):
    for _ in range(2):
        fake_glm.stream(content=json.dumps({"text": "What is 6 * 7?"}))
    in_process = await run_build(
        load_program(BUILD_INPUT_PROGRAM),
        proposal,
        WORKPIECE,
        tmp_path / "here" / "round",
        str(tmp_path / "here" / "store"),
        glm_client,
        SAMPLING,
        services,
        0,
        research=ResearchRequirement.OFF,
        build_input=BUILD_INPUT,
    )
    request = replace(make_request(BUILD_INPUT_PROGRAM), build_input=BUILD_INPUT)

    child = await run_build_in_process(request, wall_timeout=WALL_TIMEOUT)

    assert child == in_process
    assert child.task.id == BUILD_INPUT["task_id"]
    assert {outcome.status for outcome in child.provenance.steps} == {CacheStatus.MISS}
    assert fake_glm.request_headers[1]["authorization"] == f"Bearer {TOKEN}"
    ledger = JsonlLedger(tmp_path / "child" / "ledger")
    entries = list(read_entries(ledger.path_for(WORKPIECE)))
    steps = [entry.step for entry in entries if entry.kind == EntryKind.STEP]
    # The ledger records steps as they finish and the provenance keeps program order; independent steps overlap.
    assert sorted(steps) == sorted(outcome.memo.name for outcome in child.provenance.steps)
    assert [entry.step for entry in entries if entry.kind == EntryKind.LLM_CALL] == ["question"]


async def test_a_step_that_never_yields_expires_at_the_wall_clock_while_the_driver_loop_runs(make_request):
    ticks = 0

    async def tick() -> None:
        nonlocal ticks
        while True:
            await asyncio.sleep(0.05)
            ticks += 1

    ticker = asyncio.create_task(tick())
    request = make_request(with_question('    print("spinning", flush=True)\n    while True:\n        pass'))
    started = time.monotonic()

    with pytest.raises(BuildExpired) as expired:
        await run_build_in_process(request, wall_timeout=SPIN_TIMEOUT)

    elapsed = time.monotonic() - started
    ticker.cancel()
    assert (expired.value.clock, expired.value.step) == (ExpiredClock.WALL, None)
    assert str(expired.value) == f"the build ran for more than {SPIN_TIMEOUT:.0f} seconds"
    assert "spinning" in (request.round_dir / LOG_FILE).read_text()
    assert elapsed < SPIN_TIMEOUT + 5
    assert ticks > SPIN_TIMEOUT / 0.05 / 2


async def test_a_build_failure_in_the_child_comes_back_with_its_step(make_request):
    request = make_request(with_question('    raise BuildFailure("the question is ambiguous")'))

    with pytest.raises(BuildFailure) as failure:
        await run_build_in_process(request, wall_timeout=WALL_TIMEOUT)

    assert (failure.value.message, failure.value.step) == ("the question is ambiguous", "question")


async def test_a_host_failure_in_the_child_comes_back_with_its_cause(make_request):
    body = f'    raise BuildInfrastructureFailure("{InfrastructureCause.HOST_UNREACHABLE}", "the controller refused")'

    with pytest.raises(BuildInfrastructureFailure) as failure:
        await run_build_in_process(make_request(with_question(body)), wall_timeout=WALL_TIMEOUT)

    assert (failure.value.cause, failure.value.message) == (
        InfrastructureCause.HOST_UNREACHABLE,
        "the controller refused",
    )


async def test_a_step_clock_expiry_in_the_child_comes_back_with_its_clock_and_step(make_request):
    body = '    raise BuildExpired("ran for more than 1800 seconds without returning", ExpiredClock.STEP)'

    with pytest.raises(BuildExpired) as expired:
        await run_build_in_process(make_request(with_question(body)), wall_timeout=WALL_TIMEOUT)

    assert (expired.value.clock, expired.value.step) == (ExpiredClock.STEP, "question")
    assert str(expired.value) == "ran for more than 1800 seconds without returning"


async def test_any_other_exception_in_the_child_comes_back_as_its_type_message_and_traceback(make_request):
    with pytest.raises(RuntimeError) as failure:
        await run_build_in_process(make_request(with_question("    return 1 / 0")), wall_timeout=WALL_TIMEOUT)

    first, _, rest = str(failure.value).partition("\n\n")
    assert first == "ZeroDivisionError: division by zero"
    assert "Traceback (most recent call last)" in rest


async def test_a_child_that_cannot_start_its_services_is_a_host_failure_carrying_its_log(make_request, monkeypatch):
    monkeypatch.delenv(TOKEN_ENV)

    with pytest.raises(BuildInfrastructureFailure) as failure:
        await run_build_in_process(make_request(PROGRAM), wall_timeout=WALL_TIMEOUT)

    assert failure.value.cause == InfrastructureCause.BUILD_PROCESS
    assert f"the build reads a secret from {TOKEN_ENV}, which is not set" in failure.value.message


@pytest.mark.parametrize("image_cluster", [None, "cw-us-east-02a"])
def test_requests_and_results_round_trip_through_json(make_request, image_cluster):
    request = on_iris(make_request(PROGRAM), image_cluster)
    results = (
        Built(),
        Failed("the question is ambiguous", "question"),
        Expired(ExpiredClock.STEP, "question", "ran for more than 1800 seconds without returning"),
        HostFailed(InfrastructureCause.NO_FACTORY, "no docker"),
        Crashed(Raised.OS_ERROR, "OSError: disk full\n\nTraceback"),
    )

    assert REQUEST.validate_json(REQUEST.dump_json(request)) == request
    assert [RESULT.validate_json(RESULT.dump_json(result)) for result in results] == list(results)


def test_the_child_builds_images_with_the_credential_its_environment_holds_and_no_gcloud(make_request, monkeypatch):
    monkeypatch.setenv(REGISTRY_ENV, json.dumps(asdict(CREDENTIAL)))
    monkeypatch.setenv("PATH", "")

    builder = image_builder(on_iris(make_request(PROGRAM), "cw-us-east-02a").services)

    assert isinstance(builder, IrisImageBuilder)
    assert (builder.controller_url, builder.target_cluster, builder.credential) == (
        CONTROLLER_URL,
        "cw-us-east-02a",
        CREDENTIAL,
    )
    assert image_builder(make_request(PROGRAM).services) is None
    with pytest.raises(ValueError, match="controller URL"):
        image_builder(
            replace(make_request(PROGRAM).services, image_cluster="cw-us-east-02a", registry_credential_env=REGISTRY_ENV)
        )


def test_an_image_cluster_without_a_readable_credential_fails_before_any_build(make_request, monkeypatch):
    config = on_iris(make_request(PROGRAM), "cw-us-east-02a").services
    monkeypatch.delenv(REGISTRY_ENV, raising=False)

    with pytest.raises(ValueError, match=f"the build reads a secret from {REGISTRY_ENV}, which is not set"):
        image_builder(config)
    with pytest.raises(ValueError, match="registry_credential_env"):
        image_builder(replace(config, registry_credential_env=None))
    monkeypatch.setenv(REGISTRY_ENV, json.dumps({"registry": "registry.example", "user": "publisher"}))
    with pytest.raises(ValueError, match="not a JSON object of strings"):
        image_builder(config)
