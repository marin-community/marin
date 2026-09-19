# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Real Harbor replay trials for a retained TaskTrove JSON-schema task."""

import json
from pathlib import Path

import pytest

from taskcompendium.execution import (
    HarborExecutionConfig,
    HarborLaunchConfig,
    HarborTaskBinding,
    environment_for_requirements,
)
from taskcompendium.harbor.runner import run_trial
from taskcompendium.importers.tasktrove import read_archive
from taskcompendium.importers.tasktrove_structured import import_task
from taskcompendium.lowering import lower_to_harbor
from taskcompendium.models import AssistantFinal, Rejected, Rendering

FIXTURES = Path(__file__).parent / "fixtures/structured"

GOOD = {
    "propertyType": "singleFamilyHome",
    "budget": {"maxPrice": 500000, "downPayment": 25000},
    "location": {"city": "Austin", "neighborhood": "North Loop", "proximityToSchools": True},
    "preferredFeatures": ["backyard"],
    "mortgageStatus": {"applicationSubmitted": True, "preApproved": False, "interestRate": 6.5},
    "inspectionResults": {
        "inspectionDate": "2026-01-15",
        "issuesFound": ["Roof needs repair"],
        "recommendations": {
            "repairsNeeded": True,
            "negotiation": {"buyerRequested": "Repair credit", "sellerResponse": "Pending"},
        },
    },
    "closingTimeline": {
        "estimatedClosingDate": "2026-03-01",
        "contingencyPeriod": 15,
        "escrowProvider": "Example Escrow",
    },
}


def _task(tmp_path: Path):
    archive = read_archive((FIXTURES / "json-row-16636.tar.gz").read_bytes(), "16636", "other")
    specification = import_task(archive)
    assert not isinstance(specification, Rejected)
    binding = HarborTaskBinding(environment_for_requirements(specification.requirements))
    return lower_to_harbor(
        specification,
        (Rendering("structured", AssistantFinal()),),
        binding,
        tmp_path / "structured",
        reference_execution=HarborExecutionConfig(binding, HarborLaunchConfig("replay")),
    )


def _xml_task(tmp_path: Path):
    archive = read_archive((FIXTURES / "xml-row-16634.tar.gz").read_bytes(), "16634", "other")
    specification = import_task(archive)
    assert not isinstance(specification, Rejected)
    binding = HarborTaskBinding(environment_for_requirements(specification.requirements))
    return lower_to_harbor(
        specification,
        (Rendering("structured-xml", AssistantFinal()),),
        binding,
        tmp_path / "structured-xml",
        reference_execution=HarborExecutionConfig(binding, HarborLaunchConfig("replay")),
    )


@pytest.mark.parametrize(
    "attempt,reward",
    [
        ("good", 1.0),
        ("bad", 0.0),
        ("empty", None),
    ],
)
async def test_real_structured_source_grader_replay(tmp_path, attempt, reward):
    task = _task(tmp_path)
    response = ""
    if attempt != "empty":
        candidate = dict(GOOD)
        if attempt == "bad":
            candidate["propertyType"] = "apartment"
        response = json.dumps(candidate)
    execution = json.loads((task / "reference-execution.json").read_text())
    execution["agent"]["kwargs"] = {"response": response}

    result = await run_trial(task, execution, tmp_path / "trials", attempt)

    outcome = json.loads((tmp_path / f"trials/{attempt}/verifier/taskcompendium-result.json").read_text())
    assert outcome["reward"] == reward
    if reward is None:
        assert result.verifier_result is None
        assert outcome["status"] == "extraction_error"
    else:
        assert result.exception_info is None, result.exception_info
        assert result.verifier_result.rewards == {"reward": reward}
        assert outcome["status"] == "graded"


@pytest.mark.parametrize(
    "attempt,reward",
    [
        ("good", 1.0),
        ("bad", 0.0),
        ("empty", None),
    ],
)
async def test_real_xml_structured_source_grader_replay(tmp_path, attempt, reward):
    task = _xml_task(tmp_path)
    response = ""
    if attempt == "good":
        response = (
            "<aircraft><aircraft_name>Sikorsky S-9</aircraft_name><role>Experimental Monoplane</role>"
            "<manufacturer>Russian Baltic Railroad Car Works</manufacturer><designer>Igor Sikorsky</designer>"
            "<first_flight_year>1913</first_flight_year><number_built>1</number_built></aircraft>"
        )
    elif attempt == "bad":
        response = "<aircraft><aircraft_name>Sikorsky S-9</aircraft_name></aircraft>"
    execution = json.loads((task / "reference-execution.json").read_text())
    execution["agent"]["kwargs"] = {"response": response}

    result = await run_trial(task, execution, tmp_path / "trials", f"xml-{attempt}")

    outcome = json.loads((tmp_path / f"trials/xml-{attempt}/verifier/taskcompendium-result.json").read_text())
    assert outcome["reward"] == reward
    if reward is None:
        assert result.verifier_result is None
        assert outcome["status"] == "extraction_error"
    else:
        assert result.exception_info is None, result.exception_info
        assert result.verifier_result.rewards == {"reward": reward}
        assert outcome["status"] == "graded"
