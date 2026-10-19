"""SkyRL-owned group-grader specifications."""

import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from skyrl_gym.source_task import session_parameters
from rolloutengine.spec import LoweredTaskSpec
from taskcompendium.models import SessionGrader

from skyrl_gym.envs.nemotron_ultra import GENRM_AGENTS

GENRM_GROUP_GRADER = "nemotron_genrm"


class GroupGraderSpec(BaseModel):
    """A trainer-owned grader for a group of attempts."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    parameters_json: str


class GenRMGroupGraderParameters(BaseModel):
    """Private inputs for the Nemotron GenRM group grader."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    principle: str = Field(min_length=1)
    agent: str
    config: dict[str, Any]


def task_group_grader(lowered: LoweredTaskSpec) -> GroupGraderSpec | None:
    """Derive a group grader from the SkyRL environment payload."""
    task = lowered.task
    if lowered.session.task_session != "nemotron_ultra" or not isinstance(task.grader, SessionGrader):
        return None
    parameters = session_parameters(task)
    extras = parameters.extras
    config = parameters.config
    ultra = (extras.get("extra_info") or {}).get("nemotron_ultra") or {}
    if ultra.get("agent") not in GENRM_AGENTS or config.get("grading") == "skip":
        return None
    record = json.loads(ultra["record_json"])
    principle = record.get("principle")
    if not isinstance(principle, str) or not principle.strip():
        raise ValueError("GenRM tasks require a non-empty grading principle")
    grading = dict(config.get("genrm", {}))
    if config.get("verifyit_enabled", False):
        grading["verifyit_enabled"] = True
    return GroupGraderSpec(
        name=GENRM_GROUP_GRADER,
        parameters_json=GenRMGroupGraderParameters(
            principle=principle,
            agent=ultra["agent"],
            config=grading,
        ).model_dump_json(),
    )
