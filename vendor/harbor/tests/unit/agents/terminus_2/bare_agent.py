"""Shared factory for a bare Terminus2 carrying only trajectory-dump state.

Bypasses ``__init__`` (which constructs a real LLM backend) via
``object.__new__`` and sets the minimal attributes the trajectory persistence
path touches. Tests override attributes via keyword arguments so the single
attribute list cannot drift between test modules.
"""

import logging
import uuid

from harbor.agents.terminus_2.terminus_2 import Terminus2, TrajectoryDumpCadence
from harbor.models.agent.context import AgentContext
from harbor.models.trajectories import Step


def make_bare_terminus2(logs_dir, **overrides) -> Terminus2:
    agent = object.__new__(Terminus2)
    agent.logs_dir = logs_dir
    agent.logger = logging.getLogger("test-terminus-2")
    agent._context = AgentContext(
        n_input_tokens=10, n_output_tokens=20, n_cache_tokens=0, cost_usd=None
    )
    agent._linear_history = False
    agent._summarization_count = 0
    agent._continuation_segments_persisted = 0
    agent._parser_name = "json"
    agent._temperature = None
    agent._llm_kwargs = None
    agent.model_alias = "fake/model"
    agent._session_id = str(uuid.uuid4())
    agent._trajectory_dump_cadence = TrajectoryDumpCadence.FINAL
    agent._batch_turn_execs = True
    agent._session_alive_hint = None
    agent._chat = None
    agent._trajectory_steps = [Step(step_id=1, source="user", message="hi")]
    for name, value in overrides.items():
        setattr(agent, name, value)
    return agent
