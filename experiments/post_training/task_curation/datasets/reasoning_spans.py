# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Reasoning-span prefixes mined from Marin pi evaluation trajectories (#10400).

Each task is the conversation before an assistant reply in a bixbench-pi or bfclparity-pi trajectory,
with the tools pi advertised as final tools. The model answers with one turn and scores one when its
reasoning markup is well formed. The hazard split precedes replies that had a malformed span in the
source run; the control split precedes the nearest well-formed replies of the same trajectories.
``experiments/post_training/reasoning_spans/mine_pi_prefixes.py`` mines the rows and
``upload_prefix_rows.py`` publishes them to the pinned repositories.
"""

from taskcompendium.convert.pi_trajectory import prefix_row_task
from taskcompendium.models import AssistantToolCalls, ConversationToolCall, TaskSpec
from taskcompendium.pipeline.inputs import SourceFormat
from taskcompendium.pipeline.models import Controls, IntendedUse, Reply

from experiments.post_training.task_curation.pipeline import CurationRecipe, HfSource, process_rows
from experiments.post_training.task_curation.source import RlDataSource, SourceInfo

HAZARD_REPOSITORY = "open-athena/Marin-Reasoning-Span-Hazard-Prefixes"
HAZARD_REVISION = "f6ab8adc4ed03f2f779d0bb9c294fb1279700d2c"
CONTROL_REPOSITORY = "open-athena/Marin-Reasoning-Span-Control-Prefixes"
CONTROL_REVISION = "f880546422e71d236be913cfcd4443c71b69a3a0"
ROWS_FILE = "rows.parquet"
FAMILY = "reasoning_markup"
WELL_FORMED_REPLY = "<|start_think|>Check the workspace before acting.<|end_think|>"


def well_formed_reply(task: TaskSpec) -> Reply:
    """A reply with one closed reasoning span and a call to the task's first final tool; it must score one."""
    tool = task.final_tools[0]
    required = tool.parameters.get("required") or ()
    arguments = {name: "" for name in required} if isinstance(required, list) else {}
    return Reply(
        AssistantToolCalls(
            calls=(ConversationToolCall(call_id="golden", name=tool.name, arguments=arguments),),
            content=WELL_FORMED_REPLY,
        )
    )


def _source(split: str, repository: str, revision: str, count: int | None) -> RlDataSource[CurationRecipe]:
    return RlDataSource(
        info=SourceInfo(
            id=f"Marin:reasoning-span-{split}",
            title=f"Marin pi reasoning-span prefixes · {split}",
            origin="Marin",
            family=FAMILY,
            tags=("eval", "single-turn", "tool-calling", "reasoning-markup", "pi", "bixbench", "bfcl"),
            count=count,
            notes="Prefixes stop at pi's first compaction and 12,288 prompt tokens; graded in process.",
        ),
        config=CurationRecipe(
            name=f"reasoning-span-{split}",
            version="1",
            source=HfSource(repository, revision, (ROWS_FILE,), SourceFormat.PARQUET),
            convert=prefix_row_task,
            intended_use=IntendedUse.EVAL,
            controls=Controls(golden=well_formed_reply),
        ),
        pipeline=process_rows,
    )


def sources() -> list[RlDataSource[CurationRecipe]]:
    return [
        _source("hazard", HAZARD_REPOSITORY, HAZARD_REVISION, 318),
        _source("control", CONTROL_REPOSITORY, CONTROL_REVISION, 276),
    ]
