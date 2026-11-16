# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Thoughtcrime: double-think prefixes mined from Marin pi evaluation trajectories (#10400).

Each task is the conversation before an assistant reply in a bixbench-pi or bfclparity-pi trajectory
where the recorded model closed its reasoning span a second time or opened a second span,
with the tools pi advertised as final tools. The model answers with one turn and scores one when its
reasoning markup is well formed. The rows, one per benchmark task, were mined and uploaded by the code
on marin-logbooks branch ``codex/25-thoughtcrime`` (marin-community/marin-logbooks#25).
"""

from taskcompendium.convert.pi_trajectory import prefix_row_task
from taskcompendium.models import AssistantToolCalls, ConversationToolCall, TaskSpec
from taskcompendium.pipeline.inputs import SourceFormat
from taskcompendium.pipeline.models import Controls, IntendedUse, Reply

from experiments.post_training.task_curation.pipeline import CurationRecipe, HfSource, process_rows
from experiments.post_training.task_curation.source import RlDataSource, SourceInfo

REPOSITORY = "open-athena/Marin-Thoughtcrime"
REVISION = "231751144c48b5553d5b921054b196e5e2d073f9"
ROWS_FILE = "rows.parquet"
SOURCE_ID = "Marin:thoughtcrime"
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


def sources() -> list[RlDataSource[CurationRecipe]]:
    return [
        RlDataSource(
            info=SourceInfo(
                id=SOURCE_ID,
                title="Marin thoughtcrime: pi double-think prefixes",
                origin="Marin",
                family=FAMILY,
                tags=("eval", "single-turn", "tool-calling", "reasoning-markup", "pi", "bixbench", "bfcl"),
                count=328,
                notes="One prefix per benchmark task; prefixes stop at pi's first compaction and 12,288 prompt tokens.",
            ),
            config=CurationRecipe(
                name="thoughtcrime",
                version="1",
                source=HfSource(REPOSITORY, REVISION, (ROWS_FILE,), SourceFormat.PARQUET),
                convert=prefix_row_task,
                intended_use=IntendedUse.EVAL,
                controls=Controls(golden=well_formed_reply),
            ),
            pipeline=process_rows,
        )
    ]
