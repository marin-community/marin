# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The prompts of the two model calls: the extraction for one rollout, and the merge of ideas."""

from experiments.post_training.capability_driven_envs.analyze_failed_runs.models import IDEAS_PER_ROLLOUT

CAPABILITY_RULES = """\
A capability is a skill that transfers to many tasks and domains, for example: obey the answer format that a task \
asks for, select tools and their arguments, stop a loop that repeats a call or a mistake, read a task and its \
constraints carefully, do a check of the work before the answer, or reach an answer without unneeded steps.

- Write each capability as a catalog entry for the skill itself. Do not put the failed tasks, their domain, \
their data, or their mistakes in it: a different step writes new tasks that train the skill, and the failed \
tasks evaluate it.
- A rollout that stopped at its turn limit or its time limit shows a gap only if it used its turns or its time on \
repeated or unneeded steps. The agent does not see the limits unless its messages state them."""

EXTRACT_PROMPT = f"""\
You analyze one failed rollout of a language-model agent to decide what the agent must learn next. A rollout is one \
attempt at a task: the task's messages, the agent's turns and tool calls, and the tool results.

Find the general capabilities, at most {IDEAS_PER_ROLLOUT}, whose absence explains the failure, or none if the \
rollout shows no gap. {CAPABILITY_RULES}

The first line of the rollout is a JSON object with its run ID, task, grade, stop reason, the operation that \
interrupted it if any, and counts of turns and tool calls. The second line is a JSON object with the tools that the \
task gave. The policy's harness can also give a shell tool and the functions of the tool providers. After it, each \
message starts a new line with its number, for example [0]."""

MERGE_PROMPT = f"""\
Each line is a capability that failed rollouts of a language-model agent lack, with the ID of the line and the \
number of rollouts that show the gap. Merge the capabilities into at most {{count}}: merge the capabilities that are \
about the same skill, and keep separate the capabilities that explain the most rollouts. Give each merged capability \
a different ID and the IDs of the lines that it holds. Each line goes in exactly one merged capability. \
{CAPABILITY_RULES}"""
