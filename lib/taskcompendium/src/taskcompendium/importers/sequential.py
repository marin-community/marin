# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""A bounded sequential feature example with cumulative final requirements."""

from tasktrove_verify.spec import Mode

from taskcompendium.models import (
    AnswerRequirements,
    Capability,
    ContainerRuntime,
    ContextRequirement,
    Embedded,
    Resource,
    ResourceRole,
    Source,
    StepSpecification,
    TaskMetadata,
    TaskRequirements,
    TaskSpec,
    TaskSuccessPolicy,
    TaskTroveVerifier,
    WorkspaceState,
)


def greeting_task(image: str) -> TaskSpec:
    """Request an initial greeting API, then extend it without breaking callers."""
    initial_test = b"""import sys
sys.path.insert(0, '/app')
from greeting import greet

def test_greeting():
    assert greet('Ada') == 'Hello, Ada!'
    assert greet('Lin') == 'Hello, Lin!'
"""
    final_test = (
        initial_test
        + b"""
def test_uppercase():
    assert greet('Ada', uppercase=True) == 'HELLO, ADA!'
    assert greet('Lin', uppercase=False) == 'Hello, Lin!'
"""
    )
    initial_solution = b"def greet(name):\n    return f'Hello, {name}!'\n"
    final_solution = (
        b"def greet(name, uppercase=False):\n    text = f'Hello, {name}!'\n"
        b"    return text.upper() if uppercase else text\n"
    )
    instructions = (
        "Implement greet(name) in /app/greeting.py. Return 'Hello, <name>!' with the supplied name.",
        "Extend greet in /app/greeting.py with an optional uppercase=False argument. "
        "When true, uppercase the entire greeting. Preserve existing behavior for callers that omit it.",
    )
    return TaskSpec(
        id="synthetic/sequential-greeting",
        requirements=TaskRequirements(
            (Capability.FILESYSTEM, Capability.SHELL, Capability.PROCESS), WorkspaceState(image)
        ),
        resources=(Resource("greeting.py", (ResourceRole.AGENT,), Embedded(b"")),),
        metadata=TaskMetadata(
            Source("taskcompendium/synthetic", "v1", "sequential-greeting", "v0.3"),
            competencies=("software-engineering",),
            task_shape="sequential-requirements",
        ),
        success_policy=TaskSuccessPolicy.FINAL,
        steps=tuple(
            StepSpecification(
                instructions=instruction,
                verifier=TaskTroveVerifier(
                    Mode.PYTEST,
                    {"paths": ["/tests/test_greeting.py"], "python": "python3"},
                    runtime=ContainerRuntime(image),
                ),
                context_requirement=ContextRequirement.INSTRUCTION_AND_WORKSPACE,
                answer_requirements=AnswerRequirements("final_state"),
                resources=(
                    Resource("test_greeting.py", (ResourceRole.VERIFIER,), Embedded(test)),
                    Resource("greeting.py", (ResourceRole.ORACLE,), Embedded(solution)),
                ),
            )
            for instruction, test, solution in zip(
                instructions, (initial_test, final_test), (initial_solution, final_solution), strict=True
            )
        ),
    )


def sentence_revision_task() -> TaskSpec:
    """Revise a previous answer whose content is available only in conversation."""
    return TaskSpec(
        id="synthetic/conversational-revision",
        requirements=TaskRequirements(),
        resources=(),
        metadata=TaskMetadata(
            Source("taskcompendium/synthetic", "v1", "conversational-revision", "v0.3"),
            competencies=("instruction-following", "conversation-revision"),
            task_shape="sequential-requirements",
        ),
        success_policy=TaskSuccessPolicy.FINAL,
        steps=(
            StepSpecification(
                instructions="Use this sentence as your answer, preserving its wording and punctuation: "
                "Mira will meet Leo on Tuesday.",
                verifier=TaskTroveVerifier(Mode.EXACT, {"expected": ["Mira will meet Leo on Tuesday."]}),
            ),
            StepSpecification(
                instructions="Revise your previous sentence so the meeting is on Thursday. "
                "Keep every other word and the punctuation unchanged.",
                context_requirement=ContextRequirement.PRIOR_CONVERSATION,
                verifier=TaskTroveVerifier(Mode.EXACT, {"expected": ["Mira will meet Leo on Thursday."]}),
            ),
        ),
    )
