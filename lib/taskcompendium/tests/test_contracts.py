# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Semantic contracts survive storage, rendering, and submission changes."""

import json

import msgspec
import pyarrow.parquet as pq
import pytest
from tasktrove_verify.spec import Mode

from taskcompendium.execution import (
    DockerEnvironment,
    HarborExecutionConfig,
    HarborLaunchConfig,
    HarborTaskBinding,
    HarnessToolBinding,
    NoEnvironment,
    ShellSimEnvironment,
    ShellToolBinding,
    validate_requirements,
)
from taskcompendium.extraction import ExtractionError, extract
from taskcompendium.grading import grade_attempt
from taskcompendium.lowering import lower_to_harbor, resolve_harbor_execution, validate_lowering
from taskcompendium.models import (
    AnswerRequirements,
    AssistantFinal,
    BoxedLatex,
    Capability,
    Embedded,
    FileSubmission,
    FinalActionSubmission,
    JsonPath,
    NativeFunction,
    Outcome,
    PlainText,
    Rendering,
    Resource,
    ResourceRef,
    ResourceRole,
    Source,
    StepSpecification,
    TaskMetadata,
    TaskRequirements,
    TaskSpec,
    TaskSuccessPolicy,
    TaskTroveVerifier,
    WorkspaceState,
    XmlPath,
)
from taskcompendium.rendering import render_instruction, result_tags
from taskcompendium.resources import materialize
from taskcompendium.serialization import from_json, read_parquet, specification_hash, to_json, write_parquet


@pytest.fixture
def math_task():
    return TaskSpec(
        id="unit/math",
        requirements=TaskRequirements(),
        resources=(),
        metadata=TaskMetadata(Source("unit", "v1", "0", "v1")),
        steps=(
            StepSpecification(
                instructions="Compute 1/2 + 1/4.",
                verifier=TaskTroveVerifier(Mode.MATH, {"expected": "3/4"}),
            ),
        ),
    )


@pytest.mark.parametrize(
    "extractor,good,bad",
    [
        (PlainText(), "3/4", "4/3"),
        (BoxedLatex(), r"Reasoning. \boxed{\frac{3}{4}}", r"\boxed{\frac{4}{3}}"),
        (JsonPath(), '{"answer":"3/4"}', '{"answer":"4/3"}'),
        (XmlPath(), "<answer>3/4</answer>", "<answer>4/3</answer>"),
    ],
)
def test_equivalent_submissions_keep_same_semantic_grading(math_task, tmp_path, extractor, good, bad):
    protocol = Rendering("answer", AssistantFinal(extractor))
    success = grade_attempt(math_task, protocol, good, tmp_path)
    failure = grade_attempt(math_task, protocol, bad, tmp_path)
    empty = grade_attempt(math_task, protocol, "", tmp_path)
    assert (success.status, success.reward) == (Outcome.GRADED, 1.0)
    assert (failure.status, failure.reward) == (Outcome.GRADED, 0.0)
    assert (empty.status, empty.reward) == (Outcome.EXTRACTION_ERROR, None)


def test_file_submission_reuses_math_verifier(math_task, tmp_path):
    protocol = Rendering("file", FileSubmission("/app/answer.txt", JsonPath()))
    (tmp_path / "answer.txt").write_text('{"answer":"3/4"}')
    assert grade_attempt(math_task, protocol, None, tmp_path).reward == 1.0
    (tmp_path / "answer.txt").write_text('{"answer":"4/3"}')
    assert grade_attempt(math_task, protocol, None, tmp_path).reward == 0.0


@pytest.mark.parametrize(
    "extractor,text",
    [
        (JsonPath(), '{"answer":"A", "answer":"B"}'),
        (JsonPath(), '{"answer":null}'),
        (JsonPath(), '{"answer":NaN}'),
        (BoxedLatex(), r"\boxed{A} then \boxed{B}"),
        (BoxedLatex(), r"\boxed{\frac{1}{2}"),
        (XmlPath("/root/answer"), "<root><answer>A</answer><answer>B</answer></root>"),
        (XmlPath(), '<!DOCTYPE answer [<!ENTITY gold "A">]><answer>&gold;</answer>'),
    ],
)
def test_ambiguous_and_malformed_submissions_do_not_recover_answers(extractor, text):
    with pytest.raises(ExtractionError):
        extract(text, extractor)


def test_mixed_task_records_roundtrip_across_parquet_batches(math_task, tmp_path):
    reference = tmp_path / "input.txt"
    reference.write_text("public input")
    other = msgspec.structs.replace(
        math_task,
        id="unit/structured",
        requirements=TaskRequirements((Capability.FILESYSTEM, Capability.SHELL), WorkspaceState()),
        resources=(
            Resource("input.txt", (ResourceRole.AGENT,), Embedded(b"public input")),
            Resource("schema.json", (ResourceRole.VERIFIER,), Embedded(b'{"type":"object"}')),
        ),
        steps=(
            msgspec.structs.replace(
                math_task.steps[0],
                verifier=TaskTroveVerifier(Mode.JSON_SCHEMA, {"schema": "schema.json"}),
                answer_requirements=AnswerRequirements("json"),
            ),
        ),
    )
    uri = str(tmp_path / "dataset.parquet")
    assert write_parquet([math_task, other], uri, batch_size=1) == 2
    restored = list(read_parquet(uri))
    assert restored == [math_task, other]
    assert [specification_hash(s) for s in restored] == [specification_hash(s) for s in (math_task, other)]
    assert from_json(to_json(other)) == other
    table = pq.read_table(uri, columns=["id", "requirements"])
    assert [r["requirements"]["capabilities"] for r in table.to_pylist()] == [[], ["filesystem", "shell"]]


def test_format_is_semantic_and_cannot_be_replaced_by_wrapper(math_task):
    spec = msgspec.structs.replace(
        math_task, steps=(msgspec.structs.replace(math_task.steps[0], answer_requirements=AnswerRequirements("json")),)
    )
    with pytest.raises(ValueError, match="Intrinsic"):
        validate_lowering(
            spec,
            Rendering("xml", AssistantFinal(XmlPath())),
            HarborTaskBinding(NoEnvironment()),
        )
    with pytest.raises(ValueError, match="empty tool list"):
        validate_lowering(
            math_task,
            Rendering("file", FileSubmission("/app/a")),
            HarborTaskBinding(NoEnvironment()),
        )


def test_lowering_result_tags_add_output_encoding(math_task):
    specification = msgspec.structs.replace(
        math_task,
        coverage_tags=("competency:quantitative_reasoning", "shape:answer"),
        difficulty=2,
    )
    json_rendering = Rendering("json", AssistantFinal(JsonPath()))
    assert tuple(sorted((*specification.coverage_tags, *result_tags((json_rendering,))))) == (
        "competency:quantitative_reasoning",
        "result:json",
        "shape:answer",
    )
    file_rendering = Rendering("file", FileSubmission("/app/answer.json", JsonPath()))
    assert tuple(sorted((*specification.coverage_tags, *result_tags((file_rendering,))))) == (
        "competency:quantitative_reasoning",
        "result:file",
        "result:json",
        "shape:answer",
    )


def test_coverage_tags_reject_rendering_tags_on_semantic_specification(math_task):
    with pytest.raises(ValueError, match="Unsupported coverage tag"):
        msgspec.structs.replace(math_task, coverage_tags=("result:json",))


def test_coverage_tags_use_subject_namespace(math_task):
    specification = msgspec.structs.replace(
        math_task,
        coverage_tags=(
            "competency:quantitative_reasoning",
            "shape:answer",
            "subject:math.calculus.integration",
        ),
    )
    assert specification.coverage_tags[-1] == "subject:math.calculus.integration"
    with pytest.raises(ValueError, match="Unsupported coverage tag"):
        msgspec.structs.replace(math_task, coverage_tags=("domain:calculus.integration",))
    with pytest.raises(ValueError, match="Unsupported subject root"):
        msgspec.structs.replace(math_task, coverage_tags=("subject:email",))


def test_coverage_tags_accept_mime_types_for_artifact_and_context(math_task):
    specification = msgspec.structs.replace(
        math_task,
        coverage_tags=(
            "artifact:application/vnd.example+json",
            "competency:quantitative_reasoning",
            "context:application/json",
            "shape:answer",
        ),
    )
    assert specification.coverage_tags[0] == "artifact:application/vnd.example+json"
    with pytest.raises(ValueError, match="Unsupported coverage tag"):
        msgspec.structs.replace(math_task, coverage_tags=("subject:application/json",))


def test_difficulty_is_a_numeric_specification_field(math_task, tmp_path):
    specification = msgspec.structs.replace(math_task, difficulty=7)
    path = str(tmp_path / "difficulty.parquet")
    write_parquet([specification], path)
    restored = next(read_parquet(path))
    assert restored.difficulty == 7
    assert from_json(to_json(restored)) == specification
    for value in (0, 11, True, 3.5):
        with pytest.raises(ValueError, match="Difficulty must be an integer"):
            msgspec.structs.replace(math_task, difficulty=value)
    with pytest.raises(ValueError, match="Unsupported coverage tag"):
        msgspec.structs.replace(math_task, coverage_tags=("difficulty:hard",))


def test_export_materializes_only_agent_projection(math_task, tmp_path):
    spec = msgspec.structs.replace(
        math_task,
        resources=(
            Resource("visible.txt", (ResourceRole.AGENT,), Embedded(b"public")),
            Resource("gold.txt", (ResourceRole.VERIFIER,), Embedded(b"private gold")),
            Resource("solve.sh", (ResourceRole.ORACLE,), Embedded(b"secret oracle")),
        ),
    )
    task = lower_to_harbor(
        spec,
        (Rendering("file", FileSubmission("/app/answer.txt")),),
        HarborTaskBinding(
            ShellSimEnvironment(),
            (ShellToolBinding("shell", "shellsim"),),
        ),
        tmp_path / "export",
    )
    assert (task / "environment/inputs/visible.txt").read_bytes() == b"public"
    assert list((task / "environment/inputs").iterdir()) == [task / "environment/inputs/visible.txt"]
    assert "private gold" not in (task / "instruction.md").read_text()
    manifest = json.loads((task / "manifest.json").read_text())
    assert manifest["specification_sha256"] == specification_hash(spec)
    assert from_json((task / "specification.json").read_bytes()) == spec


def test_materialization_rejects_symlink_escape_and_modified_reference(math_task, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    root = tmp_path / "workspace"
    root.mkdir()
    (root / "escape").symlink_to(outside)
    spec = msgspec.structs.replace(
        math_task, resources=(Resource("escape/gold", (ResourceRole.VERIFIER,), Embedded(b"secret")),)
    )
    with pytest.raises(ValueError, match="escapes"):
        materialize(spec, ResourceRole.VERIFIER, root)
    assert not (outside / "gold").exists()
    original = tmp_path / "reference"
    original.write_text("changed")
    spec = msgspec.structs.replace(
        math_task, resources=(Resource("r", (ResourceRole.VERIFIER,), ResourceRef(str(original), "0" * 64)),)
    )
    with pytest.raises(ValueError, match="digest mismatch"):
        materialize(spec, ResourceRole.VERIFIER, root)


@pytest.mark.parametrize("candidate", [r"wrong } \boxed{3/4}", r"\boxed{4/3} then \boxed{3/4}"])
def test_plain_math_does_not_reextract_an_embedded_answer(math_task, tmp_path, candidate):
    result = grade_attempt(math_task, Rendering("plain", AssistantFinal()), candidate, tmp_path)
    assert result.status == Outcome.EXTRACTION_ERROR
    assert result.reward is None


def test_multistep_parquet_keeps_order_and_distinct_private_resources(math_task, tmp_path):
    first = msgspec.structs.replace(
        math_task.steps[0], resources=(Resource("reference.txt", (ResourceRole.VERIFIER,), Embedded(b"first")),)
    )
    second = msgspec.structs.replace(
        first,
        instructions="Compute two plus two.",
        verifier=TaskTroveVerifier(Mode.EXACT, {"expected": ["4"]}),
        resources=(Resource("reference.txt", (ResourceRole.VERIFIER,), Embedded(b"second")),),
    )
    spec = msgspec.structs.replace(math_task, steps=(first, second), success_policy=TaskSuccessPolicy.FINAL)
    uri = str(tmp_path / "multi.parquet")
    write_parquet([spec], uri)
    restored = next(read_parquet(uri))
    assert restored == spec
    assert specification_hash(restored) == specification_hash(spec)
    assert grade_attempt(restored, Rendering("plain", AssistantFinal()), "4", tmp_path, step_index=1).reward == 1.0
    materialize(restored, ResourceRole.VERIFIER, tmp_path / "second", step_index=1)
    assert (tmp_path / "second/reference.txt").read_bytes() == b"second"


def test_exact_verifier_without_runtime_survives_storage_and_grades(math_task, tmp_path):
    spec = msgspec.structs.replace(
        math_task,
        steps=(
            StepSpecification(instructions="Return A.", verifier=TaskTroveVerifier(Mode.EXACT, {"expected": ["A"]})),
        ),
    )
    document = json.loads(to_json(spec))
    assert "verifier_runtime" not in document["steps"][0]
    assert "runtime" not in document["steps"][0]["verifier"]
    path = str(tmp_path / "tasks.parquet")
    write_parquet([from_json(json.dumps(document))], path)
    restored = next(iter(read_parquet(path)))
    rendering = Rendering("json", AssistantFinal(JsonPath()))
    assert grade_attempt(restored, rendering, '{"answer":"A"}', tmp_path).reward == 1.0
    assert grade_attempt(restored, rendering, '{"answer":"B"}', tmp_path).reward == 0.0


@pytest.mark.parametrize("mode", ["stdio", "pytest", "script", "junit", "gotest"])
def test_serialized_executable_verifier_requires_isolation(math_task, mode):
    document = json.loads(to_json(math_task))
    document["steps"][0]["verifier"] = {"kind": "tasktrove", "mode": mode, "parameters": {}}
    with pytest.raises(msgspec.ValidationError, match="isolated container runtime"):
        from_json(json.dumps(document))


def test_serialized_exact_verifier_rejects_execution_runtime(math_task):
    document = json.loads(to_json(math_task))
    document["steps"][0]["verifier"] = {
        "kind": "tasktrove",
        "mode": "exact",
        "parameters": {"expected": ["A"]},
        "runtime": {"kind": "container", "image": "sha256:" + "1" * 64},
    }
    with pytest.raises(msgspec.ValidationError, match="Only executable verifiers"):
        from_json(json.dumps(document))


@pytest.mark.parametrize(
    "environment,tools",
    [
        (
            {"kind": "shellsim"},
            [{"kind": "shell", "name": "shell", "backend": "docker"}],
        ),
        (
            {"kind": "docker", "image": "sha256:" + "1" * 64},
            [{"kind": "harness", "interface": "mini-swe-agent", "backend": "docker"}],
        ),
        (
            {"kind": "none"},
            [{"kind": "shell", "name": "shell", "backend": "shellsim"}],
        ),
    ],
)
def test_binding_json_rejects_unimplementable_tool_bindings(environment, tools):
    with pytest.raises((msgspec.ValidationError, ValueError)):
        msgspec.json.decode(json.dumps({"environment": environment, "tools": tools}), type=HarborTaskBinding)


def test_binding_json_defaults_to_an_empty_tool_list():
    assert msgspec.json.decode('{"environment":{"kind":"none"}}', type=HarborTaskBinding) == HarborTaskBinding(
        NoEnvironment()
    )


def test_terminal_binding_accepts_multiple_harbor_agents_without_selecting_one():
    environment = DockerEnvironment("sha256:" + "1" * 64)
    binding = HarborTaskBinding(
        environment,
        (HarnessToolBinding("terminal", "docker"),),
    )
    for agent in ("replay", "terminus-2", "mini-swe-agent"):
        resolved = resolve_harbor_execution(
            (Rendering("workspace", FileSubmission("/app/answer.txt")),),
            HarborExecutionConfig(binding, HarborLaunchConfig(agent)),
            {"import_path": "taskcompendium.harbor.environments:TaskDockerEnvironment"},
        )
        assert resolved["environment"]["import_path"] == "taskcompendium.harbor.environments:TaskDockerEnvironment"
    with pytest.raises(ValueError, match="compatible terminal agent"):
        HarborExecutionConfig(binding, HarborLaunchConfig("tool_chat"))


def test_export_rejects_tool_binding_override_before_writing(math_task, tmp_path):
    destination = tmp_path / "invalid"
    with pytest.raises(ValueError, match="Harbor task binding"):
        lower_to_harbor(
            math_task,
            (Rendering("plain", AssistantFinal()),),
            HarborTaskBinding(NoEnvironment()),
            destination,
            agent_kwargs={"tool_binding": {"kind": "harness", "interface": "terminal", "backend": "docker"}},
        )
    assert not destination.exists()


def test_lowering_keeps_private_data_out_of_agent_projection(math_task, tmp_path):
    spec = msgspec.structs.replace(
        math_task,
        resources=(Resource("oracle.txt", (ResourceRole.ORACLE,), Embedded(b"secret rationale")),),
    )
    renderings = (Rendering("plain", AssistantFinal()),)
    binding = HarborTaskBinding(NoEnvironment())
    for agent in ("chat", "replay"):
        path = lower_to_harbor(
            spec,
            renderings,
            binding,
            tmp_path / agent,
            reference_execution=HarborExecutionConfig(binding, HarborLaunchConfig(agent)),
        )
        assert not (path / "task.json").exists()
        assert (path / "instruction.md").read_text() == render_instruction(spec, renderings[0])
        assert (path / "binding.json").is_file()
        assert not (path / "environment/inputs").exists()
        assert not (path / "execution.json").exists()
        assert "secret rationale" not in (path / "instruction.md").read_text()
        assert json.loads((path / "manifest.json").read_text())["specification_sha256"] == specification_hash(spec)
    assert grade_attempt(spec, renderings[0], "3/4", tmp_path).reward == 1.0


def test_lowering_rejects_changes_to_intrinsic_submission(math_task):
    state = msgspec.structs.replace(
        math_task,
        steps=(msgspec.structs.replace(math_task.steps[0], answer_requirements=AnswerRequirements("final_state")),),
    )
    with pytest.raises(ValueError, match="final-state"):
        validate_lowering(
            state,
            Rendering("plain", AssistantFinal()),
            HarborTaskBinding(ShellSimEnvironment(), (ShellToolBinding("shell", "shellsim"),)),
        )
    literal = msgspec.structs.replace(
        math_task,
        steps=(msgspec.structs.replace(math_task.steps[0], answer_requirements=AnswerRequirements("literal")),),
    )
    with pytest.raises(ValueError, match="Intrinsic"):
        validate_lowering(literal, Rendering("json", AssistantFinal(JsonPath())), HarborTaskBinding(NoEnvironment()))


def test_provider_matching_preserves_capabilities_and_pinned_state():
    image = "sha256:" + "a" * 64
    state = WorkspaceState(image=image, workdir="/repo", setup_commands=("touch /repo/ready",))
    required = TaskRequirements((Capability.FILESYSTEM, Capability.SHELL, Capability.PROCESS), state)
    with pytest.raises(ValueError, match="capabilities"):
        validate_requirements(required, ShellSimEnvironment(workdir="/repo"))
    with pytest.raises(ValueError, match="initial state"):
        validate_requirements(required, DockerEnvironment("sha256:" + "b" * 64, workdir="/repo"))
    with pytest.raises(ValueError, match="workspace state"):
        validate_requirements(required, DockerEnvironment(image, workdir="/repo"))


@pytest.mark.parametrize("answer_kind", ["text", "literal", "json", "final_state"])
def test_answer_verifier_rejects_native_action_submission_before_export(math_task, tmp_path, answer_kind):
    spec = msgspec.structs.replace(
        math_task,
        steps=(msgspec.structs.replace(math_task.steps[0], answer_requirements=AnswerRequirements(answer_kind)),),
    )
    destination = tmp_path / "invalid"
    with pytest.raises(ValueError, match="predicted-action verification"):
        lower_to_harbor(
            spec,
            (Rendering("action", FinalActionSubmission((NativeFunction("submit", {}),))),),
            HarborTaskBinding(NoEnvironment()),
            destination,
        )
    assert not destination.exists()


@pytest.mark.parametrize("agent", ["terminus-2", "mini-swe-agent"])
def test_native_terminal_launch_rejects_ordered_steps_without_history_support(agent):
    binding = HarborTaskBinding(DockerEnvironment("sha256:" + "1" * 64), (HarnessToolBinding("terminal", "docker"),))
    with pytest.raises(ValueError, match="retain conversation"):
        resolve_harbor_execution(
            (Rendering("plain", AssistantFinal()),) * 2,
            HarborExecutionConfig(binding, HarborLaunchConfig(agent)),
            {"import_path": "taskcompendium.harbor.environments:TaskDockerEnvironment"},
        )
