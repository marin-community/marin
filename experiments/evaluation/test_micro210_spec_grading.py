# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the self-contained Micro210 grading transforms."""

import hashlib
import json

import numpy as np
import pytest
from shellbox.machine import CommandSemantics, PackageSpec
from taskcompendium.models import (
    AnswerType,
    ConversationInput,
    EnvironmentRequirements,
    PlainText,
    ResourceGroups,
    ScriptGrader,
    Source,
    TaskSpec,
    TextMessage,
)
from taskcompendium.runtime.resources import inline_resource, resource_bytes

from experiments.evaluation.micro210_inputs import choice_seed
from experiments.evaluation.micro210_spec_grading import (
    HARBOR_PI_ADAPTER_PATCH,
    MBPP_ASSERTION_ORIGINAL,
    MBPP_ASSERTION_PATCHED,
    MBPP_FLOAT_TOLERANCE_ORIGINAL,
    MBPP_FLOAT_TOLERANCE_PATCHED,
    PROVENANCE_LABEL,
    SPEC_GRADE_DRIVER,
    GraderLibrary,
    driver_sha256,
    drop_unused_pi_adapter_pin,
    dropped_packages,
    embed_spec_grader,
    tolerant_mbpp_float_comparison,
)


@pytest.fixture
def library() -> GraderLibrary:
    return GraderLibrary()


@pytest.fixture
def task() -> TaskSpec:
    return TaskSpec(
        id="fixture/equivalence",
        context=ConversationInput(events=(TextMessage(role="user", content="What is 2+2?"),)),
        environment_requirements=EnvironmentRequirements(),
        answer_type=AnswerType.TEXT,
        answer_format=PlainText(),
        grader=ScriptGrader(
            argv=("python3", "/tests/grade.py"),
            environment=EnvironmentRequirements(
                command_semantics=CommandSemantics.LINUX_PROCESS,
                package_spec=PackageSpec(pypi_packages=("sympy==1.14.0",)),
            ),
        ),
        source=Source(dataset="fixture", revision="pinned", row="0", importer_revision="fixture-v1"),
        resources=ResourceGroups(),
    )


def script_task(task: TaskSpec) -> TaskSpec:
    """A minimal script-graded row carrying the frozen verifier environment pins."""
    config = {"spec": {"grader": {"name": "boxed_math", "reference": {}}}}
    grader = task.grader.model_copy(
        update={
            "environment": task.grader.environment.model_copy(
                update={
                    "package_spec": PackageSpec(
                        pypi_packages=(
                            "sympy==1.14.0",
                            "evalchemy[ifbench,mrcr,nupa5k-loose] @ https://example/evalchemy.whl#sha256=9970",
                            "lm-eval @ https://example/lm_eval.whl#sha256=ab96",
                        )
                    )
                }
            )
        }
    )
    resources = task.resources.model_copy(
        update={
            "verifier": (
                inline_resource("config.json", json.dumps(config).encode()),
                inline_resource("grade.py", b"import eval\n"),
            )
        }
    )
    return task.model_copy(update={"grader": grader, "resources": resources})


def harbor_task(task: TaskSpec) -> TaskSpec:
    """A minimal Harbor-trial row carrying the Pi runtime package manifests."""
    package = {
        "name": "long-loop-pi-runtime",
        "version": "1.0.0",
        "dependencies": {"@earendil-works/pi-coding-agent": "0.87.0", "pi-mcp-adapter": "7.0.0"},
    }
    lock = {
        "name": "long-loop-pi-runtime",
        "version": "1.0.0",
        "lockfileVersion": 3,
        "packages": {
            "": {"name": "long-loop-pi-runtime", "dependencies": package["dependencies"]},
            "node_modules/@earendil-works/pi-coding-agent": {"resolved": "https://example/pi.tgz"},
            "node_modules/pi-mcp-adapter": {"resolved": "https://example/adapter.tgz"},
        },
    }
    resources = task.resources.model_copy(
        update={
            "verifier": (
                inline_resource("task.toml", b"[verifier]\n"),
                inline_resource(
                    "environment/long-loop-pi/package.json",
                    (json.dumps(package, separators=(",", ":")) + "\n").encode(),
                ),
                inline_resource(
                    "environment/long-loop-pi/package-lock.json",
                    (json.dumps(lock, indent=2) + "\n").encode(),
                ),
            )
        }
    )
    return task.model_copy(update={"resources": resources})


def verifier_files(task: TaskSpec) -> dict[str, bytes]:
    return {resource.path: resource_bytes(resource) for resource in task.resources.verifier}


def test_grader_library_matches_its_manifest(library):
    assert library.manifest["source_wheels"][0]["sha256"].startswith("9970bac6")
    assert len(library.files) == len(library.manifest["files"])
    assert "micro210_registry.py" in library.files
    assert driver_sha256() == hashlib.sha256(SPEC_GRADE_DRIVER.read_bytes()).hexdigest()


def test_dropped_packages_keeps_third_party_pins():
    kept, dropped = dropped_packages(
        ("sympy==1.14.0", "evalchemy[x] @ https://w#sha256=a", "lm-eval @ https://w#sha256=b")
    )
    assert kept == ["sympy==1.14.0"]
    assert len(dropped) == 2


def test_embed_spec_grader_embeds_library_and_drops_ambient_wheels(task, library):
    embedded = embed_spec_grader(script_task(task), benchmark="aime24", library=library)
    files = verifier_files(embedded)
    assert files["grade.py"] == SPEC_GRADE_DRIVER.read_bytes()
    assert any(path.startswith("grader_lib/eval/") for path in files)
    assert "grader_lib/micro210_registry.py" in files
    assert embedded.grader.environment.package_spec.pypi_packages == ("sympy==1.14.0",)
    provenance = json.loads(embedded.labels[PROVENANCE_LABEL])
    assert provenance["removed_ambient_packages"] == [
        "evalchemy[ifbench,mrcr,nupa5k-loose] @ https://example/evalchemy.whl#sha256=9970",
        "lm-eval @ https://example/lm_eval.whl#sha256=ab96",
    ]
    assert provenance["library_manifest_sha256"] == library.manifest_sha256


def test_embed_spec_grader_bundles_nltk_data_only_for_ifbench(task, library):
    ifbench = embed_spec_grader(script_task(task), benchmark="ifbench", library=library)
    assert any(".nltk_data" in path for path in verifier_files(ifbench))
    aime = embed_spec_grader(script_task(task), benchmark="aime24", library=library)
    assert not any(".nltk_data" in path for path in verifier_files(aime))


def test_embed_spec_grader_records_data_patches_for_harbor_rows(task, library):
    patched = drop_unused_pi_adapter_pin(harbor_task(task))
    embedded = embed_spec_grader(
        patched, benchmark="bixbench-pi", library=library, data_patches=(HARBOR_PI_ADAPTER_PATCH,)
    )
    provenance = json.loads(embedded.labels[PROVENANCE_LABEL])
    assert provenance["row_data_patches"] == [HARBOR_PI_ADAPTER_PATCH]
    assert not any(path.startswith("grader_lib/") for path in verifier_files(embedded))


def test_drop_unused_pi_adapter_pin_updates_package_and_lock(task):
    patched = drop_unused_pi_adapter_pin(harbor_task(task))
    files = verifier_files(patched)
    package = json.loads(files["environment/long-loop-pi/package.json"])
    lock = json.loads(files["environment/long-loop-pi/package-lock.json"])
    assert package["dependencies"] == {"@earendil-works/pi-coding-agent": "0.87.0"}
    assert lock["packages"][""]["dependencies"] == {"@earendil-works/pi-coding-agent": "0.87.0"}
    assert "node_modules/pi-mcp-adapter" not in lock["packages"]
    assert files["environment/long-loop-pi/package.json"].endswith(b"\n")
    assert files["environment/long-loop-pi/package-lock.json"].endswith(b"\n")


def test_drop_unused_pi_adapter_pin_leaves_other_packages_alone(task):
    package = {
        "name": "long-loop-pi-runtime",
        "version": "1.0.0",
        "dependencies": {"@earendil-works/pi-coding-agent": "0.87.0", "pi-mcp-adapter": "7.0.1"},
    }
    task = harbor_task(task).model_copy(
        update={
            "resources": task.resources.model_copy(
                update={
                    "verifier": (
                        inline_resource(
                            "environment/long-loop-pi/package.json",
                            (json.dumps(package, separators=(",", ":")) + "\n").encode(),
                        ),
                    )
                }
            )
        }
    )
    assert drop_unused_pi_adapter_pin(task) == task


def test_tolerant_mbpp_float_comparison_patches_test_and_target(task):
    program = f"header\n{MBPP_FLOAT_TOLERANCE_ORIGINAL}\n{MBPP_ASSERTION_ORIGINAL}\ninputs = []\n"
    config = {
        "spec": {
            "grader": {"name": "program_suite", "reference": {"problem": {"test": program}}},
            "target": program,
        }
    }
    task = task.model_copy(
        update={
            "resources": task.resources.model_copy(
                update={"verifier": (inline_resource("config.json", json.dumps(config).encode()),)}
            )
        }
    )
    patched = tolerant_mbpp_float_comparison(task)
    config = json.loads(verifier_files(patched)["config.json"])
    test = config["spec"]["grader"]["reference"]["problem"]["test"]
    assert "isinstance(x, (float, complex))" in test
    assert "def _close(out, exp, atol):" in test
    assert config["spec"]["target"] == test


def test_tolerant_mbpp_float_comparison_fails_closed_without_the_guard(task):
    config = {"spec": {"grader": {"name": "program_suite", "reference": {"problem": {"test": "def other(): pass"}}}}}
    task = task.model_copy(
        update={
            "resources": task.resources.model_copy(
                update={"verifier": (inline_resource("config.json", json.dumps(config).encode()),)}
            )
        }
    )
    with pytest.raises(ValueError, match="does not carry the expected is_floats guard"):
        tolerant_mbpp_float_comparison(task)


def test_mbpp_patched_is_floats_recognizes_complex_and_nested_numerics():
    namespace: dict = {"np": np}
    exec(MBPP_FLOAT_TOLERANCE_PATCHED, namespace)
    is_floats = namespace["is_floats"]
    assert is_floats(((42.326324330254145, -2.359036256570777), complex(-2, 2.4492935982947064e-16)))
    assert is_floats([1.0, 2.5])
    assert not is_floats(True)
    assert not is_floats([1.0, "x"])
    assert not is_floats("text")


def test_mbpp_patched_assertion_tolerates_one_float_ulp():
    namespace: dict = {"np": np}
    # The embedded test program defines both helpers together.
    exec(f"{MBPP_FLOAT_TOLERANCE_PATCHED}\n\n{MBPP_ASSERTION_PATCHED}", namespace)
    computed = ((42.326324330254145, -2.359036256570777), complex(-2, 2.4492935982947064e-16))
    expected = ((42.32632433025415, -2.359036256570777), complex(-2, 2.4492935982947064e-16))
    namespace["assertion"](computed, expected, 0)
    with pytest.raises(AssertionError):
        namespace["assertion"]((1.0,), (2.0,), 0)


def test_choice_seed_is_stable_across_the_staged_namespace():
    assert choice_seed("truthfulqa/doc:67d9840e") == choice_seed("truthfulqa/doc:67d9840e")
    assert choice_seed("a") != choice_seed("b")
