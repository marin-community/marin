# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import httpx
import pytest

from infra.marina.applets.rl_data_catalog.server.grading_code import python_grading_code, python_grading_program
from infra.marina.applets.rl_data_catalog.server.grading_dependencies import (
    GradingRepository,
    RepositoryGradingModules,
    grading_requirements,
    locked_grading_packages,
    verifyit_repository,
)
from infra.marina.applets.rl_data_catalog.server.grading_routes import skyrl_grading_routes

GRADER = '''
from shared.math import compare
from shared.code import execute
THRESHOLD = 1
def unused(value):
    return value
def score(candidate):
    """Grade one answer."""
    if self.agent == "math":
        return compare(candidate) >= THRESHOLD
    return execute(candidate)
'''


@pytest.mark.parametrize(
    "changed",
    [
        GRADER.replace("Grade one answer.", "Updated documentation."),
        GRADER.replace("return value", "return value + 2"),
        GRADER.replace("return execute(candidate)", "return execute(candidate) / 2"),
        "# Copyright update\n" + GRADER,
    ],
)
def test_grading_identity_preserves_ratings_for_unrelated_edits(changed: str) -> None:
    original = python_grading_code(GRADER, ["score"], {"self.agent": "math"})
    current = python_grading_code(changed, ["score"], {"self.agent": "math"})
    assert original.digest == current.digest
    assert current.imports == (("shared.math", "compare"),)


@pytest.mark.parametrize(
    "changed",
    [
        GRADER.replace("THRESHOLD = 1", "THRESHOLD = 2"),
        GRADER.replace(">= THRESHOLD", "> THRESHOLD"),
        GRADER.replace("from shared.math import compare", "from shared.math_v2 import compare"),
    ],
)
def test_grading_identity_invalidates_relevant_changes(changed: str) -> None:
    original = python_grading_code(GRADER, ["score"], {"self.agent": "math"})
    current = python_grading_code(changed, ["score"], {"self.agent": "math"})
    assert original.digest != current.digest


def test_grading_identity_tracks_transitive_local_helpers_without_executing_module(tmp_path) -> None:
    sentinel = tmp_path / "executed"
    source = f"""
from pathlib import Path
Path({str(sentinel)!r}).write_text("executed")
def helper(candidate):
    return candidate == "correct"
def grade(candidate):
    return helper(candidate)
"""
    old = python_grading_code(source, ["grade"])
    new = python_grading_code(source.replace('candidate == "correct"', 'candidate == "different"'), ["grade"])
    assert old.digest != new.digest
    assert not sentinel.exists()


def test_grading_identity_keeps_unknown_route_branches_in_scope() -> None:
    original = python_grading_code(GRADER, ["score"])
    changed = python_grading_code(
        GRADER.replace("return execute(candidate)", "return execute(candidate) / 2"), ["score"]
    )
    assert original.digest != changed.digest
    assert set(original.imports) == {("shared.math", "compare"), ("shared.code", "execute")}


def test_component_identity_uses_only_reachable_methods_and_state() -> None:
    source = """
class Env:
    def __init__(self):
        self.unused_session = None
        self.threshold = 1
    def step(self, answer):
        if self.agent == "math":
            return self.math(answer)
        return self.code(answer)
    def math(self, answer):
        return answer >= self.threshold
    def code(self, answer):
        return answer == "original"
"""
    roots = ["Env.__init__", "Env.step"]
    bindings = {"self.agent": "math"}
    original = python_grading_code(source, roots, bindings)
    unrelated = source.replace('answer == "original"', 'answer == "different"').replace(
        "        self.unused_session = None\n", ""
    )
    assert python_grading_code(unrelated, roots, bindings).digest == original.digest
    assert (
        python_grading_code(source.replace("self.threshold = 1", "self.threshold = 2"), roots, bindings).digest
        != original.digest
    )


def test_grading_identity_tracks_module_initialization_without_executing_it(tmp_path) -> None:
    sentinel = tmp_path / "executed"
    source = f"""
from pathlib import Path
Path({str(sentinel)!r}).write_text("original")
def grade(candidate):
    return candidate == "correct"
"""
    assert (
        python_grading_code(source, ["grade"]).digest
        != python_grading_code(source.replace('write_text("original")', 'write_text("different")'), ["grade"]).digest
    )
    assert not sentinel.exists()


class Modules:
    def __init__(self, modules: dict[str, str]):
        self.modules = modules

    def read(self, module: str) -> str:
        if module not in self.modules:
            raise ModuleNotFoundError(module)
        return self.modules[module]


def test_mcq_source_ignores_unrelated_verifier_modes() -> None:
    modules = {
        "skyrl_gym.envs.mcq.env": (
            "from verifyit.spec import McqSpec\nclass Env:\n"
            " def __init__(self):\n  self.spec = McqSpec()\n"
            " def step(self, answer):\n  return self.spec.mode\n"
        ),
        "skyrl_gym.envs.base_text_env": (
            "class BaseTextEnv:\n def init(self,x):\n  return x\n"
            " def close(self):\n  pass\n def set_rollout_evidence(self,x):\n  pass\n"
        ),
        "skyrl_train.trajectory_runners.skyrl_gym_contracts": (
            "def verification_from_env_step(x):\n return x\ndef fold_verification_results(x):\n return x\n"
        ),
        "verifyit.spec": (
            'from enum import StrEnum\nclass Mode(StrEnum):\n MCQ="mcq"\n PYTEST="pytest"\n'
            "class McqSpec:\n mode=Mode.MCQ\n"
        ),
    }
    row = {"environment": "mcq", "gym_entrypoint": "skyrl_gym.envs.mcq.env:Env", "verifier_mode": "verifyit"}
    source = Modules(modules)
    route = skyrl_grading_routes(row, source, ())[0]
    packages = ("skyrl_gym", "skyrl_train", "verifyit")
    original = python_grading_program(source, route.roots, packages, route.bindings)
    unrelated = {**modules, "verifyit.spec": modules["verifyit.spec"].replace('PYTEST="pytest"', 'PYTEST="new-pytest"')}
    assert python_grading_program(Modules(unrelated), route.roots, packages, route.bindings).digest == original.digest
    changed = {**modules, "verifyit.spec": modules["verifyit.spec"].replace('MCQ="mcq"', 'MCQ="changed-mcq"')}
    assert python_grading_program(Modules(changed), route.roots, packages, route.bindings).digest != original.digest


def test_grading_program_tracks_only_reachable_imported_graders() -> None:
    modules = {
        "grading.entry": "from .compare import equal\ndef grade(answer):\n return equal(answer)\n",
        "grading.compare": 'def equal(answer):\n return answer == "correct"\ndef unrelated():\n return 10\n',
    }
    roots = {"grading.entry": ["grade"]}
    original = python_grading_program(Modules(modules), roots, ("grading",))
    unrelated = {**modules, "grading.compare": modules["grading.compare"].replace("return 10", "return 20")}
    changed = {**modules, "grading.compare": modules["grading.compare"].replace('== "correct"', '== "different"')}
    assert python_grading_program(Modules(unrelated), roots, ("grading",)).digest == original.digest
    assert python_grading_program(Modules(changed), roots, ("grading",)).digest != original.digest


def test_grading_program_cannot_certify_missing_internal_dependency() -> None:
    source = Modules({"grading.entry": "from .missing import grade\n"})
    with pytest.raises(ModuleNotFoundError):
        python_grading_program(source, {"grading.entry": ["grade"]}, ("grading",))


@pytest.mark.parametrize("expression", ['getattr(self, "threshold")', 'vars(self)["threshold"]', "external(self)"])
def test_component_identity_tracks_state_exposed_through_reflection(expression: str) -> None:
    source = f"""
class Env:
    def __init__(self):
        self.threshold = 1
    def step(self):
        return {expression}
"""
    roots = ["Env.__init__", "Env.step"]
    assert (
        python_grading_code(source, roots).digest
        != python_grading_code(source.replace("self.threshold = 1", "self.threshold = 2"), roots).digest
    )


def test_repository_grading_program_ignores_unrelated_package_and_resource_updates() -> None:
    files = {
        "src/grading/entry.py": "from .helpers import equal\ndef grade(candidate):\n return equal(candidate)\n",
        "src/grading/helpers.py": 'def equal(candidate):\n return candidate == "correct"\n',
    }

    def handle(request: httpx.Request) -> httpx.Response:
        path = request.url.path.split("/revision/", 1)[1]
        return httpx.Response(200, text=files[path]) if path in files else httpx.Response(404)

    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        source = RepositoryGradingModules(
            client, {"grading": GradingRepository("org/repo", "revision", "src", "pyproject.toml")}
        )
        old = python_grading_program(source, {"grading.entry": ["grade"]}, ("grading",))
        files["src/grading/unrelated.py"] = "def different():\n return 20\n"
        files["docs/readme.md"] = "Updated instructions"
        source = RepositoryGradingModules(client, source.packages)
        assert python_grading_program(source, {"grading.entry": ["grade"]}, ("grading",)).digest == old.digest
        files["src/grading/helpers.py"] = files["src/grading/helpers.py"].replace('== "correct"', '== "different"')
        source = RepositoryGradingModules(client, source.packages)
        assert python_grading_program(source, {"grading.entry": ["grade"]}, ("grading",)).digest != old.digest


def test_grading_requirements_ignore_unused_extras_but_track_imported_dependency_changes() -> None:
    program = python_grading_program(
        Modules({"grading": "from jsonschema import validate\ndef grade(x):\n return validate(x, {})\n"}),
        {"grading": ["grade"]},
        ("grading",),
    )
    project = '[project]\ndependencies=["jsonschema>=4.20"]\n[project.optional-dependencies]\nother=["openai>=1"]\n'
    original = grading_requirements(program, {"package": project}, {})
    assert grading_requirements(program, {"package": project.replace("openai>=1", "openai>=2")}, {}) == original
    assert (
        grading_requirements(program, {"package": project.replace("jsonschema>=4.20", "jsonschema>=4.25")}, {})
        != original
    )


@pytest.mark.parametrize("subdirectory", ["", "#subdirectory=lib/verifyit"])
def test_verifyit_location_preserves_historical_package_provenance(subdirectory: str) -> None:
    revision = "a" * 40
    project = (
        f'[project]\ndependencies=["verifyit[answer] @ git+https://github.com/org/repo.git@{revision}{subdirectory}"]'
    )
    repository = verifyit_repository(project)
    assert repository.repository == "org/repo" and repository.revision == revision
    assert repository.source_root == ("lib/verifyit/src" if subdirectory else "src")


def test_grading_program_scopes_namespace_module_imports_to_called_helpers() -> None:
    modules = {
        "grading.entry": "from . import helpers\ndef grade(answer):\n return helpers.equal(answer)\n",
        "grading.helpers": 'def equal(answer):\n return answer == "correct"\ndef unused():\n return 10\n',
    }
    roots = {"grading.entry": ["grade"]}
    original = python_grading_program(Modules(modules), roots, ("grading",))
    changed = {**modules, "grading.helpers": modules["grading.helpers"].replace("return 10", "return 20")}
    assert python_grading_program(Modules(changed), roots, ("grading",)).digest == original.digest
    changed["grading.helpers"] = modules["grading.helpers"].replace('== "correct"', '== "different"')
    assert python_grading_program(Modules(changed), roots, ("grading",)).digest != original.digest


@pytest.mark.parametrize(
    "source,old,new",
    [
        (
            "from validator import validate\n@validate\ndef grade(candidate: int):\n return candidate == 1\n",
            "candidate: int",
            "candidate: float",
        ),
        (
            'def grade(candidate):\n """correct"""\n return candidate == grade.__doc__\n',
            '"""correct"""',
            '"""different"""',
        ),
        (
            "from setup import install\ninstalled = install(1)\ndef grade(x):\n return x == 1\n",
            "install(1)",
            "install(2)",
        ),
    ],
    ids=["decorated-annotation", "runtime-docstring", "initialization-side-effect"],
)
def test_grading_identity_tracks_runtime_metadata_and_initialization(source: str, old: str, new: str) -> None:
    assert (
        python_grading_code(source, ["grade"]).digest != python_grading_code(source.replace(old, new), ["grade"]).digest
    )


def test_imported_grader_ignores_cli_and_unrelated_static_dispatch_entries() -> None:
    module = """
MODES = {"math": "math", "code": "code"}
INVERSE = {value: key for key, value in MODES.items()}
def grade(x):
    return x == "correct"
def main():
    return "old CLI"
if __name__ == "__main__":
    main()
"""
    roots = {"grading": ["grade"]}
    original = python_grading_program(Modules({"grading": module}), roots, ("grading",))
    changed = module.replace('"code": "code"', '"code": "new code"').replace('"old CLI"', '"new CLI"')
    assert python_grading_program(Modules({"grading": changed}), roots, ("grading",)).digest == original.digest


def test_grading_program_tracks_only_selected_verifyit_dispatch_modes() -> None:
    modules = {
        "verifyit.spec": 'from enum import StrEnum\nclass Mode(StrEnum):\n SCRIPT="script"\n PYTEST="pytest"\n',
        "verifyit.grade": (
            "from verifyit.spec import Mode\n"
            'MODE_MODULES={Mode.SCRIPT:"script_impl",Mode.PYTEST:"pytest_impl"}\n'
            "def route():\n return MODE_MODULES[Mode.SCRIPT]\n"
        ),
    }
    roots = {"verifyit.grade": ["route"]}
    bindings = {"__verifyit_modes__": ["script"]}
    old = python_grading_program(Modules(modules), roots, ("verifyit",), bindings)
    unrelated = {**modules, "verifyit.grade": modules["verifyit.grade"].replace('"pytest_impl"', '"new_pytest_impl"')}
    changed = {**modules, "verifyit.grade": modules["verifyit.grade"].replace('"script_impl"', '"new_script_impl"')}
    assert python_grading_program(Modules(unrelated), roots, ("verifyit",), bindings).digest == old.digest
    assert python_grading_program(Modules(changed), roots, ("verifyit",), bindings).digest != old.digest


def test_runtime_lock_changes_affect_only_selected_grading_dependencies() -> None:
    lock = """
[[package]]
name = "jsonschema"
version = "4.20"
source = { registry = "https://pypi.org/simple" }
dependencies = [{ name = "referencing" }]
[[package]]
name = "referencing"
version = "0.35"
source = { registry = "https://pypi.org/simple" }
[[package]]
name = "openai"
version = "1.0"
source = { registry = "https://pypi.org/simple" }
"""
    original = locked_grading_packages(("jsonschema", "json"), lock)
    assert set(original) == {"jsonschema", "referencing"}
    assert locked_grading_packages(("jsonschema",), lock.replace('version = "1.0"', 'version = "2.0"')) == original
    assert locked_grading_packages(("jsonschema",), lock.replace('version = "0.35"', 'version = "0.36"')) != original


def test_grading_program_ignores_uncalled_base_metrics_but_tracks_constructor() -> None:
    modules = {
        "grading.child": (
            "from .base import Base\nclass Env(Base):\n def __init__(self):\n  super().__init__()\n"
            " def step(self,x):\n  return x == self.expected\n"
        ),
        "grading.base": (
            'class Base:\n def __init__(self):\n  self.expected="correct"\n'
            ' def metrics(self):\n  return "unused metrics"\n'
        ),
    }
    roots = {"grading.child": ["Env.__init__", "Env.step"]}
    original = python_grading_program(Modules(modules), roots, ("grading",))
    unrelated = {**modules, "grading.base": modules["grading.base"].replace("unused metrics", "new metrics")}
    changed = {**modules, "grading.base": modules["grading.base"].replace('="correct"', '="different"')}
    assert python_grading_program(Modules(unrelated), roots, ("grading",)).digest == original.digest
    assert python_grading_program(Modules(changed), roots, ("grading",)).digest != original.digest


def test_judge_profile_binding_excludes_sibling_profile_without_pruning_other_functions() -> None:
    module = """
def evaluate(kind):
    if kind == "abstention":
        return "old abstention"
    return "safety"
def other(kind):
    if kind == "abstention":
        return "other abstention"
    return "other safety"
"""
    bindings = {"__definition_bindings__": {"grading:evaluate": {"kind": "jailbreak"}}}
    roots = {"grading": ["evaluate", "other"]}
    original = python_grading_program(Modules({"grading": module}), roots, ("grading",), bindings)
    sibling = module.replace("old abstention", "new abstention")
    actual = module.replace("other abstention", "changed other abstention")
    assert python_grading_program(Modules({"grading": sibling}), roots, ("grading",), bindings).digest == original.digest
    assert python_grading_program(Modules({"grading": actual}), roots, ("grading",), bindings).digest != original.digest
