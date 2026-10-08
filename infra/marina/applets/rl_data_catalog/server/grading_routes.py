# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Select the grading entry points and data used by each Atlas source route."""

import ast
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .grading_code import PythonModuleSource

ENVIRONMENT_METHODS = {"__init__", "init", "step", "set_rollout_evidence", "close"}
NEMOTRON_PREFIX = "skyrl_gym.envs.nemotron_ultra."
JUDGE_AGENTS = {"abstention_simple_agent", "multichallenge_simple_agent"}
LCB_EXECUTION_MODULE = "skyrl_gym.envs.lcb.verifyit_execution"


@dataclass(frozen=True)
class GradingRoute:
    name: str
    roots: dict[str, list[str]]
    bindings: dict[str, Any]
    resources: tuple[str, ...]


def environment_roots(source: PythonModuleSource, entrypoint: str) -> dict[str, list[str]]:
    module, name = entrypoint.split(":")
    tree = ast.parse(source.read(module))
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name]
    if len(classes) != 1:
        raise ValueError(f"Grading entry point {entrypoint!r} is absent or ambiguous")
    methods = {node.name for node in classes[0].body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    if "step" not in methods:
        raise ValueError(f"Grading entry point {entrypoint!r} has no terminal grading method")
    roots = {module: [name + "." + method for method in sorted(methods & ENVIRONMENT_METHODS)]}
    inherited = {"init", "close", "set_rollout_evidence"} - methods
    if inherited:
        roots["skyrl_gym.envs.base_text_env"] = ["BaseTextEnv." + method for method in sorted(inherited)]
    return roots


def skyrl_grading_routes(
    row: Mapping[str, Any], source: PythonModuleSource, agents: Sequence[str]
) -> list[GradingRoute]:
    """Keep row-selected agents separate and include child-process grading entries."""
    mode = row["verifier_mode"]
    if mode == "harbor":
        return [
            GradingRoute(
                "harbor",
                {"harbor.verifier.verifier": ["Verifier.__init__", "Verifier.verify"]},
                {},
                (),
            )
        ]
    roots = environment_roots(source, row["gym_entrypoint"])
    roots["skyrl_train.trajectory_runners.skyrl_gym_contracts"] = [
        "verification_from_env_step",
        "fold_verification_results",
    ]
    bindings = {"self.verifyit_enabled": mode == "verifyit", "verifyit_enabled": mode == "verifyit"}
    environment = row["environment"]
    if environment != "nemotron_ultra":
        if environment == "lcb" and mode == "verifyit":
            roots[LCB_EXECUTION_MODULE] = ["_execute"]
            bindings["__verifyit_modes__"] = ["script", "exact"]
        elif environment == "text_to_sql" and mode == "verifyit":
            bindings["__verifyit_modes__"] = ["script"]
        elif environment == "reasoning_gym" and mode == "verifyit":
            bindings["__verifyit_modes__"] = ["reasoning-gym"]
        return [GradingRoute(environment, roots, bindings, ())]
    if not agents:
        raise ValueError("A Nemotron grading route must identify the selected agents")
    routes = []
    for agent in sorted(agents):
        selected = {module: list(names) for module, names in roots.items()}
        values = {**bindings, "self.agent": agent}
        resources = ()
        if agent in {"ns_tools_simple_agent", "math_with_judge_simple_agent"} and mode == "verifyit":
            selected[NEMOTRON_PREFIX + "math_judge_verifyit"] = ["_evaluate"]
            values["__verifyit_modes__"] = ["script", "math", "judge"]
        elif agent in JUDGE_AGENTS or agent.startswith("jailbreak_"):
            kind = (
                "abstention"
                if agent == "abstention_simple_agent"
                else "multichallenge" if agent == "multichallenge_simple_agent" else "jailbreak"
            )
            values["__definition_bindings__"] = {
                NEMOTRON_PREFIX + "env:NemotronUltraEnv._grade_judge_profile": {"kind": kind},
                NEMOTRON_PREFIX + "judge_profiles_verifyit:_evaluate": {"kind": kind},
            }
            if mode == "verifyit":
                selected[NEMOTRON_PREFIX + "judge_profiles_verifyit"] = ["_evaluate"]
                values["__verifyit_modes__"] = ["script", "judge"]
            if agent == "abstention_simple_agent":
                resources = ("skyrl-gym/skyrl_gym/envs/nemotron_ultra/abstention_prompt.txt",)
            elif agent.startswith("jailbreak_"):
                resources = ("skyrl-gym/skyrl_gym/envs/nemotron_ultra/jailbreak_verifiers.yaml",)
        elif agent == "math_formal_lean_refinement_agent" and mode == "verifyit":
            selected[NEMOTRON_PREFIX + "lean_verifyit"] = ["_compile"]
            values["__verifyit_modes__"] = ["script"]
        elif agent == "code_gen_simple_agent" and mode == "verifyit":
            selected[LCB_EXECUTION_MODULE] = ["_execute"]
            values["__verifyit_modes__"] = ["script", "exact"]
        elif agent == "calendar_simple_agent" and mode == "verifyit":
            selected[NEMOTRON_PREFIX + "calendar_verifyit"] = ["_check"]
            values["__verifyit_modes__"] = ["script"]
        elif agent == "reasoning_gym_simple_agent" and mode == "verifyit":
            values["__verifyit_modes__"] = ["reasoning-gym"]
        elif agent == "indirect_prompt_injection_simple_agent":
            values["__verifyit_modes__"] = ["script"]
        routes.append(GradingRoute(agent, selected, values, resources))
    return routes
