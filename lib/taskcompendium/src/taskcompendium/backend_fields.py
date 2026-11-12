# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Classify TaskCompendium, deployment-runtime and Harbor schema fields by backend demand."""

from shellbox.requirements import Requirement


ENVIRONMENT_FIELDS = {
    "capabilities": tuple(
        Requirement(name)
        for name in (
            "shell",
            "filesystem",
            "python",
            "process",
            "network",
            "git_repository",
            "retained_command_cancellation",
        )
    ),
    "command_semantics": (Requirement.LINUX_PROCESS, Requirement.SHELL_SIMULATOR),
    "docker_image": (Requirement.DOCKER_IMAGE,),
    "docker_build": (Requirement.DOCKER_BUILD,),
    "working_directory": (Requirement.WORKDIR,),
    "setup_commands": (Requirement.SETUP,),
    "environment_variables": (Requirement.ENV,),
    "tool_providers": (Requirement.TOOL_PROVIDERS,),
    "packages_lock": (Requirement.PACKAGES_LOCK, Requirement.HOST_IMAGE),
    "package_spec": (Requirement.PACKAGE_SPEC,),
}

RESOURCE_FIELDS = {
    "path": (Requirement.FILE_UPLOAD,),
    "source": (Requirement.FILE_UPLOAD,),
    "mode": (Requirement.FILE_MODE,),
    "mtime_ns": (Requirement.FILE_MTIME,),
}

RUNTIME_FIELDS = {
    # Backend is a deployment choice, never a task-distribution feature.
    "backend": (),
    "network": (Requirement.NETWORK_ALLOW, Requirement.NETWORK_DENY),
    "cpus": (Requirement.CPUS,),
    "memory_mb": (Requirement.MEMORY,),
    "storage_mb": (Requirement.STORAGE,),
    "gpus": (Requirement.GPUS,),
    "user": (Requirement.USER,),
    "startup_timeout": (Requirement.STARTUP_TIMEOUT,),
    "cleanup_timeout": (Requirement.CLEANUP_TIMEOUT,),
}

EXECUTION_FIELDS = {
    "cpus": (Requirement.CPUS,),
    "memory_mb": (Requirement.MEMORY,),
    "storage_mb": (Requirement.STORAGE,),
    "agent_timeout_sec": (Requirement.COMMAND_TIMEOUT,),
    "verifier_timeout_sec": (Requirement.COMMAND_TIMEOUT,),
}

HARBOR_ENVIRONMENT_FIELDS = {
    "build_timeout_sec": (Requirement.STARTUP_TIMEOUT,),
    "docker_image": (Requirement.DOCKER_IMAGE,),
    "os": (Requirement.WINDOWS,),
    "cpus": (Requirement.CPUS,),
    "memory_mb": (Requirement.MEMORY,),
    "storage_mb": (Requirement.STORAGE,),
    "gpus": (Requirement.GPUS,),
    "gpu_types": (Requirement.GPU_TYPES,),
    "tpu": (Requirement.TPU,),
    "allow_internet": (Requirement.NETWORK_ALLOW, Requirement.NETWORK_DENY),
    "mcp_servers": (Requirement.TOOL_PROVIDERS, Requirement.SERVICES),
    "env": (Requirement.ENV,),
    "skills_dir": (Requirement.SKILLS,),
    "healthcheck": (Requirement.HEALTHCHECK,),
    "setup": (Requirement.SETUP,),
    "workdir": (Requirement.WORKDIR,),
}

# Empty tuples classify task content and orchestration metadata with no machine
# feature of their own. Nested environment/resource schemas are checked separately.
TASK_FIELDS = {
    "id": (),
    "context": (),
    "environment_requirements": tuple(Requirement),
    "execution_requirements": tuple(sorted({need for needs in EXECUTION_FIELDS.values() for need in needs})),
    "final_tools": (),
    "interaction_tools": (),
    "tool_bindings": (Requirement.TOOL_PROVIDERS,),
    "output_paths": (Requirement.FILESYSTEM,),
    "answer_type": (),
    "answer_format": (),
    "grader": (),
    "reasoning_markup": (),
    "source": (),
    "schema_version": (),
    "resources": (Requirement.FILE_UPLOAD,),
    "tags": (),
    "labels": (),
    "integrity_contract": (Requirement.USER, Requirement.PROCESS, Requirement.FILESYSTEM, Requirement.FILE_MODE),
}

SCRIPT_GRADER_FIELDS = {
    "kind": (),
    "argv": (Requirement.PROCESS,),
    "cwd": (Requirement.WORKDIR,),
    "env": (Requirement.ENV,),
    "environment": tuple(Requirement),
    "collect": (Requirement.PROCESS,),
    "artifacts": (Requirement.FILESYSTEM,),
    "answer_path": (Requirement.FILE_UPLOAD,),
    "conversation_path": (Requirement.FILE_UPLOAD,),
    "reward": (),
    "timeout": (Requirement.COMMAND_TIMEOUT,),
    "empty_output": (),
    "judge": (),
}

HARBOR_AGENT_FIELDS = {
    "timeout_sec": (Requirement.COMMAND_TIMEOUT,),
    "env": (Requirement.ENV,),
    "user": (Requirement.USER,),
}
HARBOR_VERIFIER_FIELDS = {
    **HARBOR_AGENT_FIELDS,
    "environment_mode": (),
    "environment": tuple(Requirement),
    "collect": (Requirement.PROCESS,),
}

