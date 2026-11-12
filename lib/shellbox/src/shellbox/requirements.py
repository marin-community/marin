# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Spec-able backend demands, independent of backend selection.

Field classifications are checked against the owning schemas by the Atlas
survey tests. New fields must be classified even when they create no demand.
"""

from enum import StrEnum


class Requirement(StrEnum):
    DOCKER_IMAGE = "docker_image"
    DOCKER_BUILD = "docker_build"
    PREPARED_IMAGE = "prepared_image"
    QEMU_BUNDLE = "qemu_bundle"
    HOST_IMAGE = "host_image"
    PACKAGES_LOCK = "packages_lock"
    PACKAGE_SPEC = "package_spec"
    LINUX_PROCESS = "linux_process"
    SHELL_SIMULATOR = "shell_simulator"
    SHELL = "shell"
    FILESYSTEM = "filesystem"
    PYTHON = "python"
    PROCESS = "process"
    NETWORK = "network"
    GIT_REPOSITORY = "git_repository"
    RETAINED_COMMAND_CANCELLATION = "retained_command_cancellation"
    NETWORK_ALLOW = "network_allow"
    NETWORK_DENY = "network_deny"
    STORAGE = "storage_mb"
    MEMORY = "memory_mb"
    CPUS = "cpus"
    GPUS = "gpus"
    WORKDIR = "working_directory"
    ENV = "environment_variables"
    SETUP = "setup_commands"
    USER = "non_root_user"
    MOUNTS = "mounts"
    FILE_UPLOAD = "file_upload"
    FILE_MODE = "file_mode"
    FILE_MTIME = "file_mtime"
    COMMAND_TIMEOUT = "command_timeout"
    STARTUP_TIMEOUT = "startup_timeout"
    CLEANUP_TIMEOUT = "cleanup_timeout"
    OUTPUT_LIMIT = "output_limit_bytes"
    STDIN = "stdin"
    DOCKER_IN_DOCKER = "docker_in_docker"
    TOOL_PROVIDERS = "tool_providers"
    HEALTHCHECK = "healthcheck"
    SKILLS = "skills_dir"
    WINDOWS = "windows"
    GPU_TYPES = "gpu_types"
    TPU = "tpu"
    SERVICES = "services"


class Support(StrEnum):
    SUPPORTED = "supported"
    PARTIAL = "partial"
    UNSUPPORTED = "unsupported"


MACHINE_FIELDS = {
    "source": (
        Requirement.DOCKER_IMAGE,
        Requirement.DOCKER_BUILD,
        Requirement.PREPARED_IMAGE,
        Requirement.QEMU_BUNDLE,
        Requirement.HOST_IMAGE,
        Requirement.PACKAGE_SPEC,
    ),
    "workdir": (Requirement.WORKDIR,),
    "env": (Requirement.ENV,),
    "network": (Requirement.NETWORK_ALLOW, Requirement.NETWORK_DENY),
    "memory_mb": (Requirement.MEMORY,),
    "cpus": (Requirement.CPUS,),
    "storage_mb": (Requirement.STORAGE,),
    "gpus": (Requirement.GPUS,),
    "startup_timeout": (Requirement.STARTUP_TIMEOUT,),
}

COMMAND_FIELDS = {
    "argv": (Requirement.PROCESS,),
    "cwd": (Requirement.WORKDIR,),
    "env": (Requirement.ENV,),
    "stdin": (Requirement.STDIN,),
    "timeout": (Requirement.COMMAND_TIMEOUT,),
    "output_limit_bytes": (Requirement.OUTPUT_LIMIT,),
    "user": (Requirement.USER,),
}

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

SOURCE_FIELDS = {
    "DockerImage": {"reference": (Requirement.DOCKER_IMAGE,)},
    "RegistryImage": {"reference": (Requirement.DOCKER_IMAGE,)},
    "DockerfileSource": {"context": (Requirement.DOCKER_BUILD,), "dockerfile": (Requirement.DOCKER_BUILD,)},
    "PreparedImage": {"layout": (Requirement.PREPARED_IMAGE,), "manifest_digest": (), "source": ()},
    "QemuBundle": {"path": (Requirement.QEMU_BUNDLE,), "image_reference": ()},
    "PackageSpec": {"pypi_packages": (Requirement.PACKAGE_SPEC,), "tools": (Requirement.PACKAGE_SPEC,)},
    "HostImage": {
        "read_only": (Requirement.MOUNTS,),
        "bin_dirs": (Requirement.HOST_IMAGE,),
        "mounts": (Requirement.MOUNTS,),
    },
    "Mount": {"host": (Requirement.MOUNTS,), "guest": (Requirement.MOUNTS,)},
}
