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
