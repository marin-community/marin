from enum import Enum


class EnvironmentType(str, Enum):
    LOCAL = "local"
    DAYTONA = "daytona"
    QEMU = "qemu"
    SHELLSIM = "shellsim"
