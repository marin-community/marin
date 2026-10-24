from enum import Enum


class EnvironmentType(str, Enum):
    IRIS = "iris"
    LOCAL = "local"
    DAYTONA = "daytona"
    QEMU = "qemu"
    SHELLSIM = "shellsim"
