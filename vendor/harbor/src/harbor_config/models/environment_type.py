from enum import Enum


class EnvironmentType(str, Enum):
    IRIS = "iris"
    DAYTONA = "daytona"
    QEMU = "qemu"
    SHELLSIM = "shellsim"
