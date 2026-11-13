from enum import Enum


class EnvironmentType(str, Enum):
    LOCAL = "local"
    IRIS_SANDBOX = "iris-sandbox"
    DAYTONA = "daytona"
    QEMU = "qemu"
    SHELLSIM = "shellsim"
