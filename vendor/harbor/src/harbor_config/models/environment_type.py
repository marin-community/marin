from enum import Enum


class EnvironmentType(str, Enum):
    DOCKER = "docker"
    DAYTONA = "daytona"
    DAYTONA_GUARDED = "daytona-guarded"
    E2B = "e2b"
    MODAL = "modal"
    RUNLOOP = "runloop"
    GKE = "gke"
    NOVITA = "novita"
    APPLE_CONTAINER = "apple-container"
    APPTAINER = "apptainer"
    ENROOT = "enroot"
    ISLO = "islo"
    TENSORLAKE = "tensorlake"
    CWSANDBOX = "cwsandbox"
    WANDB = "wandb"
