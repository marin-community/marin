# Environment variables that we want to set in Ray
ENV_VARS = {}

PIP_DEPS = [
    "${RAY_RUNTIME_ENV_CREATE_WORKING_DIR} --extra-index-url https://download.pytorch.org/whl/cpu",
    "levanter@git+https://github.com/stanford-crfm/levanter.git@main",
]

REMOTE_DASHBOARD_URL = "http://127.0.0.1:8265"
