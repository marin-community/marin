# Environment variables that we want to set in Ray
ENV_VARS = {
    "PYTHONPATH": "./submodules/levanter/src:${PYTHONPATH}",
}

PIP_DEPS = [
    "${RAY_RUNTIME_ENV_CREATE_WORKING_DIR} --extra-index-url https://download.pytorch.org/whl/cpu",
]

REMOTE_DASHBOARD_URL = "http://127.0.0.1:8265"
