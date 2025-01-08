# Environment variables that we want to set in Ray
ENV_VARS = {
    "PYTHONPATH": "./submodules/levanter/src:${PYTHONPATH}",
}

PIP_DEPS = []

REMOTE_DASHBOARD_URL = "http://127.0.0.1:8265"
