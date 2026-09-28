"""Release the full matched-Olmix cap x KL sweep once the canary has committed a checkpoint (Calvin's early-release rule).

Watches the canary parent; when a run under the sweep's output root has a committed checkpoint (the check of
`tpp10_execution_check.committed_checkpoint_exists`), runs the region guard and submits `launch_full_command.sh`.
Stops without releasing if the canary parent fails or is killed. Progress: early_release.log beside this file.
"""

import re
import subprocess
import time
from datetime import datetime
from pathlib import Path

import fsspec

from experiments.domain_phase_mix.tpp10_execution_check import committed_checkpoint_exists

HERE = Path(__file__).resolve().parent
ROOT = "gs://marin-us-east5/pinlin_calvin_xu/data_mixture/delphi_olmix_cap_kl_sweep_3e18_20260926"
CANARY = "/calvinxu/dm-delphi-3e18-olmix-cap-kl-canary3-v5p8-20260926"
LOG = HERE / "early_release.log"
IRIS = ["uv", "run", "--no-sync", "iris", "--config", "lib/iris/config/marin.yaml"]


def log(message: str) -> None:
    with LOG.open("a") as handle:
        handle.write(f"{datetime.now().strftime('%H:%M:%S')} {message}\n")


def state(job: str) -> str:
    out = subprocess.run([*IRIS, "job", "describe", job], capture_output=True, text=True).stdout
    return next((line.split()[1] for line in out.splitlines() if line.startswith("State:")), "unknown")


def run_dirs() -> list[str]:
    fs, _ = fsspec.core.url_to_fs(ROOT)
    if not fs.exists(ROOT):
        return []
    return ["gs://" + p for group in fs.ls(ROOT) for p in fs.ls(group) if "olmixsw_" in p]


def main() -> None:
    log("watching the canary")
    while True:
        s = state(CANARY)
        if s in ("failed", "killed"):
            log(f"STOP: canary {s}; nothing released")
            return
        ready = [d for d in run_dirs() if committed_checkpoint_exists(d)]
        if ready or s == "succeeded":
            log(f"canary committed a checkpoint ({ready[:1] or s}); releasing the full sweep")
            command = (HERE / "launch_full_command.sh").read_text().strip()
            guard = subprocess.run(["uv", "run", "--offline", "--no-sync", "python", "-m", "experiments.domain_phase_mix.east5_launch_safety",
                                    "--expected-child-zone", "us-east5-a", "--command", command], capture_output=True, text=True)
            if guard.returncode != 0:
                log(f"STOP: guard failed: {guard.stdout[-300:]} {guard.stderr[-300:]}")
                return
            # Credentials come only from a subshell that sources the secrets file; the saved log is redacted.
            shell = f"set -a; source ~/.zshrc.secrets >/dev/null 2>&1; set +a; bash {HERE / 'launch_full_command.sh'}"
            result = subprocess.run(["bash", "-c", shell], capture_output=True, text=True)
            redacted = re.sub(r"(WANDB_API_KEY|HF_TOKEN)(=|: ?)[^ ]+", r"\1=<redacted>", result.stdout + result.stderr)
            (HERE / "submit_full.log").write_text(re.sub(r"[0-9a-f]{40}", "<redacted-40hex>", redacted))
            log("released" if "Job submitted" in result.stdout + result.stderr else "STOP: full submission did not confirm; see submit_full.log")
            return
        log(f"canary {s}; no checkpoint yet")
        time.sleep(300)


if __name__ == "__main__":
    main()
