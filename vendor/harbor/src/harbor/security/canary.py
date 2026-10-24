"""Verify the deployed gateway and Daytona firewall before enabling guarded trials."""

import argparse
import asyncio
import json
import shlex
import tempfile
from pathlib import Path

from harbor.environments.daytona_guarded import GuardedDaytonaEnvironment
from harbor.models.task.config import EnvironmentConfig
from harbor.models.trial.paths import TrialPaths

DIRECT_SOURCES = (
    "https://pypi.org/simple/packaging/",
    "https://files.pythonhosted.org/",
    "https://registry.npmjs.org/",
    "https://raw.githubusercontent.com/",
)


async def run_canary(gateway: str, image: str, service_domains: list[str]) -> list[dict]:
    """Create one disposable sandbox, probe downloads, and always delete it."""
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        paths = TrialPaths(trial_dir=root / "trial")
        paths.mkdir()
        environment = GuardedDaytonaEnvironment(
            environment_dir=root, environment_name="package-security-canary",
            session_id="package-security-canary", trial_paths=paths,
            task_env_config=EnvironmentConfig(allow_internet=True),
            package_gateway=gateway, approved_image=image, service_domains=service_domains,
            ttl_minutes=5,
        )
        observations = []
        try:
            await environment.start(force_build=False)
            probes = [
                (f"{gateway}/pypi/simple/packaging/", 200),
                (f"{gateway}/artifacts/pypi/brotli-1.2.0.tar.gz?name=brotli", 403),
                (f"{gateway}/pypi/simple/litellm/", 403),
                *((url, None) for url in DIRECT_SOURCES),
            ]
            for url, expected in probes:
                result = await environment.exec(
                    command=(
                        "curl --noproxy '*' --silent --show-error --output /dev/null "
                        "--connect-timeout 10 --max-time 15 --write-out '%{http_code}' "
                        + ("--insecure " if expected is None else "")
                        + shlex.quote(url)
                    ), timeout_sec=20,
                )
                status = (result.stdout or "").strip()
                record = {"url": url, "status": status, "exit_code": result.return_code}
                observations.append(record)
                if expected is not None:
                    if result.return_code != 0 or status != str(expected):
                        raise RuntimeError(f"Gateway probe failed: {json.dumps(observations)}")
                elif result.return_code == 0 and status != "403":
                    raise RuntimeError(f"Direct download origin reachable: {json.dumps(observations)}")
            return observations
        finally:
            await environment.stop(delete=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gateway", required=True)
    parser.add_argument("--approved-image", required=True)
    parser.add_argument("--service-domain", action="append", default=[])
    args = parser.parse_args()
    print(json.dumps(asyncio.run(run_canary(args.gateway.rstrip("/"), args.approved_image, args.service_domain)), indent=2))


if __name__ == "__main__":
    main()
