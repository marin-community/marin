# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import platform
from dataclasses import asdict, dataclass
from importlib import import_module, metadata

REQUIRED_DISTRIBUTIONS = (
    "nvidia-cutlass-dsl",
    "nvidia-nvshmem-cu13",
    "nvshmem4py-cu13",
    "cuda-core",
    "cuda-python",
)

REQUIRED_CUTE_APIS = (
    "get_peer_tensor",
    "get",
    "get_nbi",
    "get_nbi_block",
    "put",
    "put_nbi",
    "put_signal",
    "put_signal_nbi",
    "signal_op",
    "signal_wait",
)


@dataclass(frozen=True)
class EnvironmentReport:
    python: str
    platform: str
    distributions: dict[str, str]
    missing_cute_apis: tuple[str, ...]


def environment_report() -> EnvironmentReport:
    distributions = {}
    for distribution in REQUIRED_DISTRIBUTIONS:
        try:
            distributions[distribution] = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            distributions[distribution] = "NOT INSTALLED"

    try:
        cute_nvshmem = import_module("nvshmem.core.device.cute")
        missing_cute_apis = tuple(name for name in REQUIRED_CUTE_APIS if not hasattr(cute_nvshmem, name))
    except Exception as error:
        missing_cute_apis = (f"import failed: {type(error).__name__}: {error}",)

    return EnvironmentReport(
        python=platform.python_version(),
        platform=platform.platform(),
        distributions=distributions,
        missing_cute_apis=missing_cute_apis,
    )


def main() -> None:
    report = environment_report()
    print(json.dumps(asdict(report), indent=2, sort_keys=True))
    if "NOT INSTALLED" in report.distributions.values() or report.missing_cute_apis:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
