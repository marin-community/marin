# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Load byte-pinned upstream parsers without importing unrelated agent packages."""

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType


def source_parser(name: str, root: Path):
    """Load the selected original parser and its standard-library-only dependencies."""
    modules = {"pytest": ("pytest_parser", "PytestParser"), "swebench": ("swebench_parser", "SWEBenchParser")}
    module_name, class_name = modules[name]
    for package in ("terminal_bench", "terminal_bench.parsers", "terminal_bench.utils"):
        module = ModuleType(package)
        module.__path__ = [str(root / package.replace(".", "/"))]
        sys.modules[package] = module
    for relative in ("utils/logger.py", "parsers/base_parser.py", f"parsers/{module_name}.py"):
        path = root / "terminal_bench" / relative
        qualified = "terminal_bench." + relative.removesuffix(".py").replace("/", ".")
        spec = importlib.util.spec_from_file_location(qualified, path)
        assert spec is not None and spec.loader is not None
        loaded = importlib.util.module_from_spec(spec)
        sys.modules[qualified] = loaded
        spec.loader.exec_module(loaded)
    return getattr(sys.modules["terminal_bench.parsers." + module_name], class_name)()


def score(name: str, output: str, root: Path) -> tuple[float, dict[str, str]]:
    """Preserve the source all-passed predicate, including its empty-map behavior."""
    results = source_parser(name, root).parse(output)
    statuses = {key: value.value for key, value in results.items()}
    return float(all(value == "passed" for value in statuses.values())), statuses


def materialize_upstream(bundle: Path, output: Path) -> None:
    """Verify each archived upstream file before loading it as executable Python."""
    data = json.loads(bundle.read_bytes())
    for name, record in data["files"].items():
        raw = record["text"].encode()
        if hashlib.sha256(raw).hexdigest() != record["sha256"]:
            raise ValueError(f"Pinned upstream parser changed: {name}")
        target = output / "terminal_bench" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parser", choices=("pytest", "swebench"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reward", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).parent / "upstream"
    materialize_upstream(Path(__file__).with_name("upstream.json"), root)
    reward, statuses = score(args.parser, args.output.read_text(), root)
    args.reward.parent.mkdir(parents=True, exist_ok=True)
    args.reward.with_suffix(".statuses.json").write_text(json.dumps(statuses) + "\n")
    args.reward.write_text(f"{reward}\n")


if __name__ == "__main__":
    main()
