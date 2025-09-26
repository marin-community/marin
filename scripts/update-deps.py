#!/usr/bin/env uv run
# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# /// script
# requires-python = ">=3.11"
# dependencies = ["tomli-w"]
# ///
"""
Update and pin dependencies to their current versions.

This script uses a constraints-based approach:
1. Export current lock as constraints
2. Strip version specs from pyproject.toml
3. Run uv lock --upgrade with constraints
4. Pin exact versions back to pyproject.toml
"""

import re
import subprocess
import sys
import tomllib
import tomli_w
from pathlib import Path


def run_command(cmd, check=True):
    """Run a shell command."""
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=check)
    return result.returncode == 0


def run_command_output(cmd):
    """Run a shell command and return output."""
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return result.stdout


def export_constraints():
    """Export current lock as constraints."""
    output = run_command_output(["uv", "export", "--format", "requirements.txt"])

    constraints = []
    for line in output.split("\n"):
        if "==" in line and not line.startswith(("#", "-e")):
            # Parse: package==version \
            pkg_spec = line.split("\\")[0].strip()
            if "==" in pkg_spec:
                constraints.append(pkg_spec)

    return constraints


def strip_version_spec(dep_spec):
    """Strip version constraints from a dependency spec, keeping git refs and markers."""
    # Skip git dependencies
    if "@git+" in dep_spec or "@ git+" in dep_spec:
        return dep_spec

    # Handle environment markers (e.g., "swebench ; sys_platform != 'darwin'")
    if ";" in dep_spec:
        pkg_part, marker_part = dep_spec.split(";", 1)
        pkg_part = pkg_part.strip()
        marker_part = marker_part.strip()

        # Strip version from package part only
        pkg_stripped = strip_version_spec(pkg_part)
        return f"{pkg_stripped} ; {marker_part}"

    # Handle extras: package[extra]>=version -> package[extra]
    if "[" in dep_spec:
        # Find the closing bracket for extras
        bracket_end = dep_spec.find("]")
        if bracket_end != -1:
            base = dep_spec[: bracket_end + 1]  # includes [extras]
            rest = dep_spec[bracket_end + 1 :].strip()
            # Remove version specs from the rest
            for op in [">=", "<=", "==", ">", "<", "~=", "!="]:
                if rest.startswith(op):
                    return base
            return dep_spec
        else:
            # Malformed, just remove version specs
            for op in [">=", "<=", "==", ">", "<", "~=", "!="]:
                if op in dep_spec:
                    return dep_spec.split(op)[0].strip()
    else:
        # Remove version specs: package>=version -> package
        for op in [">=", "<=", "==", ">", "<", "~=", "!="]:
            if op in dep_spec:
                return dep_spec.split(op)[0].strip()
        return dep_spec.strip()


def strip_versions_from_pyproject():
    """Remove version specs from pyproject.toml, keeping git refs."""
    with open("pyproject.toml", "rb") as f:
        data = tomllib.load(f)

    # Process main dependencies
    if "project" in data and "dependencies" in data["project"]:
        data["project"]["dependencies"] = [strip_version_spec(dep) for dep in data["project"]["dependencies"]]

    # Process optional dependencies
    if "project" in data and "optional-dependencies" in data["project"]:
        for group_name, deps in data["project"]["optional-dependencies"].items():
            data["project"]["optional-dependencies"][group_name] = [strip_version_spec(dep) for dep in deps]

    # Process dependency groups
    if "dependency-groups" in data:
        for group_name, deps in data["dependency-groups"].items():
            processed_deps = []
            for dep in deps:
                if isinstance(dep, dict):
                    # Keep include-group references as-is
                    processed_deps.append(dep)
                else:
                    processed_deps.append(strip_version_spec(dep))
            data["dependency-groups"][group_name] = processed_deps

    with open("pyproject.toml", "wb") as f:
        tomli_w.dump(data, f)


def get_versions_from_export():
    """Get package versions from uv export."""
    output = run_command_output(["uv", "export", "--format", "requirements.txt"])

    versions = {}
    for line in output.split("\n"):
        if "==" in line and not line.startswith(("#", "-e")):
            pkg_spec = line.split("\\")[0].strip()
            if "==" in pkg_spec:
                name, version = pkg_spec.split("==", 1)
                # Handle extras: package[extra]==version
                if "[" in name:
                    name = name.split("[")[0]
                versions[name.lower().replace("-", "_")] = version

    return versions


def pin_versions_in_pyproject():
    """Update pyproject.toml with pinned versions from lock."""
    versions = get_versions_from_export()

    with open("pyproject.toml", "rb") as f:
        data = tomllib.load(f)

    def pin_dep_list(deps):
        """Pin versions in a list of dependencies."""
        pinned_deps = []
        for dep in deps:
            if isinstance(dep, dict):
                # Keep include-group references as-is
                pinned_deps.append(dep)
                continue

            # Skip git dependencies
            if "@git+" in dep or "@ git+" in dep:
                pinned_deps.append(dep)
                continue

            # Extract package name
            if "[" in dep:
                base_name = dep.split("[")[0]
                extras = dep.split("[")[1].split("]")[0]
                pkg_key = base_name.lower().replace("-", "_")
                if pkg_key in versions:
                    pinned_deps.append(f"{base_name}[{extras}]=={versions[pkg_key]}")
                else:
                    print(f"  Warning: {base_name} not found in lock, keeping as-is")
                    pinned_deps.append(dep)
            else:
                pkg_key = dep.lower().replace("-", "_")
                if pkg_key in versions:
                    pinned_deps.append(f"{dep}=={versions[pkg_key]}")
                else:
                    print(f"  Warning: {dep} not found in lock, keeping as-is")
                    pinned_deps.append(dep)

        return pinned_deps

    # Pin main dependencies
    if "project" in data and "dependencies" in data["project"]:
        data["project"]["dependencies"] = pin_dep_list(data["project"]["dependencies"])

    # Pin optional dependencies
    if "project" in data and "optional-dependencies" in data["project"]:
        for group_name, deps in data["project"]["optional-dependencies"].items():
            data["project"]["optional-dependencies"][group_name] = pin_dep_list(deps)

    # Pin dependency groups
    if "dependency-groups" in data:
        for group_name, deps in data["dependency-groups"].items():
            data["dependency-groups"][group_name] = pin_dep_list(deps)

    with open("pyproject.toml", "wb") as f:
        tomli_w.dump(data, f)


def main():
    if not Path("pyproject.toml").exists():
        print("Error: pyproject.toml not found in current directory")
        sys.exit(1)

    try:
        print("Step 1: Exporting current constraints...")
        constraints = export_constraints()
        Path("constraints.txt").write_text("\\n".join(constraints))
        print(f"  Exported {len(constraints)} package constraints")

        print("\\nStep 2: Stripping version specs from pyproject.toml...")
        strip_versions_from_pyproject()

        print("\\nStep 3: Running uv lock --upgrade...")
        if not run_command(["uv", "lock", "--upgrade"]):
            raise subprocess.CalledProcessError(1, "uv lock failed")

        print("\\nStep 4: Pinning versions in pyproject.toml...")
        pin_versions_in_pyproject()

        print("\\nStep 5: Final lock to verify consistency...")
        if not run_command(["uv", "lock"]):
            raise subprocess.CalledProcessError(1, "final uv lock failed")

        print("\\n✓ Successfully updated and pinned dependencies!")
        print("\\nTo review changes: git diff pyproject.toml uv.lock")
        print("To revert: git checkout -- pyproject.toml uv.lock")

    except Exception as e:
        print(f"\\nError: {e}")
        print("Use 'git checkout -- pyproject.toml uv.lock' to revert changes")
        sys.exit(1)

    finally:
        # Cleanup
        if Path("constraints.txt").exists():
            Path("constraints.txt").unlink()


if __name__ == "__main__":
    main()
