# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Render the public deployment and policy mock without any provider credentials."""

import argparse
import json
from pathlib import Path

import yaml
from iac.kubernetes.openbao import OpenBaoDeployment, openbao_manifests
from iac.openbao import CredentialProfile, credential_catalog


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stack-file", type=Path, required=True)
    parser.add_argument("--catalog-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    stack = yaml.safe_load(args.stack_file.read_text())
    deployment = OpenBaoDeployment.model_validate(stack["config"]["marin-openbao:deployment"])
    catalog = yaml.safe_load(args.catalog_file.read_text())
    profiles = tuple(CredentialProfile.model_validate(profile) for profile in catalog["profiles"])
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "kubernetes.yaml").write_text(yaml.safe_dump_all(openbao_manifests(deployment), sort_keys=False))
    (args.output / "catalog.json").write_text(
        json.dumps({cluster: credential_catalog(cluster, profiles) for cluster in catalog["clusters"]}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
