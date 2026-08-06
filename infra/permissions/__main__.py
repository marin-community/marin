# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Retired: this project's grants are now declared by `iac.gcp.iam.GcpIam` in the `marin` GCP
stack under infra/pulumi (see #7576). Every resource this program used to manage is
non-authoritative, so retiring it does not revoke anything live; do not run `pulumi up` or
`pulumi preview` here."""


def main() -> None:
    raise RuntimeError(
        "infra/permissions is retired. Its grants now live in infra/pulumi's `marin` GCP stack "
        "(iac.gcp.iam.GcpIam) — see https://github.com/marin-community/marin/issues/7576."
    )


main()
