# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Standing CoreWeave organization policies for pre-registered OpenBao identities."""

from collections.abc import Mapping
from dataclasses import dataclass
from urllib.parse import urlsplit
from uuid import UUID

import pulumi
import pulumi_coreweave as coreweave
from rigging.grants import Permission

OBJECT_ACTIONS = frozenset(
    {"s3:GetObject", "s3:PutObject", "s3:DeleteObject", "s3:AbortMultipartUpload", "s3:ListMultipartUploadParts"}
)
BUCKET_ACTIONS = frozenset({"s3:ListBucket", "s3:GetBucketLocation", "s3:ListBucketMultipartUploads"})


@dataclass(frozen=True)
class CoreweaveIdentity:
    subject: str
    permissions: tuple[Permission, ...]


def identity_policy_statements(issuer: str, identity: CoreweaveIdentity) -> list[dict]:
    """Compile grant permissions to the provider's organization-policy wire schema.

    Organization policies cannot express an S3 list-prefix condition. Bucket-wide
    actions are therefore accepted only for a grant covering the bucket root.
    """
    parsed_issuer = urlsplit(issuer)
    if (
        parsed_issuer.scheme != "https"
        or not parsed_issuer.hostname
        or parsed_issuer.query
        or parsed_issuer.fragment
        or "*" in issuer
    ):
        raise ValueError("CoreWeave requires an exact public HTTPS issuer")
    UUID(identity.subject)
    principal = f"role/{issuer}:{identity.subject}"
    statements = [
        {
            "name": "create-temporary-credentials",
            "effect": "Allow",
            "actions": ["cwobject:CreateAccessKeyOIDC"],
            "resources": ["*"],
            "principals": [principal],
        }
    ]
    if not identity.permissions:
        raise ValueError("A CoreWeave identity needs an explicit storage permission set")
    for index, permission in enumerate(identity.permissions):
        if permission.service != "cw_s3" or not permission.actions <= OBJECT_ACTIONS | BUCKET_ACTIONS:
            raise ValueError("CoreWeave identities support explicit object and bucket S3 actions only")
        resource = urlsplit(permission.resource)
        if resource.scheme != "s3" or not resource.netloc:
            raise ValueError("CoreWeave storage resources must use s3://bucket/prefix/")
        prefix = resource.path.lstrip("/")
        object_actions = permission.actions & OBJECT_ACTIONS
        bucket_actions = permission.actions & BUCKET_ACTIONS
        if bucket_actions and prefix:
            raise ValueError("CoreWeave bucket actions cannot be restricted to an object prefix")
        if object_actions:
            statements.append(
                {
                    "name": f"objects-{index}",
                    "effect": "Allow",
                    "actions": sorted(object_actions),
                    "resources": [f"{resource.netloc}/{prefix}*"],
                    "principals": [principal],
                }
            )
        if bucket_actions:
            statements.append(
                {
                    "name": f"bucket-{index}",
                    "effect": "Allow",
                    "actions": sorted(bucket_actions),
                    "resources": [resource.netloc],
                    "principals": [principal],
                }
            )
    return statements


def _unique_subjects(identities: dict[str, CoreweaveIdentity]) -> dict[str, CoreweaveIdentity]:
    subjects = [identity.subject for identity in identities.values()]
    if len(set(subjects)) != len(subjects):
        raise ValueError("A CoreWeave subject must have exactly one complete policy mapping")
    return identities


class CoreweaveIdentityPolicies(pulumi.ComponentResource):
    """Own scoped organization policies; WIF registration remains a Console prerequisite."""

    def __init__(
        self,
        name: str,
        *,
        issuer: pulumi.Input[str],
        identities: Mapping[str, pulumi.Input[CoreweaveIdentity]],
        coreweave_provider: pulumi.ProviderResource,
        opts: pulumi.ResourceOptions | None = None,
    ) -> None:
        super().__init__("marin:coreweave:IdentityPolicies", name, None, opts)
        resolved = pulumi.Output.from_input(dict(identities)).apply(_unique_subjects)
        self.policies = {}
        for identity_name in sorted(identities):
            identity = resolved.apply(lambda values, identity_name=identity_name: values[identity_name])
            self.policies[identity_name] = coreweave.ObjectStorageOrganizationAccessPolicy(
                f"{name}-{identity_name}",
                name=f"marin-auth-{identity_name}",
                statements=pulumi.Output.all(issuer=issuer, identity=identity).apply(
                    lambda values: identity_policy_statements(values["issuer"], values["identity"])
                ),
                opts=pulumi.ResourceOptions(parent=self, provider=coreweave_provider),
            )
        self.register_outputs({})
