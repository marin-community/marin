# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Check the public storage boundary through serialized provider resources."""

import pulumi
import pulumi_cloudflare as cloudflare
from iac.buckets.r2 import R2DataBuckets, R2PublicBucket
from pulumi.runtime import MockCallArgs, MockResourceArgs, Mocks
from rigging.filesystem.cluster_config import DataConfig


class RecordingMocks(Mocks):
    def __init__(self) -> None:
        self.resources: list[MockResourceArgs] = []

    def new_resource(self, args: MockResourceArgs):
        self.resources.append(args)
        return f"{args.name}_id", args.inputs

    def call(self, args: MockCallArgs):
        return args.args, []


@pulumi.runtime.test
def test_public_bucket_exposes_only_release_storage_and_respects_cache_headers():
    mocks = RecordingMocks()
    pulumi.runtime.set_mocks(mocks, project="marin-buckets", stack="test", preview=True)
    provider = cloudflare.Provider("cloudflare")
    R2DataBuckets(
        "private",
        account_id="account",
        bucket_name="private-data",
        lifecycle_config=DataConfig(region_buckets={}),
        cloudflare_provider=provider,
    )
    public = R2PublicBucket(
        "public",
        account_id="account",
        bucket_name="public-releases",
        hostname="public.example.com",
        zone_id="zone",
        cloudflare_provider=provider,
    )

    def check(_):
        domains = [
            resource.inputs
            for resource in mocks.resources
            if resource.typ == "cloudflare:index/r2CustomDomain:R2CustomDomain"
        ]
        assert [(item["bucketName"], item["enabled"]) for item in domains] == [("public-releases", True)]
        managed = next(
            resource.inputs
            for resource in mocks.resources
            if resource.typ == "cloudflare:index/r2ManagedDomain:R2ManagedDomain"
        )
        assert managed["enabled"] is False
        cache = next(
            resource.inputs for resource in mocks.resources if resource.typ == "cloudflare:index/ruleset:Ruleset"
        )
        assert cache["phase"] == "http_request_cache_settings"
        (rule,) = cache["rules"]
        assert rule["expression"] == '(http.host eq "public.example.com")'
        parameters = rule["actionParameters"]
        assert parameters["cache"] is True and parameters["originCacheControl"] is True
        assert parameters["edgeTtl"] == {"mode": "respect_origin"}

    return pulumi.Output.all(public.url, public.prefix).apply(check)
