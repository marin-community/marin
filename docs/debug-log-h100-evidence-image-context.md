# H100 evidence image build context debug log

## Initial status

GitHub Actions run `31435250954`, job `93607833282`, checked out exact commit
`6096e41f6abaa88a98c7a82f504c71de312d240e`. The sole H100 image job failed
before executing the H100-specific package installation. No image or OCI digest
was produced, and the workflow was not rerun.

The BuildKit log reported a two-byte context followed by a missing source for
the H100 package manifest:

```text
#6 transferring context: 2B done
#15 COPY lib/iris/images/h100-evidence-debian12-amd64.sha256 ...
#15 ERROR: ... "/lib/iris/images/h100-evidence-debian12-amd64.sha256": not found
```

The missing registry cache tag was also reported, but BuildKit continued past
that cache-import miss. The missing `COPY` source was the fatal error.

## Hypothesis and evidence

Docker automatically uses `lib/iris/Dockerfile.dockerignore` for builds that
select `lib/iris/Dockerfile`; this file takes precedence over the repository
root `.dockerignore`. Its first active rule excludes the entire context with
`*`, after which a closed set of paths is re-included. The failing manifest is
tracked, nonempty, and copied by the Dockerfile, but it was absent from that
closed allowlist. This exactly explains both the two-byte context transfer and
the missing source at `COPY`.

## Change

Re-include the tracked package manifest in the Dockerfile-specific context.
Add a pre-build workflow check for a nonempty manifest and its exact re-include
rule, plus a behavior test that derives the source path from the real `COPY`,
reads its real bytes, and evaluates the ordered Docker ignore rules.

## Results

The focused H100 image policy suite passes 5/5, including deletion of the new
re-include rule as an explicit negative case. The repository pre-commit suite
also passes. No image build, workflow dispatch, workflow rerun, registry push,
or GPU operation was performed.
