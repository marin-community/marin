# nl2bash prepared actor image v1

The repair lane's shared actor environment for the FWeindel nl2bash winners
(lowered recipe: digest-pinned `ubuntu:24.04` base plus the task contract's own
setup commands), baked by `atlas-repair-nl2bash-actor-build-v3` in-region and
pushed to `ghcr.io/marin-community/atlas-repair-actor-nl2bash-20261118:v1`.
The Dockerfile is the generator's exact output (`actor_image_dockerfile`);
rebuild it with any Docker-compatible builder and pin the result by digest.
