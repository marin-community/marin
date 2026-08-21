# Marin fork ferry sessions

Use the repository's `refresh-fork` skill for the fork unit named in the request.
Treat `config/external/migration.toml` as the source of fork topology, validation,
and blocker policy. Carry a successful refresh through its Marin pull request and
follow the repository's commit workflow; file the descriptor's blocker issue only
for a durable external blocker.

For `tpu-vllm`, one session owns both selected sources and the paired-wheel
candidate. Stop with the qualified candidate pinned in a draft PR. Stable branch
movement, release promotion, and the final descriptor update require human review.
