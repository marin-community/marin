# CatCount native gate logs

These captures retain the metrics read by the CatCount spec, the telemetry run
identity and `Training done!` in their original order. Setup output and unrelated
metric fields are excluded. Retained values and repeated payloads are unchanged.

The healthy asynchronous run took 635.58 seconds. The completed-step divergence
control took 680 seconds; both ranks completed optimizer steps while rank 1
used unsynchronized gradients. It fails the DP weight checksum row.

- `healthy.txt`: source `phase-c-wrap-healthy-a1-native.log`, source SHA256 `15d3fe17d9719a36b221d1d0eb99af9ecd1626901eef4539c34faf29f42f4189`, trimmed SHA256 `0ab2d5cc86441b007a26a30f280e379ba0d7a26a0a36d3e728f131534864c434`.
- `divergence.txt`: source `wrap-divergence-a2-complete-native.log`, source SHA256 `719c2d419fdf9aa20f1438faf9971e253f1d24a3a54726c81ccdac0b87bd3922`, trimmed SHA256 `32faddd9c7d3bfeaeed4010021dbbda67c0c0de329bbbbe5ec72b3aa40d0372d`.
