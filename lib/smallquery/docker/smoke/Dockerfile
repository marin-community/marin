# Smoke overlay: bake the generated Parquet fixture and the queries onto the base
# Ballista image so the smoke needs no bind mounts. Every container then sees /data and
# /queries.sql at an identical path, which is what a distributed scan/shuffle requires
# and what avoids host-path bind-mount differences across docker setups.
ARG BASE=smallquery-ballista:local
FROM ${BASE}
COPY fixture/ /data/
COPY queries.sql /queries.sql
