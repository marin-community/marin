# Reuse the reviewed proof dependency closure; this overlay performs no installs.
FROM ghcr.io/marin-community/iris-worker@sha256:80703c8bcf75bc300389b6a2407b246dee1b97a7c3e0e30438d3b7eda67ef96b

USER root
RUN mv /app/.venv /opt/marin-venv \
    && mv /app/grafana /opt/marin-grafana \
    && rm -rf /opt/marin-venv/lib/python3.12/site-packages/iris \
        /opt/marin-venv/lib/python3.12/site-packages/rigging
RUN /opt/marin-venv/bin/python - <<'PY'
from pathlib import Path

for script in Path("/opt/marin-venv/bin").iterdir():
    if script.is_symlink() or not script.is_file():
        continue
    content = script.read_bytes()
    if content.startswith(b"#!/app/.venv/"):
        script.write_bytes(content.replace(b"#!/app/.venv/", b"#!/opt/marin-venv/", 1))
PY

COPY lib/iris/src/iris /opt/marin-venv/lib/python3.12/site-packages/iris
COPY lib/iris/config /opt/marin-venv/lib/python3.12/site-packages/iris/config
COPY lib/rigging/src/rigging /opt/marin-venv/lib/python3.12/site-packages/rigging
COPY config /opt/marin-venv/lib/python3.12/site-packages/rigging/clusters
COPY infra/auth/*.py /opt/marin-source/infra/auth/
COPY infra/auth/policy /opt/marin-source/infra/auth/policy
COPY --from=execution-tools /loom /tapestry /usr/local/bin/
COPY --from=loom-source /crates/loom/tests/fixtures/fake-acp-agent.mjs /opt/loom-source/crates/loom/tests/fixtures/fake-acp-agent.mjs

ARG IRIS_REVISION_DATE
RUN test -n "$IRIS_REVISION_DATE" \
    && /usr/lib/postgresql/17/bin/pg_dump --version \
    && /usr/lib/postgresql/17/bin/pg_restore --version \
    && /usr/local/bin/bao --version \
    && /usr/local/bin/loom --version
ENV PATH="/opt/marin-venv/bin:$PATH" \
    PYTHONPATH="/opt/marin-source:/opt/marin-grafana" \
    IRIS_REVISION_DATE="$IRIS_REVISION_DATE" \
    UV_PROJECT_ENVIRONMENT="/opt/marin-venv" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1
WORKDIR /opt/marin-source
USER 1000:1000
CMD ["python", "-m", "infra.auth.prove_execution", "--help"]
