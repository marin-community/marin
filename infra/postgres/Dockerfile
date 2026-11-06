# Reuse the frozen dependency closure and native tools from the service trial.
FROM ghcr.io/marin-community/iris-worker@sha256:80703c8bcf75bc300389b6a2407b246dee1b97a7c3e0e30438d3b7eda67ef96b
COPY lib/rigging/src/rigging /app/.venv/lib/python3.12/site-packages/rigging/
COPY infra/auth/*.py /app/infra/auth/
COPY infra/postgres/*.py /app/infra/postgres/
CMD ["python", "-m", "infra.postgres.prove"]
