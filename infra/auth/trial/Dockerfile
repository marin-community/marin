# Build the production auth image first, then supply the reviewed proof binaries
# and Loom fixture through named BuildKit contexts. No credentials enter either.
ARG AUTH_IMAGE=marin-auth:openfga-10950
FROM ${AUTH_IMAGE}
USER root
RUN apt-get update && apt-get install -y --no-install-recommends git nodejs \
    && rm -rf /var/lib/apt/lists/*
COPY --from=proof-tools /bao /openfga /loom /tapestry /usr/local/bin/
COPY --from=loom-source /crates/loom/tests/fixtures/fake-acp-agent.mjs /opt/loom-source/crates/loom/tests/fixtures/fake-acp-agent.mjs
COPY infra/auth/prove.py infra/auth/prove_loom.py infra/auth/proof_fga.py infra/auth/fga_operator.py infra/auth/bootstrap.py /app/infra/auth/
COPY infra/auth/policy /app/infra/auth/policy
USER 1000:1000
CMD ["python", "-m", "infra.auth.prove", "--bao", "/usr/local/bin/bao", "--openfga", "/usr/local/bin/openfga", "--loom", "/usr/local/bin/loom", "--loom-source", "/opt/loom-source"]
