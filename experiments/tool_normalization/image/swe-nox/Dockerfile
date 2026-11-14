FROM ghcr.io/marin-community/iris-task@sha256:ba67e05c032fd79776fe7eb48819868a6485df377af2201bb3cc23d00c6aa972 AS native
FROM ghcr.io/astral-sh/uv@sha256:b79b2e4fcf173bf617d3108f2b9fe4e407313203a943740e134cbfe4cf9dc2f4 AS uv
FROM docker.io/swerebenchv2/theacodes-nox@sha256:4bc3bc8910f9a6bb67e6fa962e1aef8d58cb41f06fea81ba953f1f5b202e1834
USER root
COPY --from=native /opt/opencode /opt/opencode
COPY --from=uv /uv /usr/local/bin/uv
RUN UV_PYTHON_INSTALL_DIR=/opt/tool-runner uv python install --no-bin cpython-3.12.11-linux-x86_64-gnu
RUN ln -s /opt/opencode/package/bin/opencode /usr/local/bin/opencode && opencode --version
RUN test "$(git -C /nox rev-parse HEAD)" = aa9290713a4b664ea4ef4da0554afb09a626d7e1
RUN cp -a /nox /testbed
RUN getent passwd 1000 && mkdir -p /home/tool-normalization && chown -R 1000:1000 /nox /testbed /workspace /home/tool-normalization
RUN /opt/tool-runner/cpython-3.12.11-linux-x86_64-gnu/bin/python3 --version && python3 --version
ENV HOME=/home/tool-normalization
USER 1000:1000
WORKDIR /testbed
