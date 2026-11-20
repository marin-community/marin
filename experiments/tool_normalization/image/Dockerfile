FROM docker.io/library/python@sha256:623674a14ef93d052bb9b0de4dbf5f28eb578a35263838941cc8aa43213c1487
RUN apt-get update && apt-get install -y --no-install-recommends git ripgrep libstdc++6 ca-certificates && rm -rf /var/lib/apt/lists/*
COPY package.tgz /tmp/opencode.tgz
ARG OPENCODE_SHA256
RUN echo "${OPENCODE_SHA256}  /tmp/opencode.tgz" | sha256sum --check && mkdir -p /opt/opencode && tar -xzf /tmp/opencode.tgz -C /opt/opencode && rm /tmp/opencode.tgz && ln -s /opt/opencode/package/bin/opencode /usr/local/bin/opencode
RUN useradd --create-home --uid 1000 actor && mkdir -p /workspace && chown actor:actor /workspace && dpkg-query -W > /opt/os-packages.tsv
USER actor
WORKDIR /workspace
