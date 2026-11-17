FROM node@sha256:aaa4b5baf1ecff3b283486bb1f90405cdda225a3e6f79660d2c559fa510a7292 AS node
FROM python@sha256:db6ee5170ebce0f4d165af242e8e77addc08bd9ffc37e8b1815752cf027aa4f8
COPY requirements.lock /tmp/requirements.lock
RUN python -m pip install --no-cache-dir --only-binary=:all: --require-hashes -r /tmp/requirements.lock
WORKDIR /testbed
COPY --from=node /usr/local/bin/node /usr/local/bin/node
ARG PI_VERSION=0.87.0
RUN python -c "import urllib.request; urllib.request.urlretrieve('https://github.com/earendil-works/pi/releases/download/v${PI_VERSION}/pi-linux-x64.tar.gz','/tmp/pi.tar.gz')" \
    && echo 'f55d02652175cd3f22e6db8223c93ef97a087382abb51dc9ce130fc04de8f746  /tmp/pi.tar.gz' | sha256sum -c - \
    && mkdir -p /opt/pi \
    && tar -xzf /tmp/pi.tar.gz -C /opt/pi --strip-components=1 \
    && rm /tmp/pi.tar.gz \
    && ln -s /opt/pi/pi /usr/local/bin/pi \
    && test "$(pi --version)" = "$PI_VERSION" \
    && node --version
RUN mkdir -p /testbed/benchmark /testbed/pred_results \
    && ln -s /setup_files/datasets /testbed/benchmark/datasets
