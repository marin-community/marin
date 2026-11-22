FROM node@sha256:aaa4b5baf1ecff3b283486bb1f90405cdda225a3e6f79660d2c559fa510a7292 AS node
FROM python@sha256:db6ee5170ebce0f4d165af242e8e77addc08bd9ffc37e8b1815752cf027aa4f8
COPY requirements.lock /tmp/requirements.lock
RUN python -m pip install --no-cache-dir --only-binary=:all: --no-binary=docopt,airr,docrep --no-build-isolation --require-hashes -r /tmp/requirements.lock
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
ARG EXPAT_PACKAGE=libexpat1_2.5.0-1+deb12u5_amd64.deb
RUN python -c "import urllib.request; urllib.request.urlretrieve('https://security.debian.org/debian-security/pool/updates/main/e/expat/${EXPAT_PACKAGE}', '/tmp/expat.deb')" \
    && echo 'cae5ded666d8573dd33060bc66fbac21eea8b101a6b56e2af8bd590fcfa65d79  /tmp/expat.deb' | sha256sum -c - \
    && dpkg -i /tmp/expat.deb && rm /tmp/expat.deb \
    && python -c "import rasterio"

RUN printf '%s\n' 'deb [check-valid-until=no] https://snapshot.debian.org/archive/debian/20260101T000000Z/ bookworm main' > /tmp/sab-debian.list \
    && apt-get -o Dir::Etc::sourcelist=/tmp/sab-debian.list -o Dir::Etc::sourceparts=- update \
    && apt-get -o Dir::Etc::sourcelist=/tmp/sab-debian.list -o Dir::Etc::sourceparts=- install -y --no-install-recommends libxrender1 libxext6 \
    && rm -rf /var/lib/apt/lists/* /tmp/sab-debian.list \
    && python -c 'from rdkit.Chem.Draw import rdMolDraw2D; import prolif'

RUN python -c "import urllib.request; urllib.request.urlretrieve('https://files.pythonhosted.org/packages/1f/46/4a008e5584211761bdbdad75149b09f3ee4ba64822dfe4b9054596d1a429/prolif-1.1.0-py3-none-any.whl', '/tmp/prolif-1.1.0-py3-none-any.whl')" \
    && echo '123e2ab6b445799b27bf7cd84ffcd373050b3802b9c80e38a6069fe698e52b6b  /tmp/prolif-1.1.0-py3-none-any.whl' | sha256sum -c - \
    && python -m pip install --no-index --no-deps --target /opt/prolif-1.1.0 /tmp/prolif-1.1.0-py3-none-any.whl \
    && rm /tmp/prolif-1.1.0-py3-none-any.whl \
    && PYTHONPATH=/opt/prolif-1.1.0 python -c "import prolif; assert prolif.__file__.startswith('/opt/prolif-1.1.0/')"

RUN python -m pip install --no-cache-dir --no-deps \
        'tf-keras @ https://files.pythonhosted.org/packages/21/8b/75f7572ec0273ed8da50bc19defe08aaaafcc15fda3407db53f49acec814/tf_keras-2.17.0-py3-none-any.whl#sha256=cc97717e4dc08487f327b0740a984043a9e0123c7a4e21206711669d3ec41c88' \
        'dgl @ https://files.pythonhosted.org/packages/e9/b0/daf76d5d813c6a7147850adb1a6b88244eba7986a56a09b4f3ff873516f5/dgl-1.1.3-cp310-cp310-manylinux1_x86_64.whl#sha256=157d3439f4d31cec3155ce2a224536d7ec0c90535b47d8f2790ef359af49ac95' \
    && TF_USE_LEGACY_KERAS=1 python -c 'import dgl; import tensorflow as tf; assert tf.keras.Model'
