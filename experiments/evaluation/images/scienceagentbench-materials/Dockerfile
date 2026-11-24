FROM python@sha256:db6ee5170ebce0f4d165af242e8e77addc08bd9ffc37e8b1815752cf027aa4f8
COPY requirements.lock /tmp/requirements.lock
RUN python -m pip install --no-cache-dir --prefer-binary --require-hashes -r /tmp/requirements.lock
WORKDIR /testbed
RUN mkdir -p /testbed/benchmark /testbed/pred_results \
    && ln -s /setup_files/datasets /testbed/benchmark/datasets
ARG EXPAT_PACKAGE=libexpat1_2.5.0-1+deb12u5_amd64.deb
RUN python -c "import urllib.request; urllib.request.urlretrieve('https://security.debian.org/debian-security/pool/updates/main/e/expat/${EXPAT_PACKAGE}', '/tmp/expat.deb')" \
    && echo 'cae5ded666d8573dd33060bc66fbac21eea8b101a6b56e2af8bd590fcfa65d79  /tmp/expat.deb' | sha256sum -c - \
    && dpkg -i /tmp/expat.deb && rm /tmp/expat.deb \
    && python -c "import numpy"

RUN printf '%s\n' 'deb [check-valid-until=no] https://snapshot.debian.org/archive/debian/20260101T000000Z/ bookworm main' > /tmp/sab-debian.list \
    && apt-get -o Dir::Etc::sourcelist=/tmp/sab-debian.list -o Dir::Etc::sourceparts=- update \
    && apt-get -o Dir::Etc::sourcelist=/tmp/sab-debian.list -o Dir::Etc::sourceparts=- install -y --no-install-recommends libxrender1 libxext6 \
    && rm -rf /var/lib/apt/lists/* /tmp/sab-debian.list \
    && python -c 'from rdkit.Chem.Draw import rdMolDraw2D'
