FROM python@sha256:db6ee5170ebce0f4d165af242e8e77addc08bd9ffc37e8b1815752cf027aa4f8
RUN python -m pip --version
COPY requirements.lock /tmp/requirements.lock
RUN python -m pip install --no-cache-dir --no-deps --prefer-binary --require-hashes -r /tmp/requirements.lock
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

RUN python -c "import urllib.request; urllib.request.urlretrieve('https://codeload.github.com/wjm41/molplotly/zip/60e37720998b5a02fc88c08c259eaeb755852320','/tmp/molplotly.zip')" \
    && echo '69dd4183634c273f8dbbf90b99e7a9b06731ec5922ad39b551c4cb5dd61e1b8e  /tmp/molplotly.zip' | sha256sum -c - \
    && python -m pip install --no-cache-dir --no-deps /tmp/molplotly.zip && rm /tmp/molplotly.zip

RUN python -c "import urllib.request; urllib.request.urlretrieve('https://codeload.github.com/martin-sicho/papyrus-scaffold-visualizer/zip/32fc428e6d7d457762709c02425a1b50652eafda','/tmp/papyrus-scaffold-visualizer.zip')" \
    && echo '410997624e8849cf7aa6e5c5fc9d090d3dca205fb1e61d42df42aef2eca78ecc  /tmp/papyrus-scaffold-visualizer.zip' | sha256sum -c - \
    && python -m pip install --no-cache-dir --no-deps /tmp/papyrus-scaffold-visualizer.zip && rm /tmp/papyrus-scaffold-visualizer.zip
RUN python -m pip install --no-cache-dir --no-deps 'setuptools @ https://files.pythonhosted.org/packages/a3/dc/17031897dae0efacfea57dfd3a82fdd2a2aeb58e0ff71b77b87e44edc772/setuptools-80.9.0-py3-none-any.whl#sha256=062d34222ad13e0cc312a4c02d73f059e86a4acbfbdea8f8f76b28c99f306922'
RUN python -c 'from scaffviz.depiction.plot import Plot; from scaffviz.clustering.manifold import TSNE; from qsprpred.data import MoleculeTable; from DeepPurpose import utils; import chemprop'
