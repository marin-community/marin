FROM nvcr.io/nvidia/pytorch:24.09-py3

RUN apt-get update --yes --quiet \
    && apt-get upgrade --yes --quiet \
    && apt-get install git wget curl unzip software-properties-common --yes --quiet

RUN pip install --upgrade pip

RUN git clone https://github.com/mlfoundations/evalchemy.git /evalchemy

RUN pip install -I --timeout 2000 -r https://raw.githubusercontent.com/bigcode-project/bigcodebench/refs/heads/main/Requirements/requirements.txt 

RUN pip install --no-cache-dir ninja packaging psutil 
RUN pip install --no-cache-dir -I --timeout 2000 -r https://raw.githubusercontent.com/bigcode-project/bigcodebench/refs/heads/main/Requirements/requirements-eval.txt 
RUN cd /evalchemy && pip install --no-cache-dir -e . && pip install -e eval/chat_benchmarks/alpaca_eval
RUN pip install lm-eval[vllm]@git+https://github.com/EleutherAI/lm-evaluation-harness@v0.4.12
RUN pip install "opencv-python-headless<4.3"
RUN MAX_JOBS=10 pip install -U flash-attn --no-build-isolation

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update -yqq && apt-get install -yqq default-jdk-headless 

RUN apt-get install -yqq golang-go 
RUN apt-get install -yqq  php
RUN apt-get install -yqq  ruby 
RUN apt-get install -yqq  lua5.3 
RUN apt-get install -yqq  r-base \
    rustc \
    scala \
    ocaml-interp \
    ocaml \
    ghc \
    libtest-deep-perl \
    wget \
    lua-unit \
    aspnetcore-runtime-6.0

RUN apt-get install -yqq libtest-deep-perl
RUN apt-get install -yqq wget

# JS/TS
RUN curl -fsSL https://deb.nodesource.com/setup_current.x | bash -
RUN apt-get install -y nodejs
RUN npm install -g typescript

# Dlang
RUN wget https://netcologne.dl.sourceforge.net/project/d-apt/files/d-apt.list -O /etc/apt/sources.list.d/d-apt.list
RUN apt-get update --allow-insecure-repositories
RUN apt-get -y --allow-unauthenticated install --reinstall d-apt-keyring
RUN apt-get update && apt-get install -yqq dmd-compiler dub

# C#
RUN apt install gnupg ca-certificates
RUN apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys 3FA7E0328081BFF6A14DA29AA6A19B38D3D831EF
RUN echo "deb https://download.mono-project.com/repo/ubuntu stable-focal main" | tee /etc/apt/sources.list.d/mono-official-stable.list
RUN apt update --allow-insecure-repositories
RUN apt install -yqq mono-devel

# F#
RUN apt-get install -y dotnet6

# Post-processing

# Julia
RUN curl https://julialang-s3.julialang.org/bin/linux/x64/1.8/julia-1.8.2-linux-x86_64.tar.gz | tar xz
ENV PATH="/julia-1.8.2/bin:${PATH}"
# Swift
RUN curl https://download.swift.org/swift-5.7-release/ubuntu2204/swift-5.7-RELEASE/swift-5.7-RELEASE-ubuntu22.04.tar.gz | tar xz
ENV PATH="/swift-5.7-RELEASE-ubuntu22.04/usr/bin:${PATH}"
# Javatuples
RUN mkdir /usr/multiple && wget https://repo.mavenlibs.com/maven/org/javatuples/javatuples/1.2/javatuples-1.2.jar -O /usr/multiple/javatuples-1.2.jar

# Luau
RUN wget https://github.com/Roblox/luau/releases/download/0.594/luau-ubuntu.zip -O /tmp/luau.zip && unzip /tmp/luau.zip -d /bin/

# Elixir
# Install Erlang
RUN mkdir -p /erlang && \
    cd /erlang && \
    wget -nv -O erlang.tar.gz https://builds.hex.pm/builds/otp/ubuntu-22.04/OTP-27.0.tar.gz && \
    tar xzf erlang.tar.gz --strip-components=1 && \
    ./Install -minimal "$(pwd)"

# Erlang runtime dependencies, see https://github.com/hexpm/bob/blob/4fe43eb9853bb95dbfe276957bd7d3f931a451b3/priv/scripts/docker/erlang-ubuntu-jammy.dockerfile
RUN apt-get update && \
    apt-get -y --no-install-recommends install \
    ca-certificates \
    libodbc1 \
    libssl3 \
    libsctp1

# Install Elixir
RUN mkdir -p /elixir && \
    cd /elixir && \
    wget -nv -O elixir.zip https://builds.hex.pm/builds/elixir/v1.17.2-otp-27.zip && \
    unzip -q elixir.zip

ENV PATH="/erlang/bin:/elixir/bin:${PATH}"
ENV LANG=C.UTF-8

# Dafny
# https://github.com/dafny-lang/dafny/releases/download/v4.3.0/dafny-4.3.0-x64-ubuntu-20.04.zip
RUN mkdir -p /root/dafny
RUN wget https://github.com/dafny-lang/dafny/releases/download/v4.3.0/dafny-4.3.0-x64-ubuntu-20.04.zip -O /tmp/dafny.zip && unzip /tmp/dafny.zip -d /root/dafny/ && ln -s /root/dafny/dafny/dafny /bin/dafny

# Clojure - https://clojure.org/guides/install_clojure#_linux_instructions
RUN curl -L -O https://github.com/clojure/brew-install/releases/latest/download/linux-install.sh
RUN chmod +x linux-install.sh
RUN ./linux-install.sh --prefix /clojure
ENV PATH="/clojure/bin:${PATH}"
RUN clojure -P

# Dart
RUN wget -qO- https://dl-ssl.google.com/linux/linux_signing_key.pub | gpg  --dearmor -o /usr/share/keyrings/dart.gpg
RUN echo 'deb [signed-by=/usr/share/keyrings/dart.gpg arch=amd64] https://storage.googleapis.com/download.dartlang.org/linux/debian stable main' | tee /etc/apt/sources.list.d/dart_stable.list
RUN apt-get update -yqq && apt-get install -yqq dart

# Coq
RUN apt-get install -yqq coq

# Lean
RUN wget https://github.com/leanprover/lean4/releases/download/v4.6.0-rc1/lean-4.6.0-rc1-linux.zip -O /tmp/lean.zip && unzip /tmp/lean.zip -d /root/lean/ && ln -s /root/lean/bin/lean /bin/lean


WORKDIR /app

