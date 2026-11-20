# SWE-rebench V2.1 repair — jvm group base (java/kotlin/scala/clojure rows).
# Covers java_11 (163) / java_17 (90) / java_21 (320) / kotlin-jdk-11 (191) /
# kotlin-jdk-21 (115) / scala_base (137) / clojure_base (53) rows (927 tasks).
# Upstream anchors: Dockerfile_java_11/17/21 (maven:3.9-eclipse-temurin-<v>), Dockerfile_kotlin
# (eclipse-temurin:11-jdk + kotlin compiler), Dockerfile_scala (debian + scala), Dockerfile_clojure
# (eclipse-temurin + clojure CLI).
# Per-version toolchains: Temurin JDK 11/17/21 tarballs side by side under
# /opt/toolchains/jdk-<version> (the task setup selects JAVA_HOME via /opt/java/current), plus
# kotlin compiler, scala distribution and clojure CLI on PATH.
# Interpreter provisioning (repair, disclosed): apt python3 (jammy ships 3.10).

# The temurin-21 default tag moved to ubuntu noble and its python3.12 dpkg state breaks under
# apt in-region; the upstream declared java bases were jammy (eclipse-temurin:21-jdk-jammy in
# the census), so the repair pins the jammy maven variant (clean dpkg state, python 3.10).
FROM maven:3.9-eclipse-temurin-21-jammy@sha256:7f8422bd81f19b9035007a75e2f82272b09af64db08848d7d4e93302466e8b6f

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
# Locale: the upstream jvm images set LANG=C.UTF-8 + LANGUAGE=en_US:en (read from
# swerebenchv2/1c-syntax-bsl-language-server@99eea044). Without it the JVM's sun.jnu.encoding
# degrades to ASCII and the BSL rows' Cyrillic-filename tests fail en masse (91/757 in the
# diag-v22 capture).
ENV LANG=C.UTF-8
ENV LANGUAGE=en_US:en

RUN apt-get update && apt-get install -y --no-install-recommends \
    wget curl git unzip zip python3 \
    && rm -rf /var/lib/apt/lists/*

# Declared interpreter provisioning guard.
RUN test "$(python3 -c 'import sys; print("%d%02d" % sys.version_info[:2])')" -ge 309 \
    || { echo "python3 >= 3.9 required on PATH" >&2; exit 1; }

# Per-version Temurin JDKs (adoptium API, pinned releases).
RUN set -eux; \
    arch="$(dpkg --print-architecture)"; \
    case "$arch" in \
      amd64) suffix="x64" ;; \
      arm64) suffix="aarch64" ;; \
      *) echo "unsupported arch: $arch" >&2; exit 1 ;; \
    esac; \
    for version in 11 17; do \
      url="https://api.adoptium.net/v3/binary/latest/${version}/ga/linux/${suffix}/jdk/hotspot/normal/eclipse"; \
      mkdir -p "/opt/toolchains/jdk-${version}"; \
      curl -fsSL "$url" -o /tmp/jdk.tar.gz; \
      tar -xzf /tmp/jdk.tar.gz -C "/opt/toolchains/jdk-${version}" --strip-components=1; \
      rm /tmp/jdk.tar.gz; \
    done
# JDK 21 is the base image's own java; expose it under the toolchains tree as well. NO mkdir:
# jdk-21 must be a symlink itself — with a pre-created real directory, `ln -sfn` drops the
# openjdk symlink INSIDE it and JAVA_HOME=/opt/java/current resolves to an empty dir (the
# diag-v21 evidence: gradle/mvn "JAVA_HOME is set to an invalid directory").
RUN ln -sfn /opt/java/openjdk /opt/toolchains/jdk-21
RUN test -x /opt/toolchains/jdk-11/bin/java && test -x /opt/toolchains/jdk-17/bin/java \
    && test -x /opt/toolchains/jdk-21/bin/java \
    || { echo "jdk toolchains missing" >&2; exit 1; }

# Kotlin compiler (upstream kotlin recipe installs the compiler release; pinned).
ARG KOTLIN_VERSION=2.0.21
RUN set -eux; \
    curl -fsSL "https://github.com/JetBrains/kotlin/releases/download/v${KOTLIN_VERSION}/kotlin-compiler-${KOTLIN_VERSION}.zip" -o /tmp/kotlin.zip; \
    unzip -q /tmp/kotlin.zip -d /opt/toolchains; \
    rm /tmp/kotlin.zip; \
    ln -sfn /opt/toolchains/kotlinc/bin/kotlin /usr/local/bin/kotlin \
    && ln -sfn /opt/toolchains/kotlinc/bin/kotlinc /usr/local/bin/kotlinc

# Scala distribution (pinned 2.13; GitHub release mirror, the lightbend CDN refuses builds).
ARG SCALA_VERSION=2.13.15
RUN set -eux; \
    curl -fsSL "https://github.com/scala/scala/releases/download/v${SCALA_VERSION}/scala-${SCALA_VERSION}.tgz" -o /tmp/scala.tgz; \
    tar -xzf /tmp/scala.tgz -C /opt/toolchains; \
    rm /tmp/scala.tgz; \
    ln -sfn "/opt/toolchains/scala-${SCALA_VERSION}/bin/scala" /usr/local/bin/scala \
    && ln -sfn "/opt/toolchains/scala-${SCALA_VERSION}/bin/scalac" /usr/local/bin/scalac

# Clojure CLI (upstream clojure recipe). The cli.clojure.org install script exited 0
# WITHOUT installing in-region (the first build shipped no clj/clojure binaries — content
# verified), so the repair downloads the pinned github release installer directly with a
# hard guard. Version 1.11.4.1474 verified reachable (the 1.12.x tags do not exist
# on the brew-install release feed).
RUN set -eux; \
    curl -fsSL https://github.com/clojure/brew-install/releases/download/1.11.4.1474/linux-install.sh -o /tmp/clojure-install.sh; \
    bash /tmp/clojure-install.sh; \
    rm -f /tmp/clojure-install.sh; \
    command -v clojure && command -v clj \
    || { echo "clojure CLI missing after install" >&2; exit 1; }

# Leiningen (the lein rows' upstream images ship /usr/local/bin/lein — verified from
# amperity-lein-monolith@62f2719e). Pinned 2.11.2 script + self-install; the standalone jar
# release asset URL HEAD-verified 200 (the -standalone.zip asset does not exist).
RUN set -eux; \
    curl -fsSL https://raw.githubusercontent.com/technomancy/leiningen/2.11.2/bin/lein -o /usr/local/bin/lein; \
    chmod +x /usr/local/bin/lein; \
    lein version 2>&1 | grep -q Leiningen \
    || { echo "lein missing after self-install" >&2; exit 1; }

# Default JAVA_HOME on 21 (base java); the task setup re-points per row.
RUN ln -sfn /opt/toolchains/jdk-21 /opt/java/current
ENV JAVA_HOME="/opt/java/current"
ENV PATH="${JAVA_HOME}/bin:${PATH}"

RUN mkdir -p /logs /output && chmod 777 /logs /output
