# Execution sandbox for MT-MBPP: 17 language toolchains in one native arm64 image.
# The runtime contract (commands, flags, limits, security options) is in ../sandbox.py, which runs the image as
# mt-mbpp-sandbox:<first 12 hex digits of this file's SHA-256>:
#   docker build -t mt-mbpp-sandbox:$(shasum -a 256 Dockerfile | cut -c1-12) .
# Versions are pinned by the base image digest, a dated Ubuntu snapshot, or version plus SHA-256 for direct
# downloads; /opt/sandbox/versions.txt records what the build resolved.
FROM swift:6.4.0-resolute@sha256:bb6e5d5f2a97bc07cf8022c1a3e062f8164b8fd078698327ce2ab097e02a91dc

ARG DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    HOME=/tmp \
    TMPDIR=/tmp \
    PATH=/usr/local/go/bin:$PATH \
    GOTOOLCHAIN=local \
    DOTNET_CLI_TELEMETRY_OPTOUT=1 \
    DOTNET_NOLOGO=1 \
    DOTNET_EnableDiagnostics=0 \
    SWIFT_BACKTRACE=enable=no \
    SCALA_LIBRARY=/opt/scala3/maven2/org/scala-lang/scala-library/3.9.0/scala-library-3.9.0.jar

# Everything apt installs comes from the 2026-09-25 snapshot of the Ubuntu 26.04 archive. GNU coreutils replace
# 26.04's default uutils: shell programs are written against GNU behavior, and uutils `timeout` adds 0.1 s per call.
RUN rm -f /etc/apt/sources.list.d/ubuntu.sources \
 && printf '%s\n' 'Types: deb' 'URIs: https://snapshot.ubuntu.com/ubuntu/20260925T000000Z' \
      'Suites: resolute resolute-updates resolute-security' 'Components: main universe' \
      'Signed-By: /usr/share/keyrings/ubuntu-archive-keyring.gpg' > /etc/apt/sources.list.d/snapshot.sources \
 && apt-get update \
 && apt-get install -y --no-install-recommends --allow-remove-essential coreutils-from-gnu coreutils-from-uutils- \
 && apt-get install -y --no-install-recommends \
      ca-certificates curl \
      bc bsdextrautils gawk jq \
      gcc g++ \
      ghc libghc-regex-compat-dev libghc-regex-posix-dev libghc-regex-tdfa-dev libghc-split-dev \
        libghc-unordered-containers-dev libghc-vector-dev \
      openjdk-25-jdk-headless \
      dotnet-sdk-10.0 \
      nodejs \
      octave \
      php-cli php-mbstring \
      python3 \
      r-base-core \
      ruby \
      rustc cargo \
 && rm -rf /var/lib/apt/lists/*

# Go: the whole standard library is prebuilt into GOROOT so sandbox.py can compile and link with
# `go tool compile/link` and this importcfg, without a (cold, network-free) build cache.
RUN curl -fsSL -o /tmp/go.tgz https://go.dev/dl/go1.27.1.linux-arm64.tar.gz \
 && echo '3450b45a3f9ee8568792736a5c5e70a1f2e9b36c35a8f74958c03e51d7d92bec  /tmp/go.tgz' | sha256sum -c - \
 && tar xzf /tmp/go.tgz -C /usr/local && rm /tmp/go.tgz \
 && GODEBUG=installgoroot=all CGO_ENABLED=0 GOCACHE=/tmp/gocache go install std && rm -rf /tmp/gocache \
 && mkdir -p /opt/sandbox && cd /usr/local/go/pkg/linux_arm64 \
 && find . -name '*.a' | sed 's|^\./\(.*\)\.a$|packagefile \1=/usr/local/go/pkg/linux_arm64/\1.a|' | sort \
      > /opt/sandbox/go.importcfg

# TypeScript is transpiled (not type-checked) by esbuild, then run by node.
RUN curl -fsSL -o /tmp/esbuild.tgz https://registry.npmjs.org/@esbuild/linux-arm64/-/linux-arm64-0.28.2.tgz \
 && echo 'a96dbfa41d3ef5dbd1ef22b1c10d5187be9267e86093a870f06402a7ec931596  /tmp/esbuild.tgz' | sha256sum -c - \
 && tar xzf /tmp/esbuild.tgz -C /tmp && install -m 755 /tmp/package/bin/esbuild /usr/local/bin/esbuild \
 && rm -rf /tmp/package /tmp/esbuild.tgz

# Scala 3.9.0 LTS. Only the JVM compiler is used; the bundled native scala-cli launcher is removed.
RUN curl -fsSL -o /tmp/scala3.tgz \
      https://github.com/scala/scala3/releases/download/3.9.0/scala3-3.9.0-aarch64-pc-linux.tar.gz \
 && echo '1493eef9f2ce7adcef233bfead935a7220a06f1d2b687271d79bc250a78feabd  /tmp/scala3.tgz' | sha256sum -c - \
 && mkdir -p /opt/scala3 && tar xzf /tmp/scala3.tgz -C /opt/scala3 --strip-components=1 \
 && rm -f /tmp/scala3.tgz /opt/scala3/libexec/scala-cli

# JVM compiler startup dominates javac and scalac time on small files. Train JDK AOT caches on representative
# programs; the JVM flags here must match the javac and scalac commands in sandbox.py or the cache is not used.
COPY <<'EOF' /tmp/warm/Main.java
import java.util.*;
import java.util.regex.*;
import java.util.stream.*;

class Solution {
    static List<Integer> sorted(List<Integer> xs) {
        List<Integer> out = new ArrayList<>(xs);
        Collections.sort(out);
        return out;
    }
}

public class Main {
    public static void main(String[] args) {
        Map<String, Integer> counts = new HashMap<>();
        for (String w : "a b a".split(" ")) counts.merge(w, 1, Integer::sum);
        assert Solution.sorted(List.of(3, 1, 2)).equals(List.of(1, 2, 3));
        assert Pattern.compile("ab+").matcher("xabb").find();
        System.out.println(IntStream.range(0, 3).boxed().map(String::valueOf).collect(Collectors.joining(",")) + counts);
    }
}
EOF
COPY <<'EOF' /tmp/warm/Main.scala
import scala.collection.mutable
import scala.util.matching.Regex

case class Point(x: Int, y: Int)

def countWords(text: String): Map[String, Int] = text.split(" ").groupMapReduce(identity)(_ => 1)(_ + _)

def sortedPoints(ps: Seq[Point]): Seq[Point] = ps.sortBy(p => (p.x, p.y))

@main def testMain(): Unit = {
  val buf = mutable.ArrayBuffer(3, 1, 2)
  val pattern: Regex = "ab+".r
  assert(countWords("a b a") == Map("a" -> 2, "b" -> 1))
  assert(sortedPoints(Seq(Point(2, 1), Point(1, 2))).head == Point(1, 2))
  assert(pattern.findFirstIn("xabb").contains("abb"))
  println(s"${buf.sorted.mkString(",")} ${Option(1).map(_ + 1).getOrElse(0)}")
}
EOF
RUN cd /tmp/warm \
 && javac -J-XX:TieredStopAtLevel=1 -J-XX:+UseSerialGC -J-XX:-UsePerfData \
      -J-XX:AOTCacheOutput=/opt/sandbox/javac.aot -encoding UTF-8 Main.java \
 && java -ea -cp . Main \
 && java -XX:TieredStopAtLevel=1 -XX:+UseSerialGC -XX:-UsePerfData -XX:AOTCacheOutput=/opt/sandbox/scalac.aot \
      -cp /opt/scala3/lib/scala.jar -Dscala.expandjavacp=true -Dscala.usejavacp=true dotty.tools.dotc.Main \
      -d . Main.scala \
 && java -cp .:$SCALA_LIBRARY testMain \
 && chmod a+r /opt/sandbox/*.aot && rm -rf /tmp/warm

# Rust: a few popular crates (regex and the num family), prebuilt with the image's rustc for `rustc @args`.
COPY <<'EOF' /tmp/crates/Cargo.toml
[package]
name = "crates"
version = "0.0.0"
edition = "2021"

[dependencies]
num-bigint = "=0.4.8"
num-complex = "=0.4.6"
num-integer = "=0.1.47"
num-traits = "=0.2.19"
regex = "=1.13.1"
EOF
RUN cd /tmp/crates && mkdir src && echo 'fn main() {}' > src/main.rs \
 && CARGO_HOME=/tmp/cargo cargo build --release \
 && mkdir -p /opt/sandbox/rust && cp target/release/deps/*.rlib /opt/sandbox/rust/ \
 && { echo -L; echo dependency=/opt/sandbox/rust; \
      for crate in num_bigint num_complex num_integer num_traits regex; do \
        echo --extern; echo "$crate=$(ls /opt/sandbox/rust/lib$crate-*.rlib)"; done; } > /opt/sandbox/rust/args \
 && awk '/^name = /{n=$3} /^version = / && n && n != "\"crates\"" {print "crate", n, $3}' Cargo.lock | tr -d '"' \
      > /opt/sandbox/rust/crates.txt \
 && rm -rf /tmp/crates /tmp/cargo

# C#: Roslyn's csc against the reference pack (about 0.5 s, against 1.6 s for a cold build of a restored dotnet
# project), the implicit usings of `dotnet new console`, and a runtimeconfig so `dotnet Program.dll` runs the result.
COPY <<'EOF' /opt/sandbox/csharp/GlobalUsings.cs
global using global::System;
global using global::System.Collections.Generic;
global using global::System.IO;
global using global::System.Linq;
global using global::System.Net.Http;
global using global::System.Threading;
global using global::System.Threading.Tasks;
EOF
RUN ln -s /usr/lib/dotnet/sdk/*/Roslyn/bincore/csc /usr/local/bin/csc \
 && for dll in /usr/lib/dotnet/packs/Microsoft.NETCore.App.Ref/*/ref/net10.0/*.dll; do echo "-r:$dll"; done \
      > /opt/sandbox/csharp/references.rsp \
 && printf '{"runtimeOptions": {"tfm": "net10.0", "framework": {"name": "Microsoft.NETCore.App", "version": "%s"}}}\n' \
      "$(ls /usr/lib/dotnet/shared/Microsoft.NETCore.App)" > /opt/sandbox/csharp/Program.runtimeconfig.json

# Swift: prebuild the clang module cache for the common imports. Its modules record the cache path, so it is built
# at the /tmp path where sandbox.py restores a writable copy (other imports then build into the copy).
COPY <<'EOF' /tmp/warm/main.swift
import Dispatch
import Foundation
import FoundationEssentials
import Glibc
import RegexBuilder
import Synchronization

let regex = try! NSRegularExpression(pattern: "ab+")
print(regex.pattern, String(format: "%.2f", sqrt(2.0)), "xabb".firstMatch(of: try! Regex("ab+")) != nil)
EOF
RUN cd /tmp/warm && swiftc -module-cache-path /tmp/swift-module-cache -o main main.swift && ./main \
 && mv /tmp/swift-module-cache /opt/sandbox/swift-module-cache && chmod -R a+rX /opt/sandbox/swift-module-cache \
 && rm -rf /tmp/warm

# C++: precompile <bits/stdc++.h> for the exact flags sandbox.py uses (GCC ignores a PCH whose flags differ).
RUN header=$(echo /usr/include/aarch64-linux-gnu/c++/*/bits/stdc++.h) \
 && g++ -std=gnu++20 -x c++-header "$header" -o "$header.gch"

RUN { echo "base swift:6.4.0-resolute (Ubuntu 26.04.1), apt snapshot 20260925T000000Z"; \
      bash --version | head -1; timeout --version | head -1; gcc --version | head -1; g++ --version | head -1; \
      echo "dotnet sdk $(ls /usr/lib/dotnet/sdk) runtime $(ls /usr/lib/dotnet/shared/Microsoft.NETCore.App)"; \
      go version; ghc --version; java -version 2>&1 | head -1; echo "node $(node --version)"; \
      echo "esbuild $(esbuild --version)"; octave-cli --version | head -1; php --version | head -1; python3 --version; \
      R --version | head -1; ruby --version; rustc --version; cat /opt/sandbox/rust/crates.txt; \
      echo "scala $(sed -n 's/^version:=//p' /opt/scala3/VERSION)"; swiftc --version 2>&1 | head -1; \
    } > /opt/sandbox/versions.txt && cat /opt/sandbox/versions.txt
