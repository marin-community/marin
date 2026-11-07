# Lean, Mathlib and Prime Intellect's proof checker for the Lean proof grader (lean_grade.py beside
# this file), at the toolchain LEAN_TOOLCHAIN pins; lean_grade.LEAN_VERSION must name the same release.
#
# Every download is pinned: elan by the SHA-256 of its release archive, Mathlib by commit (one whose
# lean-toolchain is LEAN_TOOLCHAIN, which the build checks) with the prebuilt object files that
# `lake exe cache get` fetches, and the checker by commit: `lean-export` and `lean-verify` from
# PrimeIntellect-ai/prime-envs (packages/lean_common/image/checker, Apache-2.0; its LICENSE is copied
# beside the build). The build makes one change to the checker: `sameExpr` no longer skips expressions
# that are already equal, so each one is compared through its own module's matchers, which
# `unchangedDeps` relies on. Grading runs `lean` directly with the LEAN_PATH recorded here, so it needs
# neither the network nor Lake.
ARG ELAN_VERSION=4.2.4
ARG ELAN_SHA256=42b94d4244e8353142c456ec0e4ca6528fd898a6c604d4059f494e706e431f63
ARG MATHLIB_COMMIT=d13f23b723b8a846827a245b89c10fc7d3f11612
ARG LEAN_TOOLCHAIN=leanprover/lean4:v4.34.1
ARG PRIME_ENVS_COMMIT=15e4e9d2f7d52d8d725a2668e483cf51a067e5ba
ENV ELAN_HOME=/opt/elan
ENV PATH=/opt/elan/bin:$PATH
RUN curl -fsSL "https://github.com/leanprover/elan/releases/download/v${ELAN_VERSION}/elan-x86_64-unknown-linux-gnu.tar.gz" \
      -o /tmp/elan.tar.gz \
    && echo "${ELAN_SHA256}  /tmp/elan.tar.gz" | sha256sum -c - \
    && tar -xzf /tmp/elan.tar.gz -C /tmp elan-init \
    && /tmp/elan-init -y --default-toolchain none --no-modify-path \
    && rm /tmp/elan.tar.gz /tmp/elan-init
# The Mathlib cache unpacks object files readable by root only; they are opened up in the same layer,
# since a later chmod would store a second copy of Mathlib.
RUN git init -q /opt/lean/mathlib \
    && cd /opt/lean/mathlib \
    && git fetch -q --depth 1 https://github.com/leanprover-community/mathlib4.git "${MATHLIB_COMMIT}" \
    && git checkout -q FETCH_HEAD \
    && test "$(cat lean-toolchain)" = "${LEAN_TOOLCHAIN}" \
    && elan default "${LEAN_TOOLCHAIN}" \
    && lake exe cache get \
    && lake build Mathlib \
    && lake env printenv LEAN_PATH > /opt/lean/lean-path \
    && readlink -f "$(elan which lean)" > /opt/lean/lean-bin \
    && rm -rf .git \
    && chmod -R a+rX /opt/elan /opt/lean
RUN git init -q /tmp/prime-envs \
    && cd /tmp/prime-envs \
    && git fetch -q --depth 1 https://github.com/PrimeIntellect-ai/prime-envs.git "${PRIME_ENVS_COMMIT}" \
    && git checkout -q FETCH_HEAD \
    && cp -r packages/lean_common/image/checker /opt/lean/verify \
    && cp LICENSE /opt/lean/verify/LICENSE \
    && cd /opt/lean/verify \
    && rm -rf /tmp/prime-envs \
    && sed -i 's/^  a == b || proofFree ma (unfoldMatchers ma a) == proofFree mb (unfoldMatchers mb b)$/  proofFree ma (unfoldMatchers ma a) == proofFree mb (unfoldMatchers mb b)/' Main.lean \
    && test "$(grep -c '^  proofFree ma (unfoldMatchers ma a) == proofFree mb (unfoldMatchers mb b)$' Main.lean)" = 1 \
    && ! grep -q 'a == b || proofFree' Main.lean \
    && echo "${LEAN_TOOLCHAIN}" > lean-toolchain \
    && lake build \
    && chmod -R a+rX /opt/lean/verify
# An unprivileged user compiles a Mathlib file offline, as the grader's candidate step does.
RUN printf 'import Mathlib\ntheorem check (n : Nat) : n + 0 = n := by simp\n' > /tmp/check.lean \
    && chmod 644 /tmp/check.lean \
    && cd /tmp \
    && setpriv --reuid=65534 --regid=65534 --clear-groups \
      env LEAN_PATH="$(cat /opt/lean/lean-path)" "$(cat /opt/lean/lean-bin)" /tmp/check.lean \
    && rm /tmp/check.lean
