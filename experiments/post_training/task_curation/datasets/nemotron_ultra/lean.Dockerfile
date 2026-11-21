# Lean 4 and Mathlib for grading Nemotron Ultra Lean proofs, laid out like the NeMo Skills sandbox
# (NVIDIA/NeMo-Skills dockerfiles/Dockerfile.sandbox since 2fd68561, 2024-10-24): the toolchain
# leanprover/lean4:v4.12.0 and Mathlib v4.12.0 in the project /lean4/my_project, whose path the
# vendored lean_feedback.parse_error expects in compiler diagnostics. MarinSkyRL pins the sandbox
# image by digest without recording its NeMo Skills revision; every NeMo Skills sandbox since that
# commit bundles these versions. Keep LEAN_PROJECT_DIR and LAKE in local_lean_sandbox.py in step.
#
# Mathlib 809c3fb3 is the commit tag v4.12.0 names; elan is pinned so the proxies are reproducible.
ARG LEAN_TOOLCHAIN=leanprover/lean4:v4.12.0
ARG MATHLIB_COMMIT=809c3fb3b5c8f5d7dace56e200b426187516535a
ARG ELAN_VERSION=4.2.4
ENV ELAN_HOME=/opt/elan
ENV PATH=/opt/elan/bin:$PATH
RUN curl -fsSL "https://github.com/leanprover/elan/releases/download/v${ELAN_VERSION}/elan-x86_64-unknown-linux-gnu.tar.gz" \
      | tar -xz -C /tmp elan-init \
    && /tmp/elan-init -y --no-modify-path --default-toolchain "${LEAN_TOOLCHAIN}" \
    && rm /tmp/elan-init
# `lake new` writes the lakefile and lean-toolchain NeMo Skills appends Mathlib to; `cache get` fetches
# Mathlib's prebuilt oleans so the build compiles only the project. The check compiles one Mathlib
# theorem as an unprivileged user from the world-writable scratch directory the grader writes to.
# `lake env` asks git for each package's remote URL; without safe.directory, git refuses another
# user's checkout and Lake tries to clone Mathlib again.
RUN git config --system safe.directory '*' \
    && mkdir -p /lean4 && cd /lean4 \
    && lake new my_project \
    && cd my_project \
    && echo "${LEAN_TOOLCHAIN}" > lean-toolchain \
    && echo "require mathlib from git \"https://github.com/leanprover-community/mathlib4\" @ \"${MATHLIB_COMMIT}\"" \
      >> lakefile.lean \
    && lake exe cache get \
    && lake build \
    && mkdir -m 1777 scratch \
    && chmod -R a+rX /opt/elan /lean4 \
    && printf 'import Mathlib\ntheorem check (n : Nat) : n + 0 = n := by simp\n' > scratch/check.lean \
    && setpriv --reuid=65534 --regid=65534 --clear-groups \
      lake env --dir /lean4/my_project lean /lean4/my_project/scratch/check.lean \
    && rm scratch/check.lean
