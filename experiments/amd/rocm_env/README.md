# AMD ROCm training environment

`build_env.sh` builds the Marin training environment for AMD Instinct GPUs
from the repository alone, on a new Linux x86_64 host: uv 0.13.1, CPython
3.12.3, a checkout at a given commit, and a venv synced from that commit's
`uv.lock` with the `rocm` extra (JAX 0.12.0, jax-rocm10-plugin and pjrt
0.12.0, AMD TheRock rocm-sdk 10.1.0 wheels with RCCL 2.30.7, Triton 3.6.0,
jax-triton 0.3.1). The host needs git, curl, internet access and the amdgpu
driver; it does not need a system ROCm install.

```bash
experiments/amd/rocm_env/build_env.sh --prefix /work/marin-env \
    --repo https://github.com/marin-community/marin.git --commit <sha> \
    --anp-plugin /opt/rocm-7.2.0/lib/librccl-anp.so
```

The build fails if any of those packages, Python or RCCL resolve to another
version. It writes `DIR/manifest` with the commit, host, full package list and
the net plugin's checksum. On a host with GPUs it then runs `selftest.sh`,
which checks the device count, an exact all-reduce over the node's GPUs, that
JAX loads the venv's RCCL rather than a system copy, a jax_triton kernel, that
RCCL loads the ANP net plugin when one is given, and a small jitted training
step. On a login node without GPUs, run the self-test later on a GPU node:

```bash
DIR/marin/experiments/amd/rocm_env/selftest.sh DIR/venv/bin/python --expect-devices 8 \
    --anp-plugin /opt/rocm-7.2.0/lib/librccl-anp.so
```

`--anp-plugin` names the host's Pensando ANP RCCL net plugin. It is a host
component tied to the NIC driver, so the build records its path and checksum
and does not copy it.
