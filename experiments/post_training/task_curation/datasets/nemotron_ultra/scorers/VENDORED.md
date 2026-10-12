# Vendored Nemotron Ultra scorers

A task ships these files to `/tests` at the same relative paths, so `skyrl_gym.envs.nemotron_ultra.*`
imports resolve unchanged. Upstream files are copied byte for byte from
[marin-community/MarinSkyRL](https://github.com/marin-community/MarinSkyRL) at the Ultra pin
`d8b6e8c163def3660e9d3072c1c174226a1709fa`, path `skyrl-gym/skyrl_gym/envs/<path>`.

| File | Upstream | Note |
| --- | --- | --- |
| `skyrl_gym/__init__.py`, `skyrl_gym/envs/__init__.py` | none | Ours: empty package markers. Upstream's versions register the Gym environments and import the full SkyRL stack. |
| `skyrl_gym/envs/aime/utils.py` | [`aime/utils.py`](https://github.com/marin-community/MarinSkyRL/blob/d8b6e8c163def3660e9d3072c1c174226a1709fa/skyrl-gym/skyrl_gym/envs/aime/utils.py) | Unchanged. `answer_extraction` imports its boxed-answer helpers. Upstream has no `aime/__init__.py`. |
| `skyrl_gym/envs/nemotron_ultra/__init__.py` | [`nemotron_ultra/__init__.py`](https://github.com/marin-community/MarinSkyRL/blob/d8b6e8c163def3660e9d3072c1c174226a1709fa/skyrl-gym/skyrl_gym/envs/nemotron_ultra/__init__.py) | Unchanged. |
| `skyrl_gym/envs/nemotron_ultra/answer_extraction.py` | [`nemotron_ultra/answer_extraction.py`](https://github.com/marin-community/MarinSkyRL/blob/d8b6e8c163def3660e9d3072c1c174226a1709fa/skyrl-gym/skyrl_gym/envs/nemotron_ultra/answer_extraction.py) | Unchanged. |
| `skyrl_gym/envs/nemotron_ultra/calendar.py` | [`nemotron_ultra/calendar.py`](https://github.com/marin-community/MarinSkyRL/blob/d8b6e8c163def3660e9d3072c1c174226a1709fa/skyrl-gym/skyrl_gym/envs/nemotron_ultra/calendar.py) | Unchanged. |
| `skyrl_gym/envs/nemotron_ultra/code_gen.py` | [`nemotron_ultra/code_gen.py`](https://github.com/marin-community/MarinSkyRL/blob/d8b6e8c163def3660e9d3072c1c174226a1709fa/skyrl-gym/skyrl_gym/envs/nemotron_ultra/code_gen.py) | Unchanged. A code task also ships [`../../skyrl/scorers/livecodebench.py`](../../skyrl/scorers/livecodebench.py) as `skyrl_gym/envs/lcb/livecodebench.py`. |
| `skyrl_gym/envs/nemotron_ultra/format_verification.py` | [`nemotron_ultra/format_verification.py`](https://github.com/marin-community/MarinSkyRL/blob/d8b6e8c163def3660e9d3072c1c174226a1709fa/skyrl-gym/skyrl_gym/envs/nemotron_ultra/format_verification.py) | Unchanged. |
| `skyrl_gym/envs/nemotron_ultra/instruction_following.py` | [`nemotron_ultra/instruction_following.py`](https://github.com/marin-community/MarinSkyRL/blob/d8b6e8c163def3660e9d3072c1c174226a1709fa/skyrl-gym/skyrl_gym/envs/nemotron_ultra/instruction_following.py) | Unchanged. Imports `verifiable_instructions` from the grader image. |
| `skyrl_gym/envs/nemotron_ultra/mcqa.py` | [`nemotron_ultra/mcqa.py`](https://github.com/marin-community/MarinSkyRL/blob/d8b6e8c163def3660e9d3072c1c174226a1709fa/skyrl-gym/skyrl_gym/envs/nemotron_ultra/mcqa.py) | Unchanged. |
| `skyrl_gym/envs/nemotron_ultra/rdkit_chemistry.py` | [`nemotron_ultra/rdkit_chemistry.py`](https://github.com/marin-community/MarinSkyRL/blob/d8b6e8c163def3660e9d3072c1c174226a1709fa/skyrl-gym/skyrl_gym/envs/nemotron_ultra/rdkit_chemistry.py) | Unchanged. Compares exact answers and does not import RDKit. |
| `skyrl_gym/envs/nemotron_ultra/structured_outputs.py` | [`nemotron_ultra/structured_outputs.py`](https://github.com/marin-community/MarinSkyRL/blob/d8b6e8c163def3660e9d3072c1c174226a1709fa/skyrl-gym/skyrl_gym/envs/nemotron_ultra/structured_outputs.py) | Unchanged. |
| `skyrl_gym/envs/nemotron_ultra/tool_call.py` | [`nemotron_ultra/tool_call.py`](https://github.com/marin-community/MarinSkyRL/blob/d8b6e8c163def3660e9d3072c1c174226a1709fa/skyrl-gym/skyrl_gym/envs/nemotron_ultra/tool_call.py) | Unchanged. |

The NVARC scorer and its sandbox client live with the ARC family in [`../../arc/scorers/`](../../arc/scorers/VENDORED.md).

License: the `nemotron_ultra` files keep NVIDIA's `SPDX-License-Identifier: Apache-2.0` headers, and
`aime/utils.py` keeps its Apache-2.0 header. The MarinSkyRL repository is Apache-2.0; its `skyrl-gym`
package metadata declares MIT.
