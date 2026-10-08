# Vendored SkyRL scorers

A task ships one of these modules to `/tests` under the same flat name and imports it by that name.
None of them imports another vendored module.

| File | Upstream | Revision | Note |
| --- | --- | --- | --- |
| `livecodebench.py` | [MarinSkyRL `skyrl-gym/skyrl_gym/envs/lcb/livecodebench.py`](https://github.com/marin-community/MarinSkyRL/blob/d8b6e8c163def3660e9d3072c1c174226a1709fa/skyrl-gym/skyrl_gym/envs/lcb/livecodebench.py) | `d8b6e8c163def3660e9d3072c1c174226a1709fa` (Ultra pin) | Unchanged. The one copy for SkyRL code rows and Ultra `code_gen`. The SkyRL pin's copy differs: it patches `sys.stdin` with a `StringIO` that also exposes `sys.stdin.buffer`, so stdin programs that read `sys.stdin.buffer` pass there and fail here. |
| `text_to_sql_scoring.py` | [MarinSkyRL `skyrl-gym/skyrl_gym/envs/text_to_sql/scoring.py`](https://github.com/marin-community/MarinSkyRL/blob/544d5d6f14116a06bde0209352585903133bd618/skyrl-gym/skyrl_gym/envs/text_to_sql/scoring.py) | `544d5d6f14116a06bde0209352585903133bd618` (SkyRL pin) | Unchanged. Identical at both pins. |
| `ifeval_utils.py` | [MarinSkyRL `skyrl-gym/skyrl_gym/envs/ifeval/utils.py`](https://github.com/marin-community/MarinSkyRL/blob/544d5d6f14116a06bde0209352585903133bd618/skyrl-gym/skyrl_gym/envs/ifeval/utils.py) | `544d5d6f14116a06bde0209352585903133bd618` (SkyRL pin) | Unchanged. The Ultra pin's copy lacks the `verifyit_enabled` argument of `compute_score`; the default (`False`) path is the same in both. |
| `apps_testing_util.py` | [hendrycks/apps `eval/testing_util.py`](https://github.com/hendrycks/apps/blob/b45c0ed78517a3a6492eb77b21cffbb79b1096f1/eval/testing_util.py) | `b45c0ed78517a3a6492eb77b21cffbb79b1096f1` | Patched for Python 3.12 by [`apps_testing_util_py312.patch`](apps_testing_util_py312.patch): `from pyext import RuntimeModule` becomes a local `RuntimeModule.from_string` that builds a `types.ModuleType`, registers it in `sys.modules` and `exec`s the program into it. pyext 0.7 fails to import on Python 3.11+ (`inspect.getargspec` was removed). |

`tests/test_scorers.py` reverses the patch and checks that the original under Python 3.10 with pyext 0.7
and the patched copy under the repository's Python return the same per-case results on APPS-format
problems: call-based, `Solution` class, stdin, a `__main__` guard, compile error, runtime error and
timeout.

License: the MarinSkyRL repository is Apache-2.0 (its `skyrl-gym` package metadata declares MIT).
`livecodebench.py` credits LiveCodeBench and rllm in its header. `ifeval_utils.py` keeps its note that
the constraint checkers come from allenai/open-instruct (Apache-2.0). hendrycks/apps is MIT:

```
Copyright (c) 2021 Dan Hendrycks

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
associated documentation files (the "Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the
following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial
portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT
LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO
EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
USE OR OTHER DEALINGS IN THE SOFTWARE.
```
