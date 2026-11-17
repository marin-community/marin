The Papyrus normalization and comparison follow `benchmark/eval_programs/eval_papyrus.py`
in the official ScienceAgentBench Verified artifact bundle (SHA-256
`46e715d3b2196d459d2dff52aa487f506a95ec44b44262e82208d086ea879610`).
The source project is [ScienceAgentBench](https://github.com/OSU-NLP-Group/ScienceAgentBench/tree/c26e151ed601ba109dc4d35e057ff8e73fec469d).
Its MIT license is retained in `ScienceAgentBench.LICENSE`.

The adaptation separates candidate pickle decoding from gold-data access. The
grader receives normalized JSON rows and preserves the original gold-prefix
comparison. Benchmark data and reference programs are not distributed here.
