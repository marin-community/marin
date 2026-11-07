# InferenceX components

Knotmeter’s frontend is distributed under GNU GPL version 3. It adapts chart
components and calculations from
[InferenceX](https://github.com/SemiAnalysisAI/InferenceX-app), revision
`315128ce4e49c8a1ec1aa67eafa8f2de26e4c1f9`, copyright SemiAnalysisAI and
its contributors. The adapted modules live in `src/upstream/`, `ingest/upstream/` and
`ingest/replay.py`; the GPL-marked metric/history/ruler helpers also derive from that pin. They retain the
GNU GPL version 3 license in `LICENSE.inferencex`.

Marina integration uses a packaged React browser build, relative API requests
and hash navigation. The source pin identifies the original calculations;
the application adapts their inputs to saved Marin collector artifacts.
