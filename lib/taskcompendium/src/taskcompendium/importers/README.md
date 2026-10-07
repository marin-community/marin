# Task importers

Importers translate supplied source formats into private TaskSpecs with source
provenance. [harbor.py](harbor.py) reads a Harbor task directory;
[swe.py](swe.py) imports SWE instances; [nemo_predicted_action.py](nemo_predicted_action.py)
decodes NeMo requests and typed expected actions; [tasktrove/](tasktrove/README.md)
reads TaskTrove archives and imports MCQ tasks.

Callers acquire inputs and provide their pinned identity. An importer preserves
public instructions and source grading requirements, or rejects an unsupported
contract. It performs no inference or job submission and does not repair source
tests. [Dataset-family policies](../datasets/README.md) and experiment declarations
compose these readers into ingestion pipelines.
