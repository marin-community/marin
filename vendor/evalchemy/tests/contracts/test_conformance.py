"""All sources use the data-only interface and the same execution boundary."""

from pathlib import Path

import pytest

from eval.contracts.conformance import build_task_contract_registry, discover_task_contracts, validate_custom_benchmark_class
from eval.contracts.task_outcome import TaskRoute
from eval.sources import SOURCES
from eval.task import BaseBenchmark


def test_every_registered_name_has_one_complete_contract():
    root = Path("eval/chat_benchmarks")
    packaged_modules = {path.parent.name for path in root.glob("*/eval_instruct.py")}
    registered_modules = {module.split(".")[-2] for module, _class in SOURCES.values()}
    assert packaged_modules == registered_modules
    contracts = discover_task_contracts(Path("eval/lm_eval_tasks"))
    assert set(SOURCES) == {contract.task_name for contract in contracts if contract.route is TaskRoute.CUSTOM}
    assert any(contract.route is TaskRoute.LM_EVAL for contract in contracts)


def test_source_resource_declarations_are_validated():
    class Source(BaseBenchmark):
        def iter_specs(self, options):
            return iter(())

    Source.RESOURCE_REQUIREMENTS = []
    with pytest.raises(TypeError, match="must be a tuple"):
        validate_custom_benchmark_class(Source, BaseBenchmark)


def test_task_cannot_belong_to_both_registries():
    with pytest.raises(ValueError, match="both .* and lm-eval"):
        build_task_contract_registry(["collision"], ["collision"])
