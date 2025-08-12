import os
from dataclasses import dataclass

from marin.execution.dynamic_executor import DynamicExecutor, use_step, prepare_step
from marin.execution.executor import Executor, ExecutorStep, InputName


def test_dynamic_basic(tmp_path):
    with DynamicExecutor(prefix=str(tmp_path)) as ex:
        with prepare_step("step1") as out1:
            open(os.path.join(out1.path, "a.txt"), "w").write("hi")
        with use_step("step1") as inp, prepare_step("step2") as out2:
            data = open(os.path.join(inp.path, "a.txt"), "r").read()
            open(os.path.join(out2.path, "b.txt"), "w").write(data)

    s1 = ex.steps["step1"]
    s2 = ex.steps["step2"]
    assert s2.dependencies == [s1]
    assert os.path.exists(os.path.join(s1.path, "a.txt"))
    assert os.path.exists(os.path.join(s2.path, "b.txt"))


def test_dynamic_static_compatibility(tmp_path):
    with DynamicExecutor(prefix=str(tmp_path)) as dex:
        with prepare_step("step1"):
            pass
        with use_step("step1"), prepare_step("step2"):
            pass

    @dataclass
    class Step1Cfg:
        pass

    def s1(cfg: Step1Cfg):
        pass

    @dataclass
    class Step2Cfg:
        inputs: list[InputName]

    def s2(cfg: Step2Cfg):
        pass

    step1 = ExecutorStep("step1", s1, Step1Cfg())
    step2 = ExecutorStep("step2", s2, Step2Cfg(inputs=[step1]))
    ex = Executor(prefix=str(tmp_path), executor_info_base_path=str(tmp_path))
    ex.run([step2], dry_run=True)

    assert ex.output_paths[step1] == dex.steps["step1"].path
    assert ex.output_paths[step2] == dex.steps["step2"].path
