from marin.execution.executor import ExecutorStep, executor_main, this_output_path
from marin.generation.logits import TextLogitsConfig, compute_logits

logits_step = ExecutorStep(
    name="tutorials/forward-logits",
    fn=compute_logits,
    config=TextLogitsConfig(
        model_name="meta-llama/Meta-Llama-3.1-8B",
        input_path="path/to/input.jsonl.gz",
        output_path=this_output_path(),
        batch_size=8,
        memory_gb=10,
    ),
)

if __name__ == "__main__":
    executor_main([logits_step])
