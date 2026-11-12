import evaluate as hf_evaluate

from eval.generation_stops import HUMANEVAL_STOP_SEQUENCES, truncate_at_stop

compute = hf_evaluate.load("code_eval")


def pass_at_k(
    references: list[str],
    predictions: list[list[str]],
    k: list[int] | int | None = None,
):
    assert k is not None
    if isinstance(k, int):
        k = [k]
    return compute.compute(references=references, predictions=predictions, k=k)[0]


def build_predictions(resps: list[list[str]], docs: list[dict]) -> list[list[str]]:
    return [
        [
            doc["prompt"] + truncate_at_stop(response, HUMANEVAL_STOP_SEQUENCES)
            for response in responses
        ]
        for responses, doc in zip(resps, docs)
    ]
