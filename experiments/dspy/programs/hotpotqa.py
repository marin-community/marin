import dspy
from dsp.utils import deduplicate


class HotpotQA(dspy.Module):
    def __init__(self, search, num_docs=3, num_hops=2):
        self.search = search
        self.num_docs = num_docs
        self.num_hops = num_hops
        self.generate_query = dspy.ChainOfThought("context, question -> search_query")
        self.generate_answer = dspy.ChainOfThought("context, question -> answer")

    def forward(self, question: str) -> dspy.Prediction:
        context = []
        hop_traces = []

        for hop_idx in range(self.num_hops):
            query = self.generate_query(context=context, question=question).search_query
            passages = self.search(query, k=self.num_docs)
            context = deduplicate(context + list(passages))

            hop_traces.append({
                "hop": hop_idx + 1,
                "search_query": query,
                "context": context,
            })

        pred = self.generate_answer(context=context, question=question)

        return dspy.Prediction(
            context=context,
            hop_traces=hop_traces,
            answer=pred.answer,
        )
