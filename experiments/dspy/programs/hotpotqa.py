import dspy


class _HotpotQAAnswerSignature(dspy.Signature):
    """Answer the multi-hop question based on the collected notes."""

    question: str    = dspy.InputField()
    notes: list[str] = dspy.InputField()
    answer: str      = dspy.OutputField(desc="A short, direct answer — a few words or a single phrase.")


class HotpotQA(dspy.Module):
    def __init__(self, search, num_docs=3, num_hops=2):
        self.search = search
        self.num_docs, self.num_hops = num_docs, num_hops
        self.generate_query  = dspy.ChainOfThought("question, notes -> search_query")
        self.append_notes    = dspy.ChainOfThought("question, notes, context -> new_notes: list[str]")
        self.generate_answer = dspy.ChainOfThought(_HotpotQAAnswerSignature)

    def forward(self, question: str) -> dspy.Prediction:
        notes      = []
        hop_traces = []

        for hop_idx in range(self.num_hops):
            query             = self.generate_query(question=question, notes=notes).search_query
            context           = self.search(query, k=self.num_docs)
            passages_this_hop = [{"text": t, "score": s} for t, s in context.items()]
            prediction        = self.append_notes(question=question, notes=notes, context=context)
            new_notes         = prediction.new_notes
            notes.extend(new_notes)

            hop_traces.append({
                "hop":          hop_idx + 1,
                "search_query": query,
                "context":      passages_this_hop,
                "new_notes":    new_notes,
            })

        pred = self.generate_answer(question=question, notes=notes)

        return dspy.Prediction(
            notes      = notes,
            hop_traces = hop_traces,
            answer     = pred.answer,
        )
