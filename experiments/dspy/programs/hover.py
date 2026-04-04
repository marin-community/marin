import enum

import dspy


class HoverLabel(enum.Enum):
    SUPPORTED = "SUPPORTED"
    NOT_SUPPORTED = "NOT_SUPPORTED"


class _HoverAnswerSignature(dspy.Signature):
    """Verify whether the claim is supported or not supported based on the collected notes."""

    claim: str = dspy.InputField()
    notes: list[str] = dspy.InputField()
    label: HoverLabel = dspy.OutputField(desc='Respond with exactly "SUPPORTED" or "NOT_SUPPORTED".')


class HoVer(dspy.Module):
    def __init__(self, search, num_docs=3, num_hops=2):
        self.search = search
        self.num_docs, self.num_hops = num_docs, num_hops
        self.generate_query = dspy.ChainOfThought("claim, notes -> search_query")
        self.append_notes = dspy.ChainOfThought("claim, notes, context -> new_notes: list[str]")
        self.generate_answer = dspy.ChainOfThought(_HoverAnswerSignature)

    def forward(self, claim: str) -> dspy.Prediction:
        notes = []
        hop_traces = []

        for hop_idx in range(self.num_hops):
            query = self.generate_query(claim=claim, notes=notes).search_query
            context = self.search(query, k=self.num_docs)
            passages_this_hop = [{"text": t, "score": s} for t, s in context.items()]
            prediction = self.append_notes(claim=claim, notes=notes, context=context)
            new_notes = prediction.new_notes
            notes.extend(new_notes)

            hop_traces.append({
                "hop": hop_idx + 1,
                "search_query": query,
                "context": passages_this_hop,
                "new_notes": new_notes,
            })

        pred = self.generate_answer(claim=claim, notes=notes)

        return dspy.Prediction(
            notes=notes,
            hop_traces=hop_traces,
            label=pred.label,
            label_int=int(pred.label == HoverLabel.SUPPORTED),
        )
