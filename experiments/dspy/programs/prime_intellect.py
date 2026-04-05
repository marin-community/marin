import dspy


class _SingleTurnAnswerSignature(dspy.Signature):
    """Solve the given problem step by step and provide a final answer."""

    question: str = dspy.InputField()
    answer: str = dspy.OutputField(desc="The final answer — a short value, number, or phrase.")


class _MultiHopAnswerSignature(dspy.Signature):
    """Solve the given problem using the collected notes and provide a final answer."""

    question: str = dspy.InputField()
    notes: list[str] = dspy.InputField()
    answer: str = dspy.OutputField(desc="The final answer — a short value, number, or phrase.")


class PrimeIntellectSolver(dspy.Module):
    """Single-turn solver for Prime Intellect environment tasks (math, QA, code, etc.)."""

    def __init__(self, search=None, num_docs=3, num_hops=0):
        self.search = search
        self.num_docs = num_docs
        self.num_hops = num_hops

        if num_hops > 0 and search is not None:
            self.generate_query = dspy.ChainOfThought("question, notes -> search_query")
            self.append_notes = dspy.ChainOfThought("question, notes, context -> new_notes: list[str]")
            self.generate_answer = dspy.ChainOfThought(_MultiHopAnswerSignature)
        else:
            self.generate_answer = dspy.ChainOfThought(_SingleTurnAnswerSignature)

    def forward(self, question: str) -> dspy.Prediction:
        if self.num_hops > 0 and self.search is not None:
            return self._multi_hop_forward(question)
        return self._single_turn_forward(question)

    def _single_turn_forward(self, question: str) -> dspy.Prediction:
        pred = self.generate_answer(question=question)
        return dspy.Prediction(
            answer=pred.answer,
            hop_traces=[],
        )

    def _multi_hop_forward(self, question: str) -> dspy.Prediction:
        notes = []
        hop_traces = []

        for hop_idx in range(self.num_hops):
            query = self.generate_query(question=question, notes=notes).search_query
            context = self.search(query, k=self.num_docs)
            passages_this_hop = [{"text": t, "score": s} for t, s in context.items()]
            prediction = self.append_notes(question=question, notes=notes, context=context)
            new_notes = prediction.new_notes
            notes.extend(new_notes)

            hop_traces.append(
                {
                    "hop": hop_idx + 1,
                    "search_query": query,
                    "context": passages_this_hop,
                    "new_notes": new_notes,
                }
            )

        pred = self.generate_answer(question=question, notes=notes)

        return dspy.Prediction(
            notes=notes,
            hop_traces=hop_traces,
            answer=pred.answer,
        )
