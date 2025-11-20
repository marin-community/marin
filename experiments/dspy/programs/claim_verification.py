import dspy

class ClaimVerification(dspy.Module):
    def __init__(self, passages_per_hop=3, max_hops=2):
        super().__init__()
        self.retrieve = dspy.Retrieve(k=passages_per_hop)
        self.generate_query = [dspy.ChainOfThought("evidence, claim -> search_query") for _ in range(max_hops)]
        self.generate_answer = dspy.ChainOfThought("evidence, claim -> answer")
        self.max_hops = max_hops

    def forward(self, claim, **kwargs):
        evidence = []
        
        for hop in range(self.max_hops):
            query = self.generate_query[hop](evidence=evidence, claim=claim).search_query
            passages = self.retrieve(query).passages
            evidence.extend(passages)

        pred = self.generate_answer(evidence=evidence, claim=claim)
        return dspy.Prediction(evidence=evidence, answer=pred.answer)