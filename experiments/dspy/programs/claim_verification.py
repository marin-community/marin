import dspy
from enum import Enum

class ClaimVerificationLabel(Enum):
    SUPPORTS = "Supports"
    REFUTES = "Refutes"

class ClaimVerificationSignature(dspy.Signature):
    """You are a helpful assistant that verifies claims based on the evidence provided. 
    
    Rules:
    - Run the claim through a series of hops, each hop retrieving a list of evidence passages.
    - The evidence passages are concatenated and used to answer the claim.
    - The label for the claim, either "Supports" or "Refutes".
    """


    claim: str = dspy.InputField()
    evidence: list[str] = dspy.InputField()
    label: ClaimVerificationLabel = dspy.OutputField()

class ClaimVerification(dspy.Module):
    def __init__(self, passages_per_hop=3, max_hops=2):
        super().__init__()
        self.retrieve = dspy.Retrieve(k=passages_per_hop)
        self.generate_query = [dspy.ChainOfThought("evidence, claim -> search_query") for _ in range(max_hops)]
        self.generate_answer = dspy.ChainOfThought(ClaimVerificationSignature)
        self.max_hops = max_hops

    def forward(self, claim, evidence: list[str]):
        evidence = []
        
        for hop in range(self.max_hops):
            query = self.generate_query[hop](evidence=evidence, claim=claim).search_query
            passages = self.retrieve(query).passages
            evidence.extend(passages)

        pred = self.generate_answer(evidence=evidence, claim=claim)
        return dspy.Prediction(evidence=evidence, label=pred.label, label_int=int(pred.label == ClaimVerificationLabel.SUPPORTS))