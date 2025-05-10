MEDU_BENCHMARK_DESCRIPTION_TEMPLATE = """
{corpus}
Help me decide the types of training data to look for to train a
language model for an evaluation with data similar to the
above.
You should keep the description brief and it is okay to generalize
or abstract specific details to do so.
Give your answer in three sections, first write what type of test
this might be from, then write out the languages, skills and
knowledge the language model would need, and finally write a
description of the ideal training data for the evaluation.
"""

MEDU_BENCHMARK_DESCRIPTION_MERGING_TEMPLATE = """
<BEGIN CORPUS DESCRIPTION A>
{description_a}
<END CORPUS DESCRIPTION A>
<BEGIN CORPUS DESCRIPTION B>
{description_b}
<END CORPUS DESCRIPTION B>
The above analyses were written about a NLP evaluation used for
Large Language Models by two different people based on equally
sized random samples of examples from the evaluation.
Help me synthesize them into a more complete analyses based on
both of them. You should keep the description brief and it is
okay to generalize or abstract specific details to do so.
Give your answer in three sections, first write what type of test
this might be from, then write out the languages, skills and
knowledge the language model would need, and finally write a
description of the ideal training data for the evaluation.
"""
MEDU_DOCUMENT_LABELING_PROMPT = """
The following document is being considered as training data for a
Large Language Model.
First, provide a concise description of the document and an assessment of
the quality of the text or code in the document.
Key Attributes to Mention
- Languages contained in the document
- The coherence of the document
- The skills the document demonstrates
- The topics the document contains facts and information about
Document:
'''
{example}
'''
Based on your reasoning, give me a concrete decision
about the utility of the document as training data for the
following benchmark.
{test_description}
Output your decision about the utility of the data as "Final Score:"
followed by one of the following words Great/Good/Okay/Poor/Useless.
"""
