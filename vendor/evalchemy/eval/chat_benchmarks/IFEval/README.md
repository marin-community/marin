# IFEval: Instruction Following Eval

This is not an officially supported Google product.

This repository contains source code and data for
[Instruction Following Evaluation for Large Language Models](arxiv.org/abs/2311.07911)

Evalchemy fetches `ifeval_input_data.jsonl` from
[`google/IFEval`](https://huggingface.co/datasets/google/IFEval) at revision
`966cd89545d6b6acfd7638bc708b98261ca58e84` through its dataset cache.
The prompt for key `2785` retains the local correction requesting three
placeholders, matching its grading arguments. The default evaluation slice
remains rows `[10:510]`.

Run the `IFEval` task through the standard Evalchemy CLI. Install the
`ifeval` extra to include the instruction checkers' dependencies.

## Reference

If you use our work, please consider citing our preprint:

```
@article{zhou2023instruction,
  title={Instruction-Following Evaluation for Large Language Models},
  author={Zhou, Jeffrey and Lu, Tianjian and Mishra, Swaroop and Brahma, Siddhartha and Basu, Sujoy and Luan, Yi and Zhou, Denny and Hou, Le},
  journal={arXiv preprint arXiv:2311.07911},
  year={2023}
}
```

## Status (ChangeLog)
This is a manual copy of the main branch folder -- instruction\_following\_eval in google-research repository -- [Link](https://github.com/google-research/google-research/tree/master/instruction_following_eval) on October 24th, 2024. To check commit history please go to the link. Any changes made in the code are tracked here README.

* Added eval\_instruct.py to integrate with lm\_eval API. (Complete)
* Added evaluation.py which is wrapper around evaluation_main.py to call important functions and calculate similar evaluations and return a json. (Complete)
* For package inclusion some changes were made. Check the earlier commit after bug fixes which works on changes to package inclusion.


Package needed to be installed: `langdetect immutabledict`
