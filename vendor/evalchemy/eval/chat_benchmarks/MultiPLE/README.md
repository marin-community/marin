# Multi-Programming Language Evaluation of Large Language Models of Code (MultiPL-E)

|HF repo ID|Splits|Subsets|Metric|Codebase|License|
|-|-|-|-|-|-|
|`nuprl/MultiPL-E`|MBPP/HumanEval|18 languages|pass@1|[GitHub](hhttps://github.com/nuprl/MultiPL-E)|[BSD 3-Clause License](https://github.com/nuprl/MultiPL-E/blob/b30e41f58ee2a5a25e2d1555881272723ce5de73/LICENSE)|

Table of Contents
- [Evaluate with a Docker container](#evaluate-with-a-docker-container)
- [Supported languages](#languages)
- [Citation](#citation)


## Evaluate with a Docker container
To safely execute the code, you can isolate the execution inside a Docker container. [Here](https://hub.docker.com/repository/docker/marianna13/evalchemy_multiple) you can find a ready-to-use docker image. The Dockerfile that was used for building the image can be found [here](docker/Dockerfile).

Then, specify your command. For example, you can use vLLM backend for generation:
```bash
CMD="python -m eval.eval \
    --model vllm \
    --tasks MultiPLE \
    --model_args pretrained=Qwen/Qwen2.5-7B-Instruct \
    --batch_size auto"
```

Then you can run the evaluation inside the container:

```bash
docker run --gpus \
    -v $(pwd):/app -t marianna13/evalchemy_multiple:latest \
    $CMD
```

## Languages

List of currently supported and tested languages:
```
Java, C++, C#, TypeScript, JavaScript, PHP,	Bash, Rust, R, Lua
```

## Citation

```bibtex
@article{Cassano_MultiPL-E_A_Scalable_2023,
author = {Cassano, Federico and Gouwar, John and Nguyen, Daniel and Nguyen, Sydney and Phipps-Costin, Luna and Pinckney, Donald and Yee, Ming-Ho and Zi, Yangtian and Anderson, Carolyn Jane and Feldman, Molly Q and Guha, Arjun and Greenberg, Michael and Jangda, Abhinav},
doi = {10.1109/TSE.2023.3267446},
journal = {IEEE Transactions on Software Engineering},
title = {{MultiPL-E: A Scalable and Polyglot Approach to Benchmarking Neural Code Generation}},
year = {2023}
}
```
