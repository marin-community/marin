# BigCodeBench

|Version|HF repo ID|Splits|Subsets|Metric|Codebase|License|
|-|-|-|-|-|-|-|
|v0.1.3|`bigcode/bigcodebench`|full/hard|complete/instruct|pass@k|[GitHub](https://github.com/bigcode-project/bigcodebench/tree/main)|[Apache 2.0](https://github.com/bigcode-project/bigcodebench/blob/main/LICENSE)|

Table of Contents
- [Evaluate with a Docker container](#evaluate-on-your-machine-nor-recommended)
- [Evaluate on your machine (not recommended)](#evaluate-on-your-machine-not-recommended)
- [Citation](#citation)
- [Aknowledgements](#aknowledgements)


## Evaluate with a Docker container
To safely execute the code, you can isolate the execution inside a Docker container. [Here](https://hub.docker.com/repository/docker/marianna13/evalchemy) you can find a ready-to-use docker image. The Dockerfile that was used for building the image can be found [here](docker/Dockerfile).

Then, specify your command. For example, you can use vLLM backend for generation:
```bash
CMD="python -m eval.eval \
    --model vllm \
    --tasks BigCodeBench \
    --model_args pretrained=Qwen/Qwen2.5-7B-Instruct \
    --batch_size auto"
```

Then you can run the evaluation inside the container:

```bash
docker run --gpus \
    -v $(pwd):/app -t marianna13/evalchemy:latest \
    $CMD
```

## Evaluate on your machine (not recommended)

> **🚨 Warning:** proceed with caution.

Install all [dependencies](requirements/requirements-eval.txt):
```bash
pip install -r requirements/requirements.txt
pip install -r requirements/requirements-eval.txt
```
Then run the evaluation:
```bash
python -m eval.eval \
    --model vllm \
    --tasks BigCodeBench \
    --model_args pretrained=Qwen/Qwen2.5-7B-Instruct \
    --batch_size auto 
```

## Citation

```bibtex
@article{zhuo2024bigcodebench,
  title={BigCodeBench: Benchmarking Code Generation with Diverse Function Calls and Complex Instructions},
  author={Zhuo, Terry Yue and Vu, Minh Chien and Chim, Jenny and Hu, Han and Yu, Wenhao and Widyasari, Ratnadira and Yusuf, Imam Nur Bani and Zhan, Haolan and He, Junda and Paul, Indraneil and others},
  journal={arXiv preprint arXiv:2406.15877},
  year={2024}
}
```

## Aknowledgements
Thanks to the wonderful team of BigCode for making their benchmark and code publically available! 🙏