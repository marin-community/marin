# VLM Benchmark Evaluation 实现计划

本文档描述如何在 Levanter 框架中实现 VLM (Vision-Language Model) Benchmark 评估系统。

---

## 1. 目标

为 Levanter 中的 VLM 模型（如 LLaVA OneVision）实现 **Benchmark 评估**，支持以下 9 个标准评测集：

| Benchmark | Dataset (HuggingFace) | 任务类型 | 评估指标 | 状态 |
|-----------|----------------------|---------|---------|------|
| MMMU | `MMMU/MMMU` | 多选题/生成 | Accuracy | lm-eval 已有 |
| ChartQA | `HuggingFaceM4/ChartQA` | 图表问答 | Relaxed Accuracy | lm-eval 已有 |
| MME | `lmms-lab/MME` | Yes/No 问答 | Accuracy/Score | 需实现 |
| GQA | `lmms-lab/GQA` | 视觉推理 | Accuracy | 需实现 |
| RealWorldQA | `xai-org/RealworldQA` | 多选题 | Accuracy | 需实现 |
| SEED | `AILab-CVC/SEED-Bench` | 多选题 | Accuracy | 需实现 |
| MMStar | `Lin-Chen/MMStar` | 多选题 | Accuracy | 需实现 |
| AI2D | `lmms-lab/ai2d` | 科学图表理解 | Accuracy | 需实现 |
| OCRBench | `echo840/OCRBench` | 文字识别 | Accuracy | 需实现 |

---

## 2. 技术方案

### 2.1 方案选择：集成 lm-eval-harness

**选择理由：**
1. MMMU 和 ChartQA 已在 lm-eval-harness 中实现，可直接复用
2. 标准化的 YAML 任务配置格式，便于扩展新 benchmark
3. 现有 `LlavaInferenceEngine` 已支持 VLM 生成任务

### 2.2 现有架构

| 组件 | 文件 | 功能 |
|-----|------|-----|
| VLM 模型 | `lib/levanter/src/levanter/models/llava_onevision.py` | LLaVA OneVision 模型、推理引擎 |
| 图像处理 | `lib/levanter/src/levanter/data/image.py` | BatchImageProcessor, CustomVLMProcessor |
| LLM Eval Harness | `lib/levanter/src/levanter/eval_harness.py` | LevanterHarnessLM 适配器 |

### 2.3 关键数据结构

**VLMRequest**（推理请求）:
```python
@dataclass
class VLMRequest:
    prompt_tokens: jnp.ndarray      # Tokenized prompt
    pixel_values: jnp.ndarray       # (TOTAL_PATCHES, C, H, W)
    grid_mask: jnp.ndarray          # (TOTAL_PATCHES,) - valid patch mask
    input_ids: jnp.ndarray          # Full input with image placeholders
    unpad_indices: Optional[jnp.ndarray]
    num_unpadded_features: Optional[int]
```

---

## 3. 实现计划

### Step 1: 创建 VLM Eval Harness 适配器

**文件**: `lib/levanter/src/levanter/vlm_eval_harness.py`

```python
class LevanterVLMHarnessLM(TemplateLM):
    """Levanter VLM 与 lm-eval-harness 的适配器"""

    MULTIMODAL = True

    def __init__(
        self,
        model: LlavaOnevisionModel,
        tokenizer: PreTrainedTokenizerBase,
        processor: ProcessorMixin,
        EvalBatch: Axis,
        EvalPos: Axis,
        axis_resources: ResourceMapping,
        mp: jmp.Policy | None = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.processor = processor
        self.engine = LlavaInferenceEngine.from_model_with_config(model, ...)

    def tok_batch_multimodal_encode(
        self,
        strings: List[str],
        images: List[List[PIL.Image]],
    ) -> Dict[str, NamedArray]:
        """编码文本+图像为模型输入"""
        # 使用 BatchImageProcessor 处理图像
        # 返回 pixel_values, input_ids, grid_mask 等
        pass

    def generate_until(self, requests: List[Instance]) -> List[str]:
        """VLM 生成任务"""
        results = []
        for req in requests:
            # 从 request.args 提取图像
            images = req.args[2].get("visual", [])
            context = req.args[0]

            # 创建 VLMRequest
            vlm_request = self._create_vlm_request(context, images)

            # 使用 LlavaInferenceEngine 生成
            result = self.engine.generate([vlm_request])
            text = self.tokenizer.decode(result.tokens[0])
            results.append(text)

        return results

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        """计算 log-likelihood（用于多选题评估）"""
        # 前向传播计算每个选项的 log probability
        pass
```

### Step 2: 创建 Benchmark 任务配置

**目录结构**: `configs/vlm_tasks/`

```
configs/vlm_tasks/
├── mme/
│   ├── mme.yaml
│   └── utils.py
├── gqa/
│   ├── gqa.yaml
│   └── utils.py
├── realworldqa/
│   ├── realworldqa.yaml
│   └── utils.py
├── seed/
│   ├── seed.yaml
│   └── utils.py
├── mmstar/
│   ├── mmstar.yaml
│   └── utils.py
├── ai2d/
│   ├── ai2d.yaml
│   └── utils.py
└── ocrbench/
    ├── ocrbench.yaml
    └── utils.py
```

**任务 YAML 模板**（以 MME 为例）:

```yaml
# mme.yaml
dataset_path: lmms-lab/MME
test_split: test
output_type: generate_until

doc_to_image: !function utils.doc_to_image
doc_to_text: !function utils.doc_to_text
doc_to_target: "answer"
process_results: !function utils.process_results

generation_kwargs:
  until: ["<|endoftext|>", "<|im_end|>"]
  temperature: 0.0
  do_sample: false
  max_gen_toks: 64

metric_list:
  - metric: acc
    aggregation: mean
    higher_is_better: true
```

**任务工具函数**（以 MME 为例）:

```python
# utils.py
def doc_to_image(doc):
    """提取图像"""
    return [doc["image"]]

def doc_to_text(doc):
    """格式化 prompt"""
    question = doc["question"]
    return f"<image>\n{question}\nPlease answer yes or no."

def process_results(doc, results):
    """评估模型输出"""
    prediction = results[0].strip().lower()
    reference = doc["answer"].lower()
    return {"acc": 1.0 if prediction == reference else 0.0}
```

### Step 3: 创建评估入口脚本

**文件**: `lib/levanter/src/levanter/main/eval_vlm.py`

```python
@dataclass
class VLMEvalHarnessConfig:
    """VLM Benchmark 评估配置"""
    task_spec: list[str]              # ["mmmu", "chartqa", "mme", ...]
    max_examples: int | None = None   # 每个任务最大样本数
    max_images: int = 10              # 每个样本最大图像数

@dataclass
class VLMEvalMainConfig:
    eval_harness: VLMEvalHarnessConfig
    model: LlavaOnevisionConfig
    checkpoint_path: str
    trainer: TrainerConfig = field(default_factory=TrainerConfig)

def main(config: VLMEvalMainConfig):
    """VLM Benchmark 评估主函数"""
    levanter.initialize(config)

    # 1. 加载模型
    model = load_vlm_checkpoint(config.checkpoint_path, config.model)

    # 2. 创建 VLM Harness 适配器
    vlm_lm = LevanterVLMHarnessLM(
        model=model,
        tokenizer=tokenizer,
        processor=processor,
        ...
    )

    # 3. 运行评估
    results = lm_eval.evaluator.simple_evaluate(
        model=vlm_lm,
        tasks=config.eval_harness.task_spec,
        limit=config.eval_harness.max_examples,
    )

    # 4. 输出结果
    for task, metrics in results["results"].items():
        print(f"{task}: {metrics}")

if __name__ == "__main__":
    levanter.config.main(main)()
```

---

## 4. 各 Benchmark 详细说明

### 4.1 MME (MultiModal Evaluation)

- **数据集**: `lmms-lab/MME`
- **任务类型**: Yes/No 问答
- **子任务**: 14 个感知任务 + 6 个认知任务
- **评估指标**:
  - Accuracy (正确率)
  - Score = accuracy_positive + accuracy_negative

### 4.2 GQA (Visual Reasoning)

- **数据集**: `lmms-lab/GQA`
- **任务类型**: 开放式视觉问答
- **评估指标**: Accuracy (答案归一化后精确匹配)
- **特殊处理**: 答案需要小写化、去除标点

### 4.3 RealWorldQA

- **数据集**: `xai-org/RealworldQA`
- **任务类型**: 多选题 (A/B/C/D)
- **评估指标**: Accuracy
- **场景**: 真实世界理解（如驾驶场景）

### 4.4 SEED (Scene Understanding)

- **数据集**: `AILab-CVC/SEED-Bench`
- **任务类型**: 多选题
- **评估指标**: Accuracy
- **维度**: 场景、实例、文本、空间等多维度评测

### 4.5 MMStar

- **数据集**: `Lin-Chen/MMStar`
- **任务类型**: 多选题 (A/B/C/D)
- **评估指标**: Accuracy
- **特点**: 视觉不可或缺的 benchmark（必须看图才能回答）

### 4.6 AI2D (Science Diagrams)

- **数据集**: `lmms-lab/ai2d`
- **任务类型**: 多选题
- **评估指标**: Accuracy
- **场景**: 科学图表理解

### 4.7 OCRBench

- **数据集**: `echo840/OCRBench`
- **任务类型**: 文字识别/问答
- **评估指标**: Accuracy (按任务类别)
- **子任务**: 文本识别、场景文字、文档、表格、KIE 等 29 个子任务

---

## 5. 使用方法

### 5.1 运行单个 Benchmark

```bash
python -m levanter.main.eval_vlm \
    --config experiments/VLM/eval_vlm_config.yaml \
    --eval_harness.task_spec='["mme"]' \
    --checkpoint_path /path/to/checkpoint
```

### 5.2 运行全部 Benchmark

```bash
python -m levanter.main.eval_vlm \
    --config experiments/VLM/eval_vlm_config.yaml \
    --eval_harness.task_spec='["mmmu","chartqa","mme","gqa","realworldqa","seed","mmstar","ai2d","ocrbench"]' \
    --checkpoint_path /path/to/checkpoint
```

### 5.3 限制样本数（调试用）

```bash
python -m levanter.main.eval_vlm \
    --config experiments/VLM/eval_vlm_config.yaml \
    --eval_harness.task_spec='["mme"]' \
    --eval_harness.max_examples=10 \
    --checkpoint_path /path/to/checkpoint
```

---

## 6. 关键技术细节

### 6.1 图像处理

VLM 使用固定形状张量以支持 JAX JIT 编译：

```python
# pixel_values: (TOTAL_PATCHES, C, H, W)
# TOTAL_PATCHES = 1 (base) + max_grid_patches (e.g., 9 for anyres_max_9)
# 无效 patch 通过 grid_mask 标记
```

### 6.2 生成限制

当前 `LlavaInferenceEngine` 仅支持单请求生成：

```python
# 单请求处理
result = engine.generate([vlm_request])

# 批量请求需要循环处理
for req in requests:
    result = engine.generate([req])
    results.append(result)
```

### 6.3 Chat Template

使用 Qwen 格式的 chat template：

```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
<image>
Question here<|im_end|>
<|im_start|>assistant
```

---

## 7. 参考资源

### 7.1 代码参考

- lm-eval-harness VLM 适配器: `.venv/lib/python3.11/site-packages/lm_eval/models/hf_vlms.py`
- lm-eval-harness MMMU 任务: `.venv/lib/python3.11/site-packages/lm_eval/tasks/mmmu/`
- Levanter LLM Eval Harness: `lib/levanter/src/levanter/eval_harness.py`
- VLM 推理引擎: `lib/levanter/src/levanter/models/llava_onevision.py`

### 7.2 文档参考

- [LM Eval Harness Documentation](https://github.com/EleutherAI/lm-evaluation-harness)
- [MMMU Benchmark](https://mmmu-benchmark.github.io/)
- [MME Benchmark](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models)
