# VLM Evaluation 实现计划

本文档描述如何在 Levanter 框架中建立一套完整的 VLM (Vision-Language Model) 评估系统。

---

## 1. 目标

为 Levanter 中的 VLM 模型（如 LLaVA OneVision）建立一套评估系统，支持：

1. **基础指标评估**：Loss、Perplexity、Bits-Per-Byte
2. **VLM Benchmark 评估**：VQA、Image Captioning 等标准任务
3. **训练期间评估**：通过回调函数定期评估
4. **独立评估脚本**：支持对保存的模型进行评估

---

## 2. 现有架构分析

### 2.1 VLM 模型实现

| 文件 | 功能 |
|------|------|
| `lib/levanter/src/levanter/models/llava_onevision.py` | LLaVA OneVision 模型实现 |
| `lib/levanter/src/levanter/models/vlm_model.py` | VLM 基类和接口定义 |
| `lib/levanter/src/levanter/models/siglip.py` | SigLIP 视觉编码器 |

### 2.2 VLM 数据处理

| 文件 | 功能 |
|------|------|
| `lib/levanter/src/levanter/data/image.py` | 图像数据处理、数据集、DataLoader |
| `lib/levanter/src/levanter/main/train_vlm.py` | VLM 训练脚本 |

### 2.3 现有 LLM 评估架构

| 文件 | 功能 | 可复用性 |
|------|------|---------|
| `lib/levanter/src/levanter/eval.py` | TaggedEvaluator, EvalResult | 需要扩展 |
| `lib/levanter/src/levanter/eval_harness.py` | LM Eval Harness 集成 | 需要扩展 |

### 2.4 关键数据结构

**ImageTextExample**（现有）:
```python
@dataclass
class ImageTextExample:
    input_ids: hax.NamedArray       # (batch, position)
    attention_mask: hax.NamedArray
    pixel_values: hax.NamedArray    # (batch, TOTAL_PATCHES, C, H, W)
    grid_mask: hax.NamedArray       # (batch, TOTAL_PATCHES)
    combined_mask: hax.NamedArray   # (batch, position)
    position_ids: hax.NamedArray    # (batch, position)
    unpad_indices: Optional[hax.NamedArray]
```

---

## 3. 实现计划

### 阶段 1：创建 VLM 评估器基础

#### 3.1.1 创建 VLMExample 数据结构

**文件**: `lib/levanter/src/levanter/models/vlm_model.py`

```python
@dataclass
class VLMExample(eqx.Module):
    """VLM 评估样本，扩展自 ImageTextExample"""
    input_ids: hax.NamedArray       # (batch, position)
    pixel_values: hax.NamedArray    # (batch, TOTAL_PATCHES, C, H, W)
    grid_mask: hax.NamedArray       # (batch, TOTAL_PATCHES)
    combined_mask: hax.NamedArray   # (batch, position)
    position_ids: hax.NamedArray    # (batch, position)
    loss_weight: hax.NamedArray     # (batch, position)
    attn_mask: AttentionMask
    unpad_indices: Optional[hax.NamedArray] = None

    @staticmethod
    def from_image_text_example(example: ImageTextExample) -> "VLMExample":
        """从 ImageTextExample 转换"""
        pass
```

#### 3.1.2 创建 VLMTaggedEvaluator

**文件**: `lib/levanter/src/levanter/eval_vlm.py`（新文件）

```python
class VLMTaggedEvaluator:
    """VLM 多数据集标注评估器"""

    def __init__(
        self,
        EvalBatch: hax.Axis,
        tagged_eval_sets: Sequence[tuple[AsyncDataset[VLMExample], Sequence[str]]],
        tokenizer: Optional[HfTokenizer] = None,
        device_mesh: Optional[Mesh] = None,
        axis_mapping: ResourceMapping | None = None,
        max_examples_per_dataset: Optional[int] = None,
    ):
        self.dataset = DomainTaggedDataset(tagged_eval_sets, max_examples_per_dataset)
        self.loader = ImageDataLoader(...)  # 使用 ImageDataLoader

    def evaluate(self, model: VlmModel) -> EvalResult:
        """执行 VLM 评估"""
        for batch, tags in self.loader:
            # 计算 VLM loss
            losses = self._compute_vlm_loss(model, batch)
            # 聚合指标
            ...
        return EvalResult(...)

    def _compute_vlm_loss(self, model: VlmModel, batch: VLMExample):
        """计算 VLM per-token loss"""
        activations, lm_head = model.forward_with_activations(
            input_ids=batch.input_ids,
            pixel_values=batch.pixel_values,
            grid_mask=batch.grid_mask,
            combined_mask=batch.combined_mask,
            position_ids=batch.position_ids,
        )
        return next_token_loss(activations, lm_head, ...)
```

#### 3.1.3 创建 VLM 评估回调

**文件**: `lib/levanter/src/levanter/eval_vlm.py`

```python
def cb_tagged_vlm_evaluate(
    EvalBatch: hax.Axis,
    tagged_eval_sets: Sequence[tuple[AsyncDataset[VLMExample], Sequence[str]]],
    tokenizer: Optional[HfTokenizer] = None,
    device_mesh: Optional[Mesh] = None,
    axis_mapping: ResourceMapping | None = None,
    max_examples_per_dataset: Optional[int] = None,
    prefix: str = "eval",
) -> Callable[[StepInfo], None]:
    """创建 VLM 训练期间评估回调"""
    evaluator = VLMTaggedEvaluator(...)

    def eval_callback(step: StepInfo):
        log_dict = eval_vlm_model(evaluator, step.model, prefix=prefix)
        levanter.tracker.log(log_dict, step=step.step)

    return eval_callback
```

### 阶段 2：集成 LM Eval Harness

#### 3.2.1 创建 LevanterVLMHarnessLM

**文件**: `lib/levanter/src/levanter/eval_harness_vlm.py`（新文件）

```python
class LevanterVLMHarnessLM(HFMultimodalLM):
    """Levanter VLM 与 lm-eval-harness 的适配器"""

    MULTIMODAL = True

    def __init__(
        self,
        model: LlavaOnevisionModel,
        tokenizer: HfTokenizer,
        processor: CustomVLMProcessor,
        EvalBatch: hax.Axis,
        axis_resources: ResourceMapping,
        mp: jmp.Policy | None = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.processor = processor
        self.inference_engine = LlavaInferenceEngine.from_model_with_config(model, ...)

    def loglikelihood(self, requests: list[Instance]) -> list[tuple[float, bool]]:
        """计算 VLM log-likelihood"""
        # 1. 处理图像和文本
        # 2. 创建 VLMRequest
        # 3. 计算 log-likelihood
        pass

    def generate_until(self, requests) -> List[str]:
        """VLM 文本生成"""
        # 使用 LlavaInferenceEngine 进行生成
        vlm_requests = [self._create_vlm_request(req) for req in requests]
        results = self.inference_engine.generate(vlm_requests)
        return [self.tokenizer.decode(r.tokens) for r in results]
```

#### 3.2.2 VLM Eval Harness 配置

```python
@dataclass(frozen=True)
class VLMEvalHarnessConfig:
    """VLM 评估配置"""
    task_spec: list[VLMTaskConfig | str]
    max_examples: int | None = None
    max_length: int | None = None

    # VLM 特定配置
    image_size: int = 384
    max_num_patches: int = 10  # anyres_max_9 = 1 + 9

    # 生成参数
    generation_kwargs: dict = field(
        default_factory=lambda: {
            "max_gen_toks": 256,
            "temperature": 0.0,
        }
    )

@dataclass(frozen=True)
class VLMTaskConfig:
    """VLM 任务配置"""
    task: str                    # 任务名称 (e.g., "okvqa", "gqa")
    task_alias: str | None = None
    num_fewshot: int = 0
    metric: str = "accuracy"     # 评估指标
```

### 阶段 3：支持常见 VLM Benchmark

#### 3.3.1 VLM Benchmark 任务类

**文件**: `lib/levanter/src/levanter/eval_vlm_tasks.py`（新文件）

```python
class VLMEvalTask(ABC):
    """VLM 评估任务基类"""

    @abstractmethod
    def load_dataset(self) -> AsyncDataset[VLMExample]:
        """加载评估数据集"""
        pass

    @abstractmethod
    def compute_metric(
        self,
        predictions: List[str],
        references: List[str]
    ) -> Dict[str, float]:
        """计算任务特定指标"""
        pass

    @property
    @abstractmethod
    def task_name(self) -> str:
        pass


class VQAEvalTask(VLMEvalTask):
    """Visual Question Answering 任务"""
    task_name = "vqa"

    def load_dataset(self) -> AsyncDataset[VLMExample]:
        # 加载 OKVQA, GQA, TextVQA 等数据集
        pass

    def compute_metric(self, predictions, references) -> Dict[str, float]:
        # VQA Accuracy 计算
        return {"vqa_accuracy": accuracy}


class ImageCaptionEvalTask(VLMEvalTask):
    """Image Captioning 任务"""
    task_name = "caption"

    def load_dataset(self) -> AsyncDataset[VLMExample]:
        # 加载 COCO Caption, Flickr30k 等
        pass

    def compute_metric(self, predictions, references) -> Dict[str, float]:
        # BLEU, METEOR, CIDEr, SPICE
        return {
            "bleu_4": bleu4,
            "meteor": meteor,
            "cider": cider,
        }


class DocVQAEvalTask(VLMEvalTask):
    """Document VQA 任务"""
    task_name = "docvqa"

    def compute_metric(self, predictions, references) -> Dict[str, float]:
        # ANLS (Average Normalized Levenshtein Similarity)
        return {"anls": anls_score}
```

#### 3.3.2 支持的 Benchmark 任务列表

| 任务类型 | 数据集 | 指标 |
|---------|-------|------|
| **VQA** | OKVQA, GQA, TextVQA, VQAv2 | VQA Accuracy |
| **Image Captioning** | COCO Caption, Flickr30k, NoCaps | BLEU, METEOR, CIDEr |
| **Visual Reasoning** | WINOGROUND, CLEVR | Accuracy |
| **Document Understanding** | DocVQA, InfographicVQA | ANLS |
| **Multi-image** | MMMU, MathVista | Task-specific |

### 阶段 4：独立评估脚本

#### 3.4.1 创建 eval_vlm.py 脚本

**文件**: `lib/levanter/src/levanter/main/eval_vlm.py`（新文件）

```python
@dataclass
class EvalVLMConfig:
    """VLM 独立评估配置"""
    checkpoint_path: Optional[str] = None
    hf_checkpoint: Optional[RepoRef] = None

    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    data: ImageMixtureDatasetConfig = field(default_factory=ImageMixtureDatasetConfig)
    model: LlavaOnevisionConfig = field(default_factory=LlavaOnevisionConfig)

    max_eval_length: int = 2048
    eval_on_train: bool = False

    # VLM 特定配置
    max_num_patches: int = 10
    image_size: int = 384


def main(config: EvalVLMConfig):
    """VLM 评估主函数"""
    levanter.initialize(config)

    # 加载模型
    model = load_vlm_model(config)

    # 加载数据集
    datasets = config.data.tagged_eval_sets(Pos)

    # 创建评估器
    evaluator = VLMTaggedEvaluator(
        EvalBatch=config.trainer.EvalBatch,
        tagged_eval_sets=datasets,
        tokenizer=config.data.the_tokenizer,
        ...
    )

    # 执行评估
    log_dict = eval_vlm_model(evaluator, model, prefix="eval")
    levanter.tracker.log(log_dict, step=0)

    print(f"Loss: {log_dict['eval/loss']}")


if __name__ == "__main__":
    levanter.config.main(main)()
```

---

## 4. 实现步骤

### Step 1: 基础数据结构（预计工作量：1-2天）

1. [ ] 在 `vlm_model.py` 中添加 `VLMExample` 数据类
2. [ ] 在 `image.py` 中添加 `ImageTextExample` 到 `VLMExample` 的转换函数
3. [ ] 添加单元测试

### Step 2: VLM 评估器（预计工作量：2-3天）

1. [ ] 创建 `eval_vlm.py` 文件
2. [ ] 实现 `VLMTaggedEvaluator` 类
3. [ ] 实现 `cb_tagged_vlm_evaluate` 回调函数
4. [ ] 添加单元测试

### Step 3: LM Eval Harness 集成（预计工作量：3-4天）

1. [ ] 创建 `eval_harness_vlm.py` 文件
2. [ ] 实现 `LevanterVLMHarnessLM` 类
3. [ ] 实现 `loglikelihood` 方法
4. [ ] 实现 `generate_until` 方法
5. [ ] 添加配置类
6. [ ] 添加单元测试

### Step 4: VLM Benchmark 任务（预计工作量：2-3天）

1. [ ] 创建 `eval_vlm_tasks.py` 文件
2. [ ] 实现 `VQAEvalTask` 类
3. [ ] 实现 `ImageCaptionEvalTask` 类
4. [ ] 添加指标计算函数
5. [ ] 添加数据集加载器

### Step 5: 独立评估脚本（预计工作量：1-2天）

1. [ ] 创建 `main/eval_vlm.py` 脚本
2. [ ] 实现 `EvalVLMConfig` 配置类
3. [ ] 实现主函数
4. [ ] 添加 CLI 支持

### Step 6: 集成测试和文档（预计工作量：1-2天）

1. [ ] 端到端测试
2. [ ] 性能优化
3. [ ] 编写使用文档

---

## 5. 关键技术挑战

### 5.1 固定形状处理

VLM 使用固定形状张量以支持 JAX JIT 编译：

```python
# pixel_values: (batch, TOTAL_PATCHES, C, H, W)
# TOTAL_PATCHES = 1 (base) + max_grid_patches (e.g., 9 for anyres_max_9)
# 无效的补丁通过 grid_mask 标记，特征会被 zeroed out
```

### 5.2 图像-文本对齐

需要正确处理图像 token 和文本 token 的对齐：

```python
# input_ids 中的 image_token_index (151646) 会被替换为图像特征
# 需要使用 unpad_indices 将特征重新排列到正确位置
```

### 5.3 生成任务支持

当前 `LlavaInferenceEngine` 仅支持单请求，需要扩展以支持批量生成。

---

## 6. 验证方式

### 6.1 单元测试

```bash
# 运行 VLM 评估器测试
pytest lib/levanter/tests/test_eval_vlm.py -v

# 运行 VLM Harness 测试
pytest lib/levanter/tests/test_eval_harness_vlm.py -v
```

### 6.2 端到端测试

```bash
# 运行独立评估脚本
python -m levanter.main.eval_vlm \
    --config_path experiments/VLM/eval_vlm_config.yaml \
    --checkpoint_path /path/to/checkpoint
```

### 6.3 Benchmark 验证

使用已知的 benchmark 数据集验证评估结果与其他框架一致：

```bash
# 运行 LM Eval Harness VLM 任务
python -m levanter.main.eval_harness_vlm \
    --tasks okvqa,gqa \
    --checkpoint_path /path/to/checkpoint
```

---

## 7. 参考资源

### 7.1 代码参考

- LM Eval Harness VLM 支持: `.venv/lib/python3.11/site-packages/lm_eval/models/hf_vlms.py`
- Levanter LLM 评估: `lib/levanter/src/levanter/eval.py`
- VLM 模型实现: `lib/levanter/src/levanter/models/llava_onevision.py`

### 7.2 文档参考

- [LM Eval Harness Documentation](https://github.com/EleutherAI/lm-evaluation-harness)
- [LLaVA OneVision Paper](https://arxiv.org/abs/2401.08849)
- Levanter LLM Evaluation Guide: `docs/levanter_evaluation_guide.md`
