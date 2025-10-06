# Porting Models to Levanter

## Overview
This guide outlines the process of porting new models into Levanter. While we emphasize the scenario of adding a causal language model (LM) that is implemented in Hugging Face, this guide is applicable to general models from all sources.

We'll start with a detailed walkthrough on implementing your model and testing it. Subsequently, we'll describe how to configure and run a training job with your model. To conclude, we'll share insights and recommendations to enhance your model's training efficiency.

## Write Your Model
At a high level, writing a new model in Levanter is very similar to Hugging Face. You start by adding a new file in [models/](https://github.com/stanford-crfm/levanter/tree/main/src/levanter/models). You first create a config class to register the key hyperparameters and axes of your model. Then, you write the model class that includes key layers and components of your model.

### Write Config
We start by writing your model config class. This class will register all the hyperparameters and axes of your models. We want to define them as the first step because they will be used immidately in the next step that implements your model.

#### Define Hyperparameters and Axes
Note that you don't need to write all of configurations at once. You can start with the key hyperparameters and axes, and add more as you implement the model.

**Hyperparameters** should be declared with their type and a default value, illustrated below:

```python
seq_len: int = 2048
hidden_dim: int = 4096
intermediate_dim: int = 11008
```

**Model axes** are used for parallelization. An Axis is registered with its name and size. The size of an Axis is normally associated with a hyperparameter. For example:

```python
Pos = property(lambda self: Axis(name="position", size=self.seq_len))
Embed = property(lambda self: Axis(name="embed", size=self.hidden_dim))
Mlp = property(lambda self: Axis(name="mlp", size=self.intermediate_dim))
```

For real examples and deeper understanding, check out `Gpt2Config` in [gpt2.py](https://github.com/stanford-crfm/levanter/blob/main/src/levanter/models/gpt2.py) and `LlamaConfig` in [llama.py](https://github.com/stanford-crfm/levanter/blob/main/src/levanter/models/llama.py).

#### [For HF models] Convert to/from Hugging Face Config
To convert your config class to and from Hugging Face config class, you will need to:
1. Extend your config class from `HFCompatConfig`
2. Write class functions `to_hf_config()` and `from_hf_config()` that maps the input parameters to the corresponding Hugging Face parameters.

For example, in Llama, we have the following:

```python
from transformers import PretrainedConfig as HfPretrainedConfig
from levanter.compat.hf_checkpoints import HFCompatConfig

# ...

class LlamaConfig(HFCompatConfig):
    # ...
    @classmethod
    def from_hf_config(cls, hf_config: HfPretrainedConfig):
        return LlamaConfig(
            seq_len=hf_config.max_position_embeddings,
            hidden_dim=hf_config.hidden_size,
            intermediate_dim=hf_config.intermediate_size,
            num_layers=hf_config.num_hidden_layers,
            num_heads=hf_config.num_attention_heads,
            activation_function=hf_config.hidden_act,
            initializer_range=hf_config.initializer_range,
            layer_norm_epsilon=hf_config.rms_norm_eps,
            rope_scaling=hf_config.rope_scaling,
        )
```

#### Register Your Model
Lastly, there are a few steps to register a model in Levanter and make it tightly integrated with the training pipeline:
1. For language models, You can register your config class with `LmConfig.register_subclass("ModelName")`. By doing so, the trainer can use the name from the `model.type` field in the training configuration file to identify your config and model.
2. You will need to decorate the config class as a dataclass for parsing the config file.
3. You will need to register your head model's class name as a class property. This step can be deferred until the head class is constructed. Same as 1, this class property would make it easier to automatically identify and select your model in the training and evaluation pipeline.

Below is an example:

```python
@LmConfig.register_subclass("gpt2") # if implementing a Causal Language model
@dataclass(frozen=True)  # for parsing the config file
class MyConfig(HFCompatConfig):
    # ...
    @property
    def model_type(cls) -> Type["Gpt2LMHeadModel"]:
        return Gpt2LMHeadModel
```

### Write Model
After you have defined your config class, you can start writing your model.
You can follow the breakdown of the model in your Hugging Face implementation (or from other sources). This would make it easier to validate the implementation through unit tests.

For example, in GPT2, we have the following breakdown:
- `Gpt2Mlp`
- `Gpt2Attention`
- `Gpt2Block`: a block of Gpt2Mlp and Gpt2Attention
- `Gpt2Transformer`: a stack of Gpt2Block
- `Gpt2Embeddings`: token and position embeddings
- `Gpt2LMHead`: a complete GPT2 model with embedding, transformer, and LM head.

We follow the same breakdown in the implementation of Llama in Levanter.

#### Note on the Implementation Format
- Each class will have its key layers and components defined as attributes and be initialized with a static method `init()`.
- Each class will be extended from Equinox's `Module` class, except for classes with custom serialization logic, which instead inherit
from [haliax.state_dict.ModuleWithStateDictSerialization][].
- [haliax.nn.Linear][] modules can have "articulated" input or output axes, where PyTorch and other libraries typically require
a single input and output axis. For instance, attention modules in Levanter typically have a `Linear`  from `Embed` to `(Heads, HeadSize)`.
When serializing these linear modules to state dicts (see the next section), Haliax will automatically flatten them. You should
ensure that `out_first=True` is set on Linear modules if they're going to be loaded as PyTorch Linear modules.

### Serialization to/from State Dicts

PyTorch and Hugging Face Transformers use "state dicts" as their preferred serialization format, either as pickles or as the new [safetensors](https://github.com/huggingface/safetensors) format.
A state dict is a Python `dict` with string keys and tensor values. The keys of the dict are json-ish "key paths" like `model.blocks.0.mlp.c_proj` and the values are the corresponding parameters for that key path.
You can read [PyTorch's State Dict docs](https://pytorch.org/tutorials/recipes/recipes/what_is_state_dict.html)
if you want to learn more.

[Haliax has machinery for (de)serializing to and from state dicts](https://haliax.readthedocs.io/state-dict/).
Simple cases are handled automatically, but sometimes custom logic is needed.

#### Easy Case: Identical Module Structure
If your module has exactly the same fields with the same names and same shapes as the Hugging Face state dict (e.g. Gpt2Mlp), you don't need to do anything.

#### Medium Case: Different Names
If for some reason you want to use different names from the HF implementation (e.g. because the names from HF aren't clear...), you can extend your class from  `StateDictSerializationMixin` and use `_state_dict_key_map` to rename keys. For instance, the `Gpt2Transformer` class has this method:

```python
from haliax.state_dict import ModuleWithStateDictSerialization

class Gpt2Transformer(ModuleWithStateDictSerialization):
    ...

    def _state_dict_key_map(self) -> Dict[str, Optional[str]]:
        return {"blocks": "h"}
```

This says that the field called `blocks` in this class should be (de)serialized as `h`, because the Hugging Face GPT-2 implementation uses `h`, which is not very clear. You can also "flatten" the submodules of a field by using `None`.

#### Hard Case: Custom Serialization

If your modules need special logic, you'll need to extend your class from `ModuleWithStateDictSerialization` and
overwrite the default function `update_state_dict()` and `from_state_dict()`. It takes in and returns a modified
[haliax.state_dict.StateDict][]. As of May 2024, we almost never this in Levanter.

For implementation, there are a few helper methods from `haliax.state_dict` that you can use:
- To join specific prefix to the keys of Hugging Face state_dict, you can use the helper function `apply_prefix()`. The prefix comes from the name of attributes defined at the beginning of your model class.

For example, below is the implementation of `update_state_dict()` in [levanter.models.backpack.BackpackLMHeadModel][].
In this class, we want to preserve HF compatibility by saving untied output embeddings. (We chose not to implement
non-weight-tied embeddings.)

```python
    def update_state_dict(self, state_dict: StateDict, prefix: Optional[str] = None) -> StateDict:
        state_dict = super().update_state_dict(state_dict, prefix=prefix)
        # In levanter's implementation, we have a shared embedding matrix for both the word
        # embeddings and the sense embeddings
        state_dict[apply_prefix(prefix, "backpack.word_embeddings.weight")] = state_dict[
            apply_prefix(prefix, "backpack.gpt2_model.wte.weight")
        ]
        state_dict[apply_prefix(prefix, "backpack.position_embeddings.weight")] = state_dict[
            apply_prefix(prefix, "backpack.gpt2_model.wpe.weight")
        ]
        return state_dict
```

Similarly, to load weights from the state dict, you'll need to implement `from_state_dict`.

The correctness of your implementation can be validated through serialization tests, which will be discussed in the next section.

## Write Tests
There are two types of tests that you should write for your model: unit tests and serialization tests.

### [Optional] Unit Tests
This is optional, but sometimes it helps to make unit tests for each of your modules. This way you can make sure that each module is working as expected and capture any surprises early on, before you test them end-to-end.

In a unit test, you test your module in the following aspects:
1. The forward pass is successful.
2. The output shape is correct.
3. The output value is correct.

The point 3 is the most important one, but it is also the most difficult one to implement. If you have a reference implementation in another framework, like Hugging Face, you can use it to test your implementation on the output consistency.
Here is a piece of example code to test a same module of Levanter against Hugging Face Transformers:

```python
x = random.normal(key, (1, seq_len))
x_torch = torch.from_numpy(np.array(x))  # convert to torch tensor

levanter_rope = LlamaRotaryEmbedding(HeadSize=HeadSize, Pos=Pos)
levanter_output = levanter_rope(seq_len=seq_len)
hf_rope = HFLlamaRotaryEmbedding(dim=hidden_dim, max_position_embeddings=seq_len, device="cpu")
hf_output = hf_rope(x_torch, seq_len=seq_len)

# compare the output values
for jax_out, torch_out in zip(levanter_output, hf_output):
    torch_out = torch_out.numpy()
    assert np.isclose(torch_out, np.array(jax_out.array), rtol=1e-4, atol=1e-4).all(), f"{torch_out} != {jax_out}"
```

Be sure to use `np.isclose` with reasonably loose tolerances to account for numerical differences.

For input variables that are common among tests, you can create a helper function to generate them.
For example, below is the helper function for unit tests in Llama:

```python
def _get_random_inputs(config: LlamaConfig):
    Embed = config.Embed
    Pos = config.Pos
    Batch = hax.Axis("batch", 2)
    x = hax.random.normal(random.PRNGKey(0), (Batch, Pos, Embed))
    mask = hax.nn.attention.causal_mask(config.Pos, config.KeyPos)
    return x, mask
```

If your module contains model weights, such consistency tests would require you to load weight from your module in Levanter to the reference module in HuggingFace. We will discuss how to do that in the next section.

### Serialization Tests

In order to validate the implementation on Levanter matches with the reference implementation in HuggingFace, we would like to pass the same input to both implementations and compare the output.
However, if the model are initialized with different weights, such comparison would not be meaningful. Therefore, we will need to serialize the weights from Hugging Face transformers and load them into Levanter, or vice versa. We call such test "serialization test".

For modules like Attention, Mlp, and Embeddings, you can read the weight from Levanter as state dict in memory and load into corresponding modules in HuggingFace. For example, in Llama, we did the following:

```python
# initialize the module in Levanter
import haliax

attention = LlamaAttention.init(config=config, key=random.PRNGKey(0))

# read the weights from Levanter
state = haliax.state_dict.to_torch_compatible_state_dict(attention.state_dict())
state = {k: torch.from_numpy(np.array(v)) for k, v in state.items()}

# load the weights into HuggingFace
hf_attention = HFLlamaAttention(config.to_hf_config(32000))
hf_attention.load_state_dict(state, strict=True)

# ...

# compare the output values
out = attention(x, mask)
hf_out = hf_attention(x_torch, mask_torch)

assert np.isclose(
    hf_out[0].detach().cpu().numpy(), np.array(outarray),
    rtol=1e-4, atol=1e-4
).all()
```

For the end-to-end model, you can save the model weight to disk as a checkpoint and load it into the reference model. For example, in Llama, we can do the following:

```python
config = LLamaConfig()
converter = config.default_hf_converter()

# initialize the model in HF...
hf_model = transformer.AutoModelForCausalLM(...)

# save the model weight to disk
with tempfile.TemporaryDirectory() as tmpdir:
    ck_path = f"{tmpdir}/hf_model"
    hf_model.save_pretrained(ck_path)

    model = converter.load_pretrained(config.model_type, ref=ck_path, resize_vocab_to_match_tokenizer=False)

# compare the output values between Levanter and HF
# ...
```

The serialization tests are very useful for testing the correctness of your implementation and make sure you can load your pretrained HuggingFace model into Levanter.

## Training
After you have implemented your model and validated it through tests, you can start training your model with Levanter.

### Write Training Configuration
To launch a training job, you will need to write a training configuration file in yaml. It includes the dataset, model, and trainer specifications for the training job. You can find many examples in [configs/](https://github.com/stanford-crfm/levanter/tree/main/config).

Under the model section, you will need to specify the model name as `type` and modify the hyperparameters that you would like to change. For parameters that are not specified, the default values will be used.

For example, the following configuration uses Llama with default hyperparameters:

```yaml
model:
    type: llama
```

To initialize with the model configuration and checkpoint from Hugging Face, you can specify the `initialize_from_hf` and `use_hf_model_config` parameters:

```yaml
model:
    type: llama
initialize_from_hf: "NousResearch/Llama-2-7b-hf"
use_hf_model_config: true
```

Sometimes it is helpful to start with a small model for debugging. For example, the configuration below specifies a very small Llama model with only 4M parameters. It can be run on a single chip with ease and is useful for debugging.

```yaml
model:
  type: llama
  hidden_dim: 32
  num_heads: 4
  num_layers: 2
```

For more details on the training configuration, please refer to [Configuration Guide](../reference/Configuration.md).

### Launch Training Job
Once you have your training configuration ready and your training environment set up, you can launch a training job with the following command:

```bash
# HUGGING_FACE_HUB_TOKEN is only needed if you want to load private checkpoint from Hugging Face
WANDB_API_KEY=$WANDB_API_KEY \
HUGGING_FACE_HUB_TOKEN=$HUGGING_FACE_HUB_TOKEN \
python levanter/src/levanter/main/train_lm.py --config_path $CONFIG_PATH
```

Check out [Training on Your Own Data](../Training-On-Your-Data.md) for more detailed guide on how to spin off a training cluster and launch a training job.

### Profile Your Model
If you are interested in profiling the training throughput of your model, good news is that it comes for free with automatic job monitoring in Levanter, powered through Weights & Biases.

Once you run a training job, on the corresponding job page on Weights & Biases, you will be able to find a section named "Throughput". It reports metrics like `examples_per_second` and `tokens_per_second` across the training time.
